"""
Feature engineering, built *per prediction checkpoint* to avoid leakage.

Checkpoints = the moment at which we make the prediction. Only information that a
control tower would genuinely know at that moment is allowed.

    T0_departure     : shipment has just left the origin
    T1_port_arrival  : vessel has reached the destination port
    T2_customs_clear : customs clearance done, only last-mile left

Missing intermediate observations are imputed with IterativeImputer (BayesianRidge),
which learns the stage-to-stage correlations (= delay propagation) from training rows only,
and a missing-flag is kept so the model knows the value was inferred.
"""
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

CHECKPOINTS = ["T0_departure", "T1_port_arrival", "T2_customs_clear"]
TS_COLS = ["sched_departure", "sched_port_arrival", "sched_customs_clear", "sched_delivery",
           "act_departure", "act_port_arrival", "act_customs_clear", "act_delivery"]

# which stage / "cause" each feature belongs to (used in explanations)
CAUSE = {
    "dep_delay": "Origin departure delay",
    "weather_origin": "Origin weather",
    "weather_sea": "Sea weather / transit",
    "sched_transit_h": "Route length",
    "route_hist": "Route baseline (historical)",
    "proj_clear_weekend": "Weekend clearance risk",
    "peak_season": "Peak season (Q4)",
    "port_arr_delay": "Late port arrival",
    "port_congestion": "Port congestion",
    "cong_x_arr_delay": "Congestion x late arrival",
    "late_slot_risk": "Missed customs slot (>12h late)",
    "customs_delay": "Customs stage delay",
    "customs_proc_h": "Customs processing time",
    "customs_excess": "Customs slower than schedule",
    "cong_x_proc": "Congestion x processing time",
    "flag_arr_missing": "Missing data (arrival)",
    "flag_cong_missing": "Missing data (congestion)",
    "flag_cust_missing": "Missing data (customs)",
    "flag_proc_missing": "Missing data (processing)",
}


def load(path):
    df = pd.read_csv(path)
    for c in TS_COLS:
        df[c] = pd.to_datetime(df[c])
    df["sched_transit_h"] = (df.sched_port_arrival - df.sched_departure).dt.total_seconds() / 3600
    return df


def _raw_block(df, cp):
    """Raw observed columns available at the checkpoint (NaN where missing)."""
    X = pd.DataFrame(index=df.index)
    X["dep_delay"] = df.delay_departure_h
    X["weather_origin"] = df.weather_origin
    X["weather_sea"] = df.weather_sea          # forecast, known at departure
    if cp in ("T1_port_arrival", "T2_customs_clear"):
        X["port_arr_delay"] = df.delay_port_arrival_h
        X["port_congestion"] = df.port_congestion_index
    if cp == "T2_customs_clear":
        X["customs_delay"] = df.delay_customs_h
        X["customs_excess"] = df.customs_processing_hours - df.sched_customs_hours
    return X


class Featurizer:
    """fit() on training rows only -> no information leaks from the test fold."""

    def __init__(self, cp, smoothing=10):
        self.cp, self.m = cp, smoothing

    def fit(self, df):
        raw = _raw_block(df, self.cp)
        self.cols_ = list(raw.columns)
        self.imp_ = IterativeImputer(max_iter=15, random_state=0, sample_posterior=False)
        self.imp_.fit(raw)
        self.gmean_ = df.final_delay_h.mean()
        g = df.groupby("route").final_delay_h.agg(["sum", "count"])
        self.route_enc_ = ((g["sum"] + self.m * self.gmean_) / (g["count"] + self.m)).to_dict()
        return self

    def transform(self, df):
        raw = _raw_block(df, self.cp)
        X = pd.DataFrame(self.imp_.transform(raw), columns=self.cols_, index=df.index)
        F = pd.DataFrame(index=df.index)
        F["dep_delay"] = X.dep_delay
        F["weather_origin"] = X.weather_origin
        F["weather_sea"] = X.weather_sea
        F["sched_transit_h"] = df.sched_transit_h
        F["route_hist"] = df.route.map(self.route_enc_).fillna(self.gmean_)
        F["peak_season"] = (df.sched_port_arrival.dt.month >= 10).astype(int)

        latest = X.dep_delay.copy()
        if "port_arr_delay" in X:
            latest = X.port_arr_delay
        if "customs_delay" in X:
            latest = X.customs_delay
        # projected clearance time = schedule + latest known delay carried forward
        proj = df.sched_customs_clear + pd.to_timedelta(latest, unit="h")
        F["proj_clear_weekend"] = (proj.dt.dayofweek >= 5).astype(int)

        if self.cp in ("T1_port_arrival", "T2_customs_clear"):
            F["port_arr_delay"] = X.port_arr_delay
            F["port_congestion"] = X.port_congestion.clip(0, 1)
            F["cong_x_arr_delay"] = F.port_congestion * X.port_arr_delay.clip(lower=0)
            F["late_slot_risk"] = (X.port_arr_delay > 12).astype(int)
            F["flag_arr_missing"] = raw.port_arr_delay.isna().astype(int)
            F["flag_cong_missing"] = raw.port_congestion.isna().astype(int)
        if self.cp == "T2_customs_clear":
            F["customs_delay"] = X.customs_delay
            F["customs_excess"] = X.customs_excess
            F["customs_proc_h"] = X.customs_excess + df.sched_customs_hours
            F["cong_x_proc"] = F.port_congestion * F.customs_proc_h
            F["flag_cust_missing"] = raw.customs_delay.isna().astype(int)
            F["flag_proc_missing"] = raw.customs_excess.isna().astype(int)
        return F

    def fit_transform(self, df):
        return self.fit(df).transform(df)
