"""
Synthetic shipment-journey generator.

Stages (hours are always measured relative to the SCHEDULED timestamp):
    S0 origin departure -> S1 destination port arrival -> S2 customs clearance -> S3 final delivery

Ground-truth delay mechanism (this is what the report calls "how delays were introduced"):

    d0 = route_risk * Gamma(1.2, 3) + 6 * weather_origin + N(0, 1)                (own delay at origin)
    d1 = RHO1 * d0 + route_risk * N(0, 3) + 14 * weather_sea * (transit/400)      (ships recover ~25% of departure delay)
    congestion = route_base_congestion + 0.012 * max(d1, 0) + seasonal + N(0, .1)  (late vessels bunch up at the port)
    proc_time  = sched_customs_h * (1 + 1.5 * (congestion - port_norm)) * LogNormal(0, .15)  (congestion x processing interaction;
                 schedules assume 'normal' congestion for that port, so only ABOVE-normal congestion slows customs)
    d2 = RHO2 * d1 + (proc_time - sched_customs_h) + 8 * [d1 > 12]                  (missed customs slot if arrival is very late)
    d3 = RHO3 * d2 + 10 * [customs clears on weekend] + route_risk * N(0, 3)        (last-mile, weekend trucking gap)

    final_delay = d3  (actual delivery - scheduled delivery)
    delayed     = final_delay > DELAY_THRESHOLD_H

Missing data: intermediate observations are randomly blanked out AFTER simulation
(port arrival timestamp, customs timestamp, customs processing time, congestion index).
The complete ground truth is also saved so imputation quality can be checked.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RHO1, RHO2, RHO3 = 0.75, 0.90, 0.80
DELAY_THRESHOLD_H = 12.0

# origin, destination, scheduled transit (h), route risk multiplier, base congestion at dest port,
# scheduled customs hours, scheduled last-mile hours
ROUTES = [
    ("Shanghai",  "Los Angeles", 336, 1.4, 0.55, 30, 48),
    ("Shanghai",  "Rotterdam",   624, 1.0, 0.35, 20, 36),
    ("Mumbai",    "Rotterdam",   456, 1.2, 0.35, 20, 36),
    ("Mumbai",    "Singapore",   240, 0.7, 0.25, 12, 24),
    ("Singapore", "Sydney",      288, 0.9, 0.40, 36, 30),
    ("Hamburg",   "New York",    264, 1.1, 0.45, 28, 40),
]


def generate(n=240, seed=42, missing_rate=0.10):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        o, d, transit, risk, base_cong, cust_h, last_h = ROUTES[rng.integers(len(ROUTES))]
        sched_dep = pd.Timestamp("2025-01-01") + pd.Timedelta(hours=float(rng.integers(0, 365 * 24)))
        sched_arr = sched_dep + pd.Timedelta(hours=transit)
        sched_cus = sched_arr + pd.Timedelta(hours=cust_h)
        sched_del = sched_cus + pd.Timedelta(hours=last_h)

        w_org, w_sea = rng.beta(2, 5), rng.beta(2, 5)
        d0 = risk * rng.gamma(1.2, 3.0) + 6 * w_org + rng.normal(0, 1)
        d1 = RHO1 * d0 + risk * rng.normal(0, 3) + 14 * w_sea * (transit / 400)

        peak = 0.10 if sched_arr.month in (10, 11, 12) else 0.0
        cong = float(np.clip(base_cong + 0.012 * max(d1, 0) + peak + rng.normal(0, 0.10), 0, 1))
        proc = cust_h * max(0.5, 1 + 1.5 * (cong - base_cong)) * rng.lognormal(0, 0.15)
        d2 = RHO2 * d1 + (proc - cust_h) + (8.0 if d1 > 12 else 0.0)

        act_cus = sched_cus + pd.Timedelta(hours=d2)
        weekend = int(act_cus.dayofweek >= 5)
        d3 = RHO3 * d2 + 10 * weekend + risk * rng.normal(0, 3)

        rows.append(dict(
            shipment_id=f"SH{i:04d}", origin=o, destination=d, route=f"{o}->{d}",
            sched_departure=sched_dep, sched_port_arrival=sched_arr,
            sched_customs_clear=sched_cus, sched_delivery=sched_del,
            act_departure=sched_dep + pd.Timedelta(hours=d0),
            act_port_arrival=sched_arr + pd.Timedelta(hours=d1),
            act_customs_clear=act_cus,
            act_delivery=sched_del + pd.Timedelta(hours=d3),
            sched_customs_hours=cust_h, customs_processing_hours=proc,
            port_congestion_index=cong, weather_origin=w_org, weather_sea=w_sea,
            delay_departure_h=d0, delay_port_arrival_h=d1, delay_customs_h=d2, final_delay_h=d3,
        ))
    full = pd.DataFrame(rows)
    full["delayed"] = (full.final_delay_h > DELAY_THRESHOLD_H).astype(int)

    # ---- inject missing intermediate data (observed file) ----
    obs = full.copy()
    def blank(cols, rate):
        m = rng.random(len(obs)) < rate
        obs.loc[m, cols] = np.nan
    blank(["act_port_arrival", "delay_port_arrival_h"], missing_rate)
    blank(["act_customs_clear", "delay_customs_h"], missing_rate)
    blank(["customs_processing_hours"], 0.08)
    blank(["port_congestion_index"], 0.05)
    return full, obs


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data"
    out.mkdir(exist_ok=True)
    full, obs = generate()
    full.to_csv(out / "shipments_truth.csv", index=False)
    obs.to_csv(out / "shipments.csv", index=False)
    print(f"Saved {len(obs)} shipments | delayed rate = {full.delayed.mean():.2%} | "
          f"mean final delay = {full.final_delay_h.mean():.1f} h")
    print("Missing share per column:\n", obs.isna().mean()[lambda s: s > 0].round(3).to_string())
