"""
Train + evaluate ETA / delay models at each checkpoint.

Tasks
  * Regression : predict final delay (hours).  Predicted ETA = scheduled delivery + predicted delay.
  * Classification : P(delayed) where delayed = final delay > DELAY_THRESHOLD_H.

Evaluation: repeated 5-fold cross-validation (3 repeats). With only ~240 shipments a single
80/20 split is noisy, so we report the mean +- std over folds. Everything that learns from data
(imputer, route encoding, scaler, model) is fit on the training fold only.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_gen import DELAY_THRESHOLD_H
from features import CHECKPOINTS, Featurizer, load

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]


def make_models():
    return {
        "reg": {
            "Ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=3.0)),
            "GradBoost": lambda: GradientBoostingRegressor(
                n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0),
        },
        "clf": {
            "Logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000)),
            "GradBoost": lambda: GradientBoostingClassifier(
                n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0),
        },
    }


def cross_validate(df, cp, n_splits=5, n_repeats=3, seed=0):
    y = df.final_delay_h.values
    yb = df.delayed.values
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    res = {}
    models = make_models()

    def add(name, key, val):
        res.setdefault(name, {}).setdefault(key, []).append(val)

    for tr, te in rkf.split(df):
        dtr, dte = df.iloc[tr], df.iloc[te]
        fz = Featurizer(cp).fit(dtr)
        Xtr, Xte = fz.transform(dtr), fz.transform(dte)

        # --- baselines ---
        add("Schedule (ETA=plan)", "MAE", mean_absolute_error(y[te], np.zeros(len(te))))
        add("Route mean delay", "MAE", mean_absolute_error(y[te], Xte.route_hist))
        add("Route mean delay", "AUC", roc_auc_score(yb[te], Xte.route_hist))

        for name, mk in models["reg"].items():
            m = mk().fit(Xtr, y[tr])
            add(name, "MAE", mean_absolute_error(y[te], m.predict(Xte)))
        for name, mk in models["clf"].items():
            c = mk().fit(Xtr, yb[tr])
            p = c.predict_proba(Xte)[:, 1]
            add(name + " (clf)", "AUC", roc_auc_score(yb[te], p))
            add(name + " (clf)", "F1", f1_score(yb[te], p > 0.5))
            add(name + " (clf)", "Acc", accuracy_score(yb[te], p > 0.5))

    return {k: {mk: (float(np.mean(v)), float(np.std(v))) for mk, v in d.items()} for k, d in res.items()}


def main():
    df = load(ROOT / "data" / "shipments.csv")
    out = {"threshold_h": DELAY_THRESHOLD_H, "n": len(df), "delayed_rate": float(df.delayed.mean()),
           "checkpoints": {}}
    for cp in CHECKPOINTS:
        r = cross_validate(df, cp)
        out["checkpoints"][cp] = r
        print(f"\n=== {cp} ===")
        for name, d in r.items():
            print(f"  {name:22s} " + "  ".join(f"{k}={m:.2f}±{s:.2f}" for k, (m, s) in d.items()))
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "metrics.json").write_text(json.dumps(out, indent=2))
    print("\nSaved outputs/metrics.json")


if __name__ == "__main__":
    main()
