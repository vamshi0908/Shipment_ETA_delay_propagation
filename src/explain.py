"""
Explanations at three levels

 A. Structural propagation model  : d_k = a_k + rho_k * d_{k-1} + e_k   (estimated from data)
    -> exact root-cause split of a shipment's final delay into the part that ORIGINATED at each stage
       and was then carried downstream. Contributions sum exactly to the final delay.
 B. Global model importance       : permutation importance of the gradient-boosting model.
 C. Local model explanation       : per-shipment additive contributions from the Ridge model
       (coef * (x - mean)), grouped into human-readable causes.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import data_gen
from data_gen import DELAY_THRESHOLD_H
from features import CAUSE, CHECKPOINTS, Featurizer, load

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
STAGES = ["Origin departure", "Sea transit / port arrival", "Customs & port processing", "Last mile"]
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#999999"]


# ---------------------------------------------------------------- A. propagation model
def fit_propagation(df):
    """OLS of each stage delay on the previous one, using rows where both are observed."""
    pairs = [("delay_departure_h", None), ("delay_port_arrival_h", "delay_departure_h"),
             ("delay_customs_h", "delay_port_arrival_h"), ("final_delay_h", "delay_customs_h")]
    coefs = []
    for cur, prev in pairs:
        if prev is None:
            coefs.append((float(df[cur].mean()), 0.0))
            continue
        d = df[[cur, prev]].dropna()
        rho, a = np.polyfit(d[prev], d[cur], 1)
        coefs.append((float(a), float(rho)))
    return coefs  # [(a0,0),(a1,rho1),(a2,rho2),(a3,rho3)]


def decompose(row, coefs):
    """Exact decomposition of final delay into baseline + origin of delay at each stage."""
    (a0, _), (a1, r1), (a2, r2), (a3, r3) = coefs
    d0, d1, d2, d3 = (row.delay_departure_h, row.delay_port_arrival_h,
                      row.delay_customs_h, row.final_delay_h)
    e0, e1, e2, e3 = d0 - a0, d1 - a1 - r1 * d0, d2 - a2 - r2 * d1, d3 - a3 - r3 * d2
    contrib = [r3 * r2 * r1 * e0, r3 * r2 * e1, r3 * e2, e3]
    base = a3 + r3 * (a2 + r2 * (a1 + r1 * a0))
    return base, contrib


# ---------------------------------------------------------------- helpers for C
def fit_cp(train, cp):
    fz = Featurizer(cp).fit(train)
    X = fz.transform(train)
    reg = make_pipeline(StandardScaler(), Ridge(alpha=3.0)).fit(X, train.final_delay_h)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000)).fit(X, train.delayed)
    return fz, reg, clf


def local_explain(row_df, fz, reg, clf):
    X = fz.transform(row_df)
    sc, rg = reg.named_steps["standardscaler"], reg.named_steps["ridge"]
    contrib = rg.coef_ * (X.values[0] - sc.mean_) / sc.scale_
    s = pd.Series(contrib, index=X.columns)
    grouped = s.groupby(lambda f: CAUSE.get(f, f)).sum().sort_values(key=np.abs, ascending=False)
    return dict(pred=float(reg.predict(X)[0]), base=float(rg.intercept_), p_delay=float(clf.predict_proba(X)[0, 1]),
                contrib=grouped)


def fmt_example(row, cp, ex):
    eta = row.sched_delivery + pd.Timedelta(hours=ex["pred"])
    lines = [f"**{row.shipment_id}** | {row.route} | checkpoint `{cp}`",
             f"- Scheduled delivery: {row.sched_delivery:%Y-%m-%d %H:%M} | **Predicted ETA: {eta:%Y-%m-%d %H:%M} "
             f"({ex['pred']:+.1f} h)** | P(delay > {DELAY_THRESHOLD_H:.0f} h) = {ex['p_delay']:.2f} | "
             f"actual delay: {row.final_delay_h:+.1f} h",
             f"- Average shipment delay (model baseline): {ex['base']:+.1f} h. Main drivers of this prediction:"]
    for cause, v in ex["contrib"].head(5).items():
        lines.append(f"  - {cause}: {v:+.1f} h")
    return "\n".join(lines)


def main():
    df = load(ROOT / "data" / "shipments.csv")
    truth = {"RHO1": data_gen.RHO1, "RHO2": data_gen.RHO2, "RHO3": data_gen.RHO3}

    # ---- A. propagation coefficients + decomposition ----
    coefs = fit_propagation(df)
    est = {"rho1": coefs[1][1], "rho2": coefs[2][1], "rho3": coefs[3][1]}
    print("Estimated propagation coefficients:", {k: round(v, 3) for k, v in est.items()},
          "| true direct:", truth)
    complete = df.dropna(subset=["delay_port_arrival_h", "delay_customs_h"]).copy()
    rows = []
    for _, r in complete.iterrows():
        base, c = decompose(r, coefs)
        assert abs(base + sum(c) - r.final_delay_h) < 1e-6
        rows.append([base] + c)
    contrib = pd.DataFrame(rows, columns=["Baseline"] + STAGES, index=complete.index)
    contrib["delayed"] = complete.delayed
    contrib["route"] = complete.route
    by_status = contrib.groupby("delayed")[["Baseline"] + STAGES].mean()
    by_route = contrib.groupby("route")[STAGES].mean()
    by_route["total"] = by_route.sum(axis=1)

    # ---- split used for model-based explanations ----
    train, test = train_test_split(df, test_size=0.2, random_state=7, stratify=df.delayed)

    # ---- B. global permutation importance (GBM) at T1 and T2 ----
    imp = {}
    for cp in ["T1_port_arrival", "T2_customs_clear"]:
        fz = Featurizer(cp).fit(train)
        Xtr, Xte = fz.transform(train), fz.transform(test)
        gb = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                       subsample=0.8, random_state=0).fit(Xtr, train.final_delay_h)
        pi = permutation_importance(gb, Xte, test.final_delay_h, n_repeats=30, random_state=0,
                                    scoring="neg_mean_absolute_error")
        s = pd.Series(pi.importances_mean, index=Xte.columns).sort_values(ascending=False)
        imp[cp] = s

    # ---- C. local explanations ----
    models = {cp: fit_cp(train, cp) for cp in CHECKPOINTS}
    test_complete = test.dropna(subset=["delay_port_arrival_h", "delay_customs_h"])
    worst = test_complete.sort_values("final_delay_h", ascending=False).iloc[0]
    mid = test_complete.iloc[(test_complete.final_delay_h - test_complete.final_delay_h.median()).abs().argsort().iloc[0]]
    best = test_complete.sort_values("final_delay_h").iloc[0]

    md = ["# Example per-shipment explanations (held-out shipments)\n",
          f"Delayed = final delay > {DELAY_THRESHOLD_H:.0f} h. 'Main drivers' = additive Ridge contributions "
          "relative to the average shipment, grouped into human-readable causes. "
          "'Root cause' = structural propagation split (sums exactly to the actual final delay).\n",
          f"## 1. Same shipment, three checkpoints ({worst.shipment_id}) - the explanation sharpens as data arrives\n"]
    for cp in CHECKPOINTS:
        fz, reg, clf = models[cp]
        md.append(fmt_example(worst, cp, local_explain(worst.to_frame().T.infer_objects(), fz, reg, clf)))
        md.append("")
    md.append("## 2. Other held-out shipments at the customs checkpoint\n")
    for r in (mid, best):
        fz, reg, clf = models["T2_customs_clear"]
        md.append(fmt_example(r, "T2_customs_clear", local_explain(r.to_frame().T.infer_objects(), fz, reg, clf)))
        md.append("")
    md.append("## 3. Root-cause propagation breakdown (actual delays, exact)\n")
    for r in (worst, mid, best):
        base, c = decompose(r, coefs)
        parts = ", ".join(f"{n}: {v:+.1f} h" for n, v in zip(STAGES, c))
        md.append(f"- **{r.shipment_id}** ({r.route}) final delay {r.final_delay_h:+.1f} h = baseline {base:+.1f} h; {parts}")
    (OUT / "example_explanations.md").write_text("\n".join(md))

    # ---------------- figures ----------------
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    # Fig 1: where the delay originates (deviation from the average shipment, after propagation)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    x = np.arange(len(STAGES)); w = 0.38
    ax.bar(x - w / 2, by_status.loc[0, STAGES].values, w, label="On-time", color="#55A868")
    ax.bar(x + w / 2, by_status.loc[1, STAGES].values, w, label=f"Delayed (>{DELAY_THRESHOLD_H:.0f} h)", color="#C44E52")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([s_.replace(" / ", "/\n").replace(" & ", " &\n") for s_ in STAGES], fontsize=7.5)
    ax.set_ylabel("Contribution to final delay (h)")
    ax.set_title("Where final delay originates (propagated to delivery)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "fig_stage_origin.png", dpi=200); plt.close(fig)

    # Fig 2: permutation importance
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, (cp, s) in zip(axes, imp.items()):
        top = s.head(6)[::-1]
        ax.barh([CAUSE.get(i, i) for i in top.index], top.values, color="#4C72B0")
        ax.set_title(cp.replace("_", " "), fontsize=9)
        ax.set_xlabel("MAE increase (h)")
    fig.suptitle("Global permutation importance (gradient boosting)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig_importance.png", dpi=200); plt.close(fig)

    # Fig 3: MAE by checkpoint
    m = json.loads((OUT / "metrics.json").read_text())["checkpoints"]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    names = ["Schedule (ETA=plan)", "Route mean delay", "Ridge", "GradBoost"]
    w = 0.2
    for i, nm in enumerate(names):
        mu = [m[cp][nm]["MAE"][0] for cp in CHECKPOINTS]
        sd = [m[cp][nm]["MAE"][1] for cp in CHECKPOINTS]
        ax.bar(np.arange(3) + (i - 1.5) * w, mu, w, yerr=sd, label=nm, capsize=2,
               color=["#BBBBBB", "#8C8C8C", "#4C72B0", "#C44E52"][i])
    ax.set_xticks(range(3)); ax.set_xticklabels(["T0 departure", "T1 port arrival", "T2 customs clear"])
    ax.set_ylabel("ETA error, MAE (h)"); ax.set_title("ETA error falls as the shipment progresses")
    ax.set_ylim(0, 21); ax.legend(fontsize=7, frameon=False, loc="upper center", ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "fig_mae.png", dpi=200); plt.close(fig)

    # Fig 4: local explanation of the worst shipment at T2 (+ root-cause)
    fz, reg, clf = models["T2_customs_clear"]
    ex = local_explain(worst.to_frame().T.infer_objects(), fz, reg, clf)
    top = ex["contrib"].head(6)[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.barh(top.index, top.values, color=["#C44E52" if v > 0 else "#55A868" for v in top.values])
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Contribution to predicted delay vs average (h)")
    ax.set_title(f"{worst.shipment_id}: predicted {ex['pred']:+.1f} h, actual {worst.final_delay_h:+.1f} h", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig_local_example.png", dpi=200); plt.close(fig)

    (OUT / "propagation.json").write_text(json.dumps(dict(
        estimated=est, true_direct=truth,
        by_status=by_status.round(2).to_dict(), by_route=by_route.round(2).to_dict(),
        worst=worst.shipment_id,
        importance={k: v.head(6).round(3).to_dict() for k, v in imp.items()}), indent=2))
    print(by_status.round(1)); print(by_route.round(1))
    print("\n".join(md[:14]))


if __name__ == "__main__":
    main()
