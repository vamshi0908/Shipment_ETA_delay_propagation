# Shipment ETA & Delay Propagation (Aiqon take-home)

Predicts the final ETA of a shipment, the likelihood/magnitude of delay, and explains which stage or factor caused it.

## Run (under a minute)
```bash
pip install -r requirements.txt
bash run_all.sh
```
This runs, in order: `data_gen.py` (synthetic data) -> `train_eval.py` (CV metrics) -> `explain.py` (explanations + figures) -> `report/build_report.py` (PDF).

## Layout
| Path | Purpose |
|---|---|
| `src/data_gen.py` | Simulates 240 shipments with an explicit delay-propagation mechanism and injected missing data |
| `src/features.py` | Checkpoint-aware features (no leakage), interactions, propagation-aware imputation |
| `src/train_eval.py` | Ridge / gradient boosting / logistic models vs baselines, repeated 5-fold CV -> `outputs/metrics.json` |
| `src/explain.py` | Propagation model (root cause), permutation importance, per-shipment explanations, figures |
| `report/` | 2-page technical report (`Technical_Report.pdf`) and its builder |
| `data/` | `shipments.csv` (observed, with gaps), `shipments_truth.csv` (complete ground truth) |
| `outputs/` | Metrics, figures, `example_explanations.md` |

## Definitions
- Stage delay = actual - scheduled timestamp (hours). Final delay = actual - scheduled delivery.
- Predicted ETA = scheduled delivery + predicted final delay. Delayed = final delay > 12 h.
- Checkpoints: T0 departure, T1 port arrival, T2 customs cleared. Each uses only information known at that time.

## Explanations
1. **Propagation model**: d_k = a_k + rho_k * d_(k-1) + e_k. Exact root-cause split of the final delay by stage.
2. **Permutation importance** (global) from gradient boosting.
3. **Additive Ridge contributions** (per shipment), grouped into human-readable causes.

## Caveats
Data is synthetic, so metrics show the method recovers a known mechanism, not real-world accuracy. All assumptions are listed in the report.
