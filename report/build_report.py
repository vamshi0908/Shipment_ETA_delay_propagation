"""Builds the 3-page technical report PDF from the numbers/figures produced by the pipeline."""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
M = json.loads((OUT / "metrics.json").read_text())
P = json.loads((OUT / "propagation.json").read_text())
CPS = ["T0_departure", "T1_port_arrival", "T2_customs_clear"]
ex_lines = (OUT / "example_explanations.md").read_text()

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=8.6, leading=11.2, spaceAfter=3)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, spaceAfter=2, alignment=1)
sub = ParagraphStyle("sub", parent=body, alignment=1, textColor=colors.grey, fontSize=8.5)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=10.5, spaceBefore=5, spaceAfter=2, textColor=colors.HexColor("#1F3A5F"))
bul = ParagraphStyle("bul", parent=body, leftIndent=10, bulletIndent=2, spaceAfter=1.5)
cap = ParagraphStyle("cap", parent=body, fontSize=7.5, textColor=colors.grey, alignment=1)


def B(items):
    return [Paragraph(t, bul, bulletText="\u2022") for t in items]


def img(name, w):
    from PIL import Image as PI
    iw, ih = PI.open(OUT / name).size
    return Image(str(OUT / name), width=w, height=w * ih / iw)


def mval(cp, model, metric, d=2):
    m, s = M["checkpoints"][cp][model][metric]
    return f"{m:.{d}f} \u00b1 {s:.{d}f}"


est = P["estimated"]
chain_eff = est["rho1"] * est["rho2"] * est["rho3"]
chain_dir = 0.75 * 0.90 * 0.80

story = []
story += [Paragraph("Predicting Shipment ETA and Delay Propagation", h1),
          Paragraph("Technical report - Aiqon AI Internship take-home assignment", sub), Spacer(1, 4)]

# 1 ------------------------------------------------------------------
story += [Paragraph("1. Problem framing and definitions", h2),
          Paragraph("A shipment passes four stages: <b>S0</b> origin departure, <b>S1</b> destination-port arrival, "
                    "<b>S2</b> customs clearance, <b>S3</b> final delivery. Each stage k has a scheduled and an actual timestamp. "
                    "I define <b>stage delay</b> d<sub>k</sub> = actual - scheduled (hours, negative = early), "
                    "<b>final delay</b> = d<sub>3</sub>, and <b>predicted ETA</b> = scheduled delivery + predicted final delay. "
                    f"A shipment is <b>delayed</b> if its final delay exceeds {M['threshold_h']:.0f} h (half a day, an assumption). "
                    "The system answers the three required questions: (1) ETA via regression, (2) delay magnitude (regression) and "
                    "likelihood (classifier), (3) why, via three complementary explanations (Section 5).", body),
          Paragraph("<b>Key design idea - predict at checkpoints.</b> A control tower must predict while the shipment is in motion, so "
                    "I build one model per checkpoint (T0 just departed, T1 at the destination port, T2 customs cleared) and only allow "
                    "information that is known at that moment. This avoids target leakage and shows how forecast quality improves as "
                    "delays materialise.", body)]

# 2 ------------------------------------------------------------------
story += [Paragraph("2. Synthetic data and how ground truth was defined", h2),
          Paragraph(f"{M['n']} shipments on 6 routes (e.g. Shanghai-Los Angeles, Mumbai-Singapore) were simulated with scheduled/actual "
                    "timestamps per stage, customs processing time, port-congestion index and weather indices. Delays are created by an "
                    "explicit mechanism (code: <font face='Courier'>src/data_gen.py</font>):", body)]
story += B([
    "<b>Propagation:</b> d<sub>1</sub> = 0.75 d<sub>0</sub> + noise; d<sub>2</sub> = 0.90 d<sub>1</sub> + (processing excess) + missed-slot penalty; "
    "d<sub>3</sub> = 0.80 d<sub>2</sub> + weekend gap + noise (vessels/trucks recover part of an upstream delay; the rest is carried on).",
    "<b>Congestion x processing:</b> late vessels bunch up, raising port congestion (+0.012 per hour late); customs processing time grows with "
    "congestion <i>above the port's normal level</i>, because schedules already assume normal congestion.",
    "<b>Real-world effects:</b> a vessel more than 12 h late misses its customs slot (+8 h); clearance on a weekend adds 10 h to the last mile; "
    "Q4 peak season raises congestion; routes differ in reliability (risk multiplier 0.7-1.4) and port norms.",
    "<b>Missing data:</b> after simulation, port-arrival timestamp (9%), customs timestamp (12%), processing time (7%) and congestion (5%) "
    "were randomly blanked. The complete truth file is kept to verify the mechanism.",
    f"<b>Ground truth:</b> final delay = actual - scheduled delivery; label delayed = final delay > {M['threshold_h']:.0f} h "
    f"({M['delayed_rate']:.0%} of shipments).",
])

# 3 ------------------------------------------------------------------
story += [Paragraph("3. Features (stage interactions rather than isolated stages)", h2)]
story += B([
    "<b>Stage delays</b> d<sub>0</sub>, d<sub>1</sub>, d<sub>2</sub> available at each checkpoint (cumulative, so they encode propagation); customs processing excess vs plan.",
    "<b>Interactions:</b> congestion x late-arrival, congestion x customs processing time, missed-slot flag (arrival > 12 h late).",
    "<b>Route/temporal:</b> smoothed historical mean delay of the route (fit on training folds only), route length, Q4 peak flag, "
    "and a <i>projected weekend-clearance</i> flag (scheduled clearance + latest known delay falls on Sat/Sun).",
    "<b>Missing data:</b> IterativeImputer (BayesianRidge) fitted on training rows only, so it uses stage-to-stage correlation to infer a "
    "missing stage delay; a missing-flag is kept so the model knows the value was inferred.",
])

# 4 ------------------------------------------------------------------
story += [Paragraph("4. Models and evaluation", h2),
          Paragraph("Regression: <b>Ridge</b> (interpretable, additive) and <b>gradient boosting</b> (captures non-linear thresholds). "
                    "Classification: logistic regression and gradient boosting. Baselines: <b>schedule</b> (ETA = plan, predicts zero delay) and "
                    "<b>route mean delay</b>. With only 240 rows a single split is noisy, so I report mean \u00b1 std over "
                    "<b>5-fold cross-validation repeated 3 times</b>; imputer, route encoding, scaler and model are all fitted inside each training fold. "
                    "Metrics: MAE in hours of the ETA; AUC / F1 for delay detection.", body)]

# 5 results ----------------------------------------------------------
rows = [["Model (CV mean \u00b1 std)", "T0 departure", "T1 port arrival", "T2 customs clear"]]
for label, mdl, met in [("ETA MAE (h): schedule baseline", "Schedule (ETA=plan)", "MAE"),
                        ("ETA MAE (h): route-mean baseline", "Route mean delay", "MAE"),
                        ("ETA MAE (h): Ridge", "Ridge", "MAE"),
                        ("ETA MAE (h): Gradient boosting", "GradBoost", "MAE"),
                        ("Delay AUC: logistic", "Logistic (clf)", "AUC"),
                        ("Delay F1: logistic", "Logistic (clf)", "F1"),
                        ("Delay AUC: gradient boosting", "GradBoost (clf)", "AUC")]:
    rows.append([label] + [mval(cp, mdl, met) for cp in CPS])
tbl = Table(rows, colWidths=[6.2 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm])
tbl.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7.8), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                         ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF6")),
                         ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                         ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
r0 = M["checkpoints"]["T0_departure"]["Ridge"]["MAE"][0]
r2 = M["checkpoints"]["T2_customs_clear"]["Ridge"]["MAE"][0]
sch = M["checkpoints"]["T0_departure"]["Schedule (ETA=plan)"]["MAE"][0]
story += [Paragraph("5. Results", h2), tbl, Spacer(1, 3),
          Paragraph(f"Using the plain schedule as ETA is off by <b>{sch:.1f} h</b> on average. Already at departure the Ridge model cuts this to "
                    f"<b>{r0:.1f} h</b>, and once customs has cleared to <b>{r2:.1f} h</b>, with AUC for the delayed/on-time call rising from "
                    f"{M['checkpoints']['T0_departure']['Logistic (clf)']['AUC'][0]:.2f} to {M['checkpoints']['T2_customs_clear']['Logistic (clf)']['AUC'][0]:.2f}. "
                    "Ridge is slightly better than gradient boosting: the simulated mechanism is close to linear and the sample is small, so the simpler model "
                    "generalises better - an honest finding, not a claim that linear models are always superior on real data.", body)]
story += [Table([[img("fig_mae.png", 8.2 * cm), img("fig_stage_origin.png", 8.2 * cm)]], colWidths=[8.8 * cm, 8.8 * cm]),
          Paragraph("Figure 1: ETA error by checkpoint (left); where final delay originates after propagation (right).", cap)]

# 6 explanations -----------------------------------------------------
story += [Paragraph("6. Explanations: which stage / factor caused the delay", h2)]
story += B([
    "<b>(A) Structural propagation model.</b> I estimate d<sub>k</sub> = a<sub>k</sub> + &rho;<sub>k</sub> d<sub>k-1</sub> + e<sub>k</sub> from observed stage pairs. "
    "The new delay e<sub>k</sub> created at stage k reaches delivery multiplied by the downstream &rho;'s, and the pieces plus a baseline sum "
    "<i>exactly</i> to the final delay. This gives a root-cause statement such as \"+9 h of the final delay originated in customs/port processing\".",
    f"<b>Estimated propagation:</b> &rho;<sub>1</sub>={est['rho1']:.2f}, &rho;<sub>2</sub>={est['rho2']:.2f}, &rho;<sub>3</sub>={est['rho3']:.2f}. "
    f"&rho;<sub>1</sub> and &rho;<sub>3</sub> are close to the built-in values (0.75, 0.80). &rho;<sub>2</sub> is far above the built-in direct 0.90 because the regression "
    "captures the <i>total</i> effect of a late arrival on customs, including the congestion-processing path and the missed-slot penalty. In other words, the "
    f"customs stage <b>amplifies</b> late arrivals; an hour of origin delay ends up as ~{chain_eff:.1f} h at delivery versus {chain_dir:.2f} h if propagation were purely direct.",
    "<b>(B) Global importance</b> (permutation, gradient boosting): at port arrival the top driver is the <i>congestion x late-arrival</i> interaction; "
    "after customs, the customs-stage delay and the weekend-clearance flag dominate (Figure 2). Importance of correlated stage delays is split between them, "
    "so small bars for upstream delays do not mean they are harmless - panel (A) quantifies their propagated impact.",
    "<b>(C) Local explanation</b> (Ridge): each prediction is an additive sum of cause contributions in hours versus an average shipment (Figure 3).",
])
story += [Table([[img("fig_importance.png", 9.6 * cm), img("fig_local_example.png", 7.6 * cm)]], colWidths=[10 * cm, 7.8 * cm]),
          Paragraph("Figure 2: global importance at T1 and T2 (left). Figure 3: local explanation for one held-out shipment (right).", cap)]

# example text
import re
m_ = re.search(r"\*\*(SH\d+)\*\* \| ([^|]+) \| checkpoint `T2_customs_clear`\n- (.*)\n- .*\n((?:  - .*\n?){5})", ex_lines)
if m_:
    drivers = "; ".join(l.strip()[2:] for l in m_.group(4).strip().split("\n")[:4])
    clean = lambda t: t.replace("**", "")
    story += [Paragraph(f"<b>Example output (held-out shipment {m_.group(1)}, {m_.group(2).strip()}, at customs):</b> {clean(m_.group(3))}. "
                        f"Main drivers: {drivers}.", body)]

# 7 ------------------------------------------------------------------
story += [Paragraph("7. Assumptions, limitations, next steps", h2)]
story += B([
    "<b>Synthetic data:</b> metrics show the pipeline recovers a known mechanism; they are not evidence of real-world accuracy. The 12 h threshold and all "
    "coefficients are assumptions. Real data (AIS, port call logs) would add vessel speed, berth queues, tides and strikes.",
    "<b>Missing data</b> was random (MCAR). Real gaps are often systematic (e.g. missing when a port is overloaded), which needs explicit modelling.",
    "<b>Explanations are associative:</b> panel (A) estimates effective propagation from observational data and cannot separate direct from mediated effects without "
    "the causal structure; correlated features make attributions share credit.",
    "<b>Next steps:</b> quantile / conformal intervals for ETA, a per-port hierarchical model for processing times, hazard-style (survival) models for clearance time, "
    "and validation on a public maritime dataset.",
    "<b>Reproduce:</b> <font face='Courier'>pip install -r requirements.txt; bash run_all.sh</font> (all seeds fixed; runtime under a minute).",
])


def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 7); c.setFillColor(colors.grey)
    c.drawCentredString(A4[0] / 2, 0.8 * cm, f"Aiqon take-home - ETA & delay propagation - page {d.page}")
    c.restoreState()


doc = SimpleDocTemplate(str(ROOT / "report" / "Technical_Report.pdf"), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                        topMargin=1.4 * cm, bottomMargin=1.5 * cm, title="Predicting Shipment ETA and Delay Propagation",
                        author="Vamshi Krishna")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("Report written")
