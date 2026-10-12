#!/usr/bin/env bash
# Reproduces everything: data -> models/metrics -> explanations/figures -> PDF report
set -e
cd "$(dirname "$0")/src"
python data_gen.py
python train_eval.py
python explain.py
python ../report/build_report.py
echo "Done. See outputs/ and report/Technical_Report.pdf"
