#!/usr/bin/env bash
set -euo pipefail
python -m src.evaluation.eval_sweep_siglip_reg --help
python -m src.evaluation.score_generated_vs_original --help
python -m src.evaluation.score_generated_vs_original_bg --help
