# robust-fashionability-enhancement

Public release-ready reproducibility repository for manuscript revision.

## Paper

- Title: Robust Fashionability Enhancement under Minimal Image Editing
- Authors: Qice Qin, Ryotaro Shimizu, Yuki Hirakawa, Edgar Simo-Serra
- Venue: The Visual Computer (under review)


### pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### conda

```bash
conda env create -f environment.yml
conda activate robust-fashionability-enhancement
```

## Evaluation Instructions

```bash
bash scripts/run_evaluation.sh
```

Or run directly:

```bash
python -m src.evaluation.eval_sweep_siglip_reg --help
python -m src.evaluation.score_generated_vs_original --help
python -m src.evaluation.score_generated_vs_original_bg --help
```

## Reproducibility Instructions

```bash
bash scripts/run_analysis.sh
bash scripts/generate_figures.sh
```

## Dataset Restriction Statement

No private/restricted dataset or image artifact is redistributed in this release.
See DATASET_NOTICE.md for details.

## Current Release Status

Implemented:
- evaluation
- LPIPS
- analysis
- figure generation

Planned Future Release:
- inference code
- training code
- additional documentation

Any missing component may be listed as: "To be released in a future update."
