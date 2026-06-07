# Setup

## Option 1: pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Option 2: conda

```bash
conda env create -f environment.yml
conda activate robust-fashionability-enhancement
```

## Verify

```bash
bash scripts/run_evaluation.sh
bash scripts/run_analysis.sh
bash scripts/generate_figures.sh
```
