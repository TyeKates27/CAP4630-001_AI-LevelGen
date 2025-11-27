# AI for Generating Game Levels — Starter Kit

This repo is a scaffold for your CAP 4630 project on Procedural Content Generation (PCG) via AI.

## Project goals
- Train an **n‑gram/Markov** baseline and an **LSTM** model to generate 2D tile‑based levels (e.g., Super Mario‑like).
- Evaluate **playability (rule checks)**, **style similarity (n‑gram KL)**, and **diversity/novelty**.
- Deliver a 6–15 slide deck + short demo video per course guidelines.

## Structure
```
ai-levelgen-starter/
  data/
    raw/         # put training .txt tile maps here (one level per file, same width per line)
    processed/   # tokenized/sequenced data written here
  notebooks/
    EDA.ipynb    # optional: quick data peek
  reports/
    figures/     # charts and visualizations
  models/        # trained models (lstm.pt, ngram.pkl)
  src/
    prepare_data.py
    ngram.py
    lstm_train.py
    generate.py
    evaluate.py
  README.md
```

## Quickstart
1) Place plain‑text levels into `data/raw/` (each line = row of tiles; use characters like `X` for solid, `-` for empty, `E` for enemy, `?` for block, etc.).  
2) Run preprocessing:
```bash
python src/prepare_data.py --input_dir data/raw --out data/processed/dataset.txt
```
3) Train baseline n‑gram:
```bash
python src/ngram.py --data data/processed/dataset.txt --n 3 --out models/ngram.pkl
```
4) Train LSTM:
```bash
python src/lstm_train.py --data data/processed/dataset.txt --epochs 3 --out models/lstm.pt
```
5) Generate samples:
```bash
python src/generate.py --model lstm --ckpt models/lstm.pt --length 2000 --out reports/figures/sample_lstm.txt
python src/generate.py --model ngram --ckpt models/ngram.pkl --length 2000 --out reports/figures/sample_ngram.txt
```
6) Evaluate:
```bash
python src/evaluate.py --ref data/processed/dataset.txt --gen reports/figures/sample_lstm.txt --tile_set "X-?E" --width 136
```

## Notes
- This is a **minimal** reference implementation intended for learning. You can extend with CNN/Transformer, grammar constraints, or Wave Function Collapse for structure.
- Keep your deck aligned with the **CAP 4630 presentation guidelines**.
