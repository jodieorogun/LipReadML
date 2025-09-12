# Lip Reading (GRID) — Student Project

End-to-end lip-reading pipeline: face/mouth ROI, HOG/3D-CNN features, and classifier.

## Quickstart
python -m venv .venv
# activate venv (Windows: .venv\Scripts\activate; macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python main.py  # or your entry script

## Data
Uses the GRID corpus (not included). Place videos under data/… as described in scripts/README_data.md.

## Results (current)
Top-1 accuracy:  **X%** on **N** words (baseline).
Notes: small training split; noisy mouth crops; room to improve.

## Roadmap
- Better ROI (stabilized mouth crop)
- Data balance/augmentation
- Try 3D-CNN with optical flow
- Tune classifier & eval (top-k, per-class)
