# Lip Reading (GRID)

End-to-end baseline: face/mouth ROI (OpenCV + dlib) → HOG / 3D-CNN features → classifier.

## Project Layout
LipReadML/
├── main.py
├── .gitignore
├── requirements.txt
├── data/                # local only (ignored) e.g. landmarks.dat
└── GRID/                # local only (ignored: s1/s2/s3 with videos/ and align/)

## Quickstart (macOS, Homebrew Python on Apple Silicon)
# 1) Create & activate a virtual environment
/opt/homebrew/bin/python3 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run
python main.py



## Data (not included in repo)
- GRID dataset in `GRID/` (with `s1/s2/s3/videos` and `align`) — **not committed**.
- dlib landmarks model in `data/landmarks.dat` (or `shape_predictor_68_face_landmarks.dat`) — **not committed**.

## Results (baseline)
- Notes: small training split; ROI jitter; class imbalance.

## Roadmap
- Stabilize mouth ROI
- Balance/augment data
- Try 3D-CNN (optionally with optical flow)
- Tune classifier & eval (top-k, per-class)

