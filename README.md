# RASL: Slow/Fast Highlight Maker

Auto-detect and stylize highlights by training a small PyTorch autoencoder to reconstruct per-segment visual features. Segments the model reconstructs poorly (high reconstruction error) are treated as highlights and played in slow-motion; the rest are hyperlapsed.

## How it works
- Extract frames that show motion or scene change, encode them with a pretrained Swin Transformer into feature vectors ([scripts/extract_features.py](scripts/extract_features.py)).
- Train a residual MLP autoencoder on those features ([scripts/train_autoencoder.py](scripts/train_autoencoder.py)).
- Score each segment by its reconstruction MSE; higher error = more novel/interesting ([scripts/infer_scores.py](scripts/infer_scores.py)).
- Render a slow/fast highlight reel, slowing top-scoring segments and skipping through the rest ([scripts/hyperslow_highlight.py](scripts/hyperslow_highlight.py)).

## Quickstart
1) Install deps (Python 3.9+):
   ```bash
   pip install torch torchvision timm opencv-python pillow tqdm scipy numpy
   ```
2) Put a source video at `videos/myvideo.mp4`.
3) Extract features:
   ```bash
   python scripts/extract_features.py
   ```
4) Train the autoencoder (adjust epochs as needed):
   ```bash
   python scripts/train_autoencoder.py -- if you prefer CLI, edit the params inside the script
   ```
5) Score segments:
   ```bash
   python scripts/infer_scores.py
   ```
6) Make the highlight reel:
   ```bash
   python scripts/hyperslow_highlight.py
   ```

Artifacts land in `features/`, `model/`, and `outputs/` by default. Tune thresholds in the scripts (scene/motion gates, smoothing sigma, top-percent, slow/fast factors) to match your footage.
