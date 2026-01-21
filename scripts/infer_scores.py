import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from model.autoencoder import AutoencoderRASL
from scipy.ndimage import gaussian_filter1d

def score(feat_path, model_path, out_path="outputs/scores.npy", smooth_sigma=2.0):
    print(f"🔍 Scoring features using Autoencoder model...")

    if not os.path.exists(feat_path):
        print(f"❌ Feature file not found: {feat_path}")
        return
    if not os.path.exists(model_path):
        print(f"❌ Model checkpoint not found: {model_path}")
        return

    # Load feature data
    data = np.load(feat_path)
    if data.ndim != 2:
        print("⚠️ Invalid feature format. Expected 2D feature array.")
        return

    tensor_data = torch.tensor(data, dtype=torch.float32)

    # Initialize model
    model = AutoencoderRASL(input_dim=data.shape[1])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        recon = model(tensor_data)
        mse_errors = ((tensor_data - recon) ** 2).mean(dim=1).numpy()

    # Smooth errors with Gaussian filter
    smooth_errors = gaussian_filter1d(mse_errors, sigma=smooth_sigma)

    # Save results
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, smooth_errors)
    print(f"✅ Highlight scores saved → {out_path}")
    print(f"📊 Mean Error: {smooth_errors.mean():.4f} | Std: {smooth_errors.std():.4f} | Max: {smooth_errors.max():.4f}")

if __name__ == "__main__":
    score(
        feat_path="features/myvideo.npy", 
        model_path="model/ae.pth", 
        out_path="outputs/scores.npy", 
        smooth_sigma=2.0  # ⬅️ Tweak for smoother or sharper detection
    )
