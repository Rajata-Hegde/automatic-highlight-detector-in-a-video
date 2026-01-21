import os
import cv2
import timm
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# ✅ Load pretrained Swin Transformer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "swin_base_patch4_window7_224"
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
model.eval().to(device)

# 🧠 Swin-compatible image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def is_scene_change(prev_frame, curr_frame, threshold=0.6):
    """Detect if the current frame is significantly different from the previous one."""
    prev_hist = cv2.calcHist([prev_frame], [0], None, [256], [0, 256])
    curr_hist = cv2.calcHist([curr_frame], [0], None, [256], [0, 256])
    similarity = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
    return similarity < threshold  # Less similarity = scene change

def is_motion_detected(prev_gray, curr_gray, motion_threshold=1.5):
    """Detect motion based on optical flow magnitude."""
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(magnitude) > motion_threshold

def extract_features(video_path, output_path, scene_threshold=0.6, motion_threshold=1.5):
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"🎞 Total frames: {total_frames} | FPS: {fps:.2f}")

    features = []
    extracted = 0
    failed = 0

    # Setup first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("❌ Failed to read first frame.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_scene = prev_gray.copy()

    pbar = tqdm(total=total_frames, desc="🔍 Extracting features")

    with torch.inference_mode():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion = is_motion_detected(prev_gray, curr_gray, motion_threshold)
            scene_changed = is_scene_change(prev_scene, curr_gray, scene_threshold)

            if motion or scene_changed:
                try:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    inp = transform(img).unsqueeze(0).to(device)
                    feat = model(inp).squeeze().cpu().numpy().astype(np.float32)
                    features.append(feat)
                    extracted += 1
                except Exception:
                    failed += 1

                prev_scene = curr_gray  # Update scene reference

            prev_gray = curr_gray
            pbar.update(1)

    cap.release()
    pbar.close()

    if not features:
        print("⚠️ No features extracted.")
        return

    np.save(output_path, np.stack(features))
    print(f"✅ Saved {extracted} features → {output_path}")
    if failed:
        print(f"⚠️ Skipped {failed} frames due to extraction errors.")

if __name__ == "__main__":
    extract_features(
        video_path="videos/myvideo.mp4",
        output_path="features/myvideo.npy",
        scene_threshold=0.6,         # 🔧 adjust for more/less scene change
        motion_threshold=1.5         # 🔧 adjust for slow/fast motion videos
    )
