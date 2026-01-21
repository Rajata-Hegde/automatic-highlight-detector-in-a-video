#!/usr/bin/env python3
"""
hybrid_optflow_slomo_hyperlapse.py

Lightweight hybrid pipeline:
 - Reads scores.npy (segment-level importance)
 - Stabilizes each segment
 - For high-importance segments: motion-guided optical-flow interpolation to produce clean slow-mo
 - For low-importance segments: motion-adaptive hyperlapse (skip more in static areas)
 - Output: MP4 (no audio)
"""

import os
import numpy as np
import cv2
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

# ------------------------
# CONFIG (tune if needed)
# ------------------------
VIDEO_PATH = "videos/myvideo.mp4"
SCORES_PATH = "outputs/scores.npy"
OUT_PATH = "outputs/hybrid_smart_edit.mp4"

HIGHLIGHT_RATIO = 0.25        # top fraction of segments forced-as-highlights
SEGMENT_MERGE_GAP = 1         # merge small gaps between highlight segments

SLOW_MAX_FACTOR = 4           # integer maximum slow factor (>=1). 4 => insert 3 mids between pairs
HYPER_MIN_SKIP = 1            # min skip (no skip)
HYPER_MAX_SKIP = 6            # max skip (fast)
SLOW_RAMP_SECONDS = 0.6       # seconds around climax to center slow-mo window

STABILIZE_SMOOTH_SIGMA = 3    # smoothing for stabilization trajectory
FLOW_METHOD = "farneback"     # 'farneback' (default) or 'dis' (if cv2 contrib available)

MOTION_NORMALIZE = 12.0       # value to normalize motion into [0,1] (tune per dataset)
MOTION_PIXEL_THRESH = 0.8     # pixel-level mag threshold for "moving" mask in interpolation

# ------------------------
# Utilities
# ------------------------
def load_scores(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    s = np.load(path)
    return s.squeeze() if s.ndim > 1 else s

def scores_to_intervals(scores, ratio_keep=HIGHLIGHT_RATIO, merge_gap=SEGMENT_MERGE_GAP):
    n = len(scores)
    topk = max(1, int(ratio_keep * n))
    top_idx = set(np.argsort(scores)[-topk:])
    sel = np.array([i in top_idx for i in range(n)], dtype=np.int32)
    intervals, start = [], None
    for i, v in enumerate(sel):
        if v and start is None:
            start = i
        if not v and start is not None:
            intervals.append((start, i-1)); start = None
    if start is not None:
        intervals.append((start, n-1))
    merged = []
    for s, e in intervals:
        if not merged:
            merged.append((s, e))
        else:
            ps, pe = merged[-1]
            if s - pe <= merge_gap:
                merged[-1] = (ps, e)
            else:
                merged.append((s, e))
    return merged

def stabilize_frames(frames, smoothing_sigma=STABILIZE_SMOOTH_SIGMA):
    """Simple affine stabilization (small buffer)."""
    if len(frames) < 2:
        return frames
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    transforms = []
    for i in range(1, len(frames)):
        cur_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=8)
        if pts is None:
            M = np.eye(2,3, dtype=np.float32)
        else:
            pts2, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts, None)
            if pts2 is None:
                M = np.eye(2,3, dtype=np.float32)
            else:
                idx = st.flatten()==1
                pts1 = pts[idx].reshape(-1,2)
                pts2 = pts2[idx].reshape(-1,2)
                if len(pts1) < 6:
                    M = np.eye(2,3, dtype=np.float32)
                else:
                    M, _ = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                    if M is None:
                        M = np.eye(2,3, dtype=np.float32)
        transforms.append(M)
        prev_gray = cur_gray
    # accumulate transforms to get trajectory
    acc = np.eye(3)
    traj = []
    for M in transforms:
        M3 = np.vstack([M, [0,0,1]])
        acc = acc @ M3
        traj.append(acc.copy())
    if not traj:
        return frames
    flat = np.stack([t.ravel() for t in traj], axis=0)
    smooth = gaussian_filter1d(flat, sigma=smoothing_sigma, axis=0)
    smooth = smooth.reshape(traj.shape)
    stabilized = [frames[0]]
    for i in range(len(smooth)):
        corr = np.linalg.inv(smooth[i]) @ traj[i]
        Mcorr = corr[:2,:].astype(np.float32)
        h, w = frames[0].shape[:2]
        warped = cv2.warpAffine(frames[i+1], Mcorr, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        stabilized.append(warped)
    return stabilized

def compute_frame_flow(g1, g2, method=FLOW_METHOD):
    if method == 'dis':
        try:
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
            return dis.calc(g1, g2, None)
        except Exception:
            pass
    return cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def interpolate_midpoint_motionguided(a, b, method=FLOW_METHOD, pixel_thresh=MOTION_PIXEL_THRESH):
    """Create a clean midpoint between a and b:
       - warp where pixels move (flow mag > thresh)
       - average where static
       - apply light sharpening to reduce softness
    """
    g1 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    flow12 = compute_frame_flow(g1, g2, method=method)
    flow21 = compute_frame_flow(g2, g1, method=method)

    h, w = g1.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')

    map_x1 = (grid_x + 0.5 * flow12[...,0]).astype(np.float32)
    map_y1 = (grid_y + 0.5 * flow12[...,1]).astype(np.float32)
    warp1 = cv2.remap(a, map_x1, map_y1, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    map_x2 = (grid_x + 0.5 * flow21[...,0]).astype(np.float32)
    map_y2 = (grid_y + 0.5 * flow21[...,1]).astype(np.float32)
    warp2 = cv2.remap(b, map_x2, map_y2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    mid_warp = ((warp1.astype(np.float32) + warp2.astype(np.float32)) * 0.5).astype(np.uint8)
    avg = ((a.astype(np.float32) + b.astype(np.float32)) * 0.5).astype(np.uint8)

    mag, _ = cv2.cartToPolar(flow12[...,0], flow12[...,1])
    mag_blur = cv2.GaussianBlur(mag, (9,9), 0)
    move_mask = (mag_blur > pixel_thresh).astype(np.uint8)
    move_mask = cv2.morphologyEx(move_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    out = np.where(move_mask[...,None]==1, mid_warp, avg)

    # unsharp / light sharpen
    blurred = cv2.GaussianBlur(out, (0,0), 1.0)
    sharp = cv2.addWeighted(out, 1.4, blurred, -0.4, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def segment_motion_mean(frames):
    if len(frames) < 2:
        return 0.0
    mags = []
    for i in range(len(frames)-1):
        g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
        flow = compute_frame_flow(g1, g2)
        mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
        mags.append(np.mean(mag))
    return float(np.mean(mags)) if mags else 0.0

# ------------------------
# Main hybrid processing
# ------------------------
def process_hybrid(video_path=VIDEO_PATH, scores_path=SCORES_PATH, out_path=OUT_PATH):
    assert os.path.exists(video_path), "video missing"
    assert os.path.exists(scores_path), "scores missing"

    scores = load_scores(scores_path)
    if scores.ndim > 1:
        scores = scores.squeeze()
    scores = gaussian_filter1d(scores, sigma=1.0)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    seg_count = len(scores)
    seg_frames = max(1, total_frames // seg_count)

    intervals = scores_to_intervals(scores)
    highlight_set = set()
    for s,e in intervals:
        for seg in range(s, e+1):
            highlight_set.add(seg)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w,h))

    print(f"Frames={total_frames} | Segments={seg_count} | Frames/segment~{seg_frames}")
    print("Highlight intervals (segments):", intervals)

    for seg_idx in tqdm(range(seg_count), desc="segments"):
        start_frame = seg_idx * seg_frames
        end_frame = min(total_frames, (seg_idx + 1) * seg_frames)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for f in range(start_frame, end_frame):
            ret, frm = cap.read()
            if not ret:
                break
            frames.append(frm)
        if not frames:
            continue

        # stabilize segment to remove camera jitter
        try:
            frames_stab = stabilize_frames(frames)
        except Exception:
            frames_stab = frames

        # compute normalized motion and combine with score
        motion_val = segment_motion_mean(frames_stab)
        motion_norm = np.clip(motion_val / MOTION_NORMALIZE, 0.0, 1.0)
        min_s, max_s = float(np.min(scores)), float(np.max(scores))
        seg_norm = 0.0 if (max_s - min_s) < 1e-6 else (float(scores[seg_idx]) - min_s) / (max_s - min_s)
        combined = float(np.clip(0.7 * seg_norm + 0.3 * motion_norm, 0.0, 1.0))

        # map to slow & skip factors
        slow_factor = max(1, int(round(1 + (SLOW_MAX_FACTOR - 1) * combined)))
        skip_factor = max(1, int(round(HYPER_MAX_SKIP - (HYPER_MAX_SKIP - HYPER_MIN_SKIP) * combined)))

        # bias highlighted segments strongly toward slow-mo
        if seg_idx in highlight_set:
            slow_factor = max(slow_factor, SLOW_MAX_FACTOR)
            skip_factor = min(skip_factor, HYPER_MIN_SKIP)

        # produce output for this segment
        if slow_factor > 1:
            # find a climax (frame index with highest local motion)
            mags = []
            for i in range(len(frames_stab)-1):
                g1 = cv2.cvtColor(frames_stab[i], cv2.COLOR_BGR2GRAY)
                g2 = cv2.cvtColor(frames_stab[i+1], cv2.COLOR_BGR2GRAY)
                flow = compute_frame_flow(g1, g2)
                mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
                mags.append(np.mean(mag))
            climax = int(np.argmax(mags)) if mags else len(frames_stab)//2
            ramp = int(SLOW_RAMP_SECONDS * fps)
            slow_s = max(0, climax - ramp)
            slow_e = min(len(frames_stab), climax + ramp + 1)

            pre = frames_stab[:slow_s]
            mid = frames_stab[slow_s:slow_e]
            post = frames_stab[slow_e:]

            # adaptive hyperlapse for pre/post
            def adaptive_hyper(chunk):
                if not chunk: return []
                out = []
                # compute local motion mags for chunk
                cmags = []
                for i in range(len(chunk)-1):
                    g1 = cv2.cvtColor(chunk[i], cv2.COLOR_BGR2GRAY)
                    g2 = cv2.cvtColor(chunk[i+1], cv2.COLOR_BGR2GRAY)
                    flow = compute_frame_flow(g1, g2)
                    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
                    cmags.append(np.mean(mag))
                cmags = np.array([0.0] + cmags)
                if cmags.max() - cmags.min() > 1e-6:
                    norm = (cmags - cmags.min()) / (cmags.max() - cmags.min())
                else:
                    norm = np.zeros_like(cmags)
                i = 0
                while i < len(chunk):
                    m = float(norm[min(i, len(norm)-1)])
                    local_skip = max(1, int(round(HYPER_MAX_SKIP - (HYPER_MAX_SKIP - HYPER_MIN_SKIP) * m)))
                    out.append(chunk[i])
                    i += local_skip
                return out

            out_frames = []
            out_frames += adaptive_hyper(pre)

            # slow mid: insert (slow_factor -1) midpoints between each pair using motion-guided interpolation
            if len(mid) <= 1:
                out_frames += mid
            else:
                for i in range(len(mid)-1):
                    out_frames.append(mid[i])
                    a = mid[i]; b = mid[i+1]
                    for rep in range(slow_factor - 1):
                        midf = interpolate_midpoint_motionguided(a, b, method=FLOW_METHOD, pixel_thresh=MOTION_PIXEL_THRESH)
                        out_frames.append(midf)
                out_frames.append(mid[-1])

            out_frames += adaptive_hyper(post)

            for fr in out_frames:
                writer.write(fr)

        else:
            # adaptive hyperlapse across entire stabilized chunk
            cmags = []
            for i in range(len(frames_stab)-1):
                g1 = cv2.cvtColor(frames_stab[i], cv2.COLOR_BGR2GRAY)
                g2 = cv2.cvtColor(frames_stab[i+1], cv2.COLOR_BGR2GRAY)
                flow = compute_frame_flow(g1, g2)
                mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
                cmags.append(np.mean(mag))
            cmags = np.array([0.0] + cmags)
            if cmags.max() - cmags.min() > 1e-6:
                norm = (cmags - cmags.min()) / (cmags.max() - cmags.min())
            else:
                norm = np.zeros_like(cmags)
            i = 0
            while i < len(frames_stab):
                m = float(norm[min(i, len(norm)-1)])
                local_skip = max(1, int(round(HYPER_MAX_SKIP - (HYPER_MAX_SKIP - HYPER_MIN_SKIP) * m)))
                writer.write(frames_stab[i])
                i += local_skip

    cap.release()
    writer.release()
    print("Done ->", out_path)

if __name__ == "__main__":
    process_hybrid(VIDEO_PATH, SCORES_PATH, OUT_PATH)
