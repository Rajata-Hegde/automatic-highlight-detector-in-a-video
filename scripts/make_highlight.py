import os
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d

# ----------------------------
# CONFIG (simple)
# ----------------------------
VIDEO_PATH   = "videos/myvideo.mp4"
SCORES_PATH  = "outputs/scores.npy"   # shape: [num_segments]
OUT_PATH     = "outputs/smart_edit.mp4"

HIGHLIGHT_RATIO = 0.25   # top 25% segments are highlights
MERGE_GAP       = 1      # merge small gaps between highlight segments (in segments)

SLOW_FACTOR  = 4         # 4x slower (insert 3 mid/dup frames between originals)
HYPER_SKIP   = 6         # write every 6th frame in non-highlight (≈6x faster)

USE_BLEND_INTERP = True  # if True: simple midpoint blend; if False: duplicate frames

# ----------------------------
# utils
# ----------------------------
def scores_to_intervals(scores, ratio_keep=HIGHLIGHT_RATIO, merge_gap=MERGE_GAP):
    """Pick top-k segments and merge tiny gaps."""
    n = len(scores)
    topk = max(1, int(ratio_keep * n))
    top_idx = set(np.argsort(scores)[-topk:])
    sel = np.array([i in top_idx for i in range(n)], dtype=np.int32)

    intervals = []
    start = None
    for i, v in enumerate(sel):
        if v and start is None:
            start = i
        if not v and start is not None:
            intervals.append((start, i - 1))
            start = None
    if start is not None:
        intervals.append((start, n - 1))

    # merge small gaps
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

def midpoint_blend(a, b):
    """Cheap midpoint ‘interpolation’ by averaging two frames (BGR uint8)."""
    return cv2.addWeighted(a, 0.5, b, 0.5, 0)

# ----------------------------
# main
# ----------------------------
def process_simple(video_path, scores_path, out_path):
    assert os.path.exists(video_path), "video missing"
    assert os.path.exists(scores_path), "scores missing"

    scores = np.load(scores_path).squeeze()
    # optional mild smoothing to reduce noisy flips
    scores = gaussian_filter1d(scores, sigma=1.0)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    seg_count = len(scores)
    seg_frames = max(1, total_frames // seg_count)

    intervals = scores_to_intervals(scores, HIGHLIGHT_RATIO, MERGE_GAP)
    highlight_set = set()
    for s, e in intervals:
        for seg in range(s, e + 1):
            highlight_set.add(seg)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    print(f"frames={total_frames}, segments={seg_count}, frames/segment={seg_frames}")
    print("highlight intervals:", intervals)

    for seg_idx in range(seg_count):
        sframe = seg_idx * seg_frames
        eframe = min(total_frames, (seg_idx + 1) * seg_frames)

        cap.set(cv2.CAP_PROP_POS_FRAMES, sframe)
        frames = []
        for f in range(sframe, eframe):
            ret, fr = cap.read()
            if not ret: break
            frames.append(fr)

        if not frames:
            continue

        if seg_idx in highlight_set:
            # SLOW MOTION: write each frame, and insert (SLOW_FACTOR-1) mids/dups
            for i in range(len(frames) - 1):
                a, b = frames[i], frames[i + 1]
                writer.write(a)
                # insert extra frames
                for _ in range(SLOW_FACTOR - 1):
                    if USE_BLEND_INTERP:
                        mid = midpoint_blend(a, b)
                        writer.write(mid)
                    else:
                        writer.write(a)  # simple duplication
            writer.write(frames[-1])
        else:
            # HYPERLAPSE: write every HYPER_SKIP-th frame
            for i in range(0, len(frames), max(1, HYPER_SKIP)):
                writer.write(frames[i])

    cap.release()
    writer.release()
    print("Done:", out_path)

if __name__ == "__main__":
    process_simple(VIDEO_PATH, SCORES_PATH, OUT_PATH)
