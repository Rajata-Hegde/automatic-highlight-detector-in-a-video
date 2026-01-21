import numpy as np
import cv2
import os

def make_slowfast_highlight_opencv(original, scores_path, out_video="outputs/styled_highlight.mp4",
                                    top_percent=0.3, slow_factor=2, fast_skip=3):
    if not os.path.exists(original):
        print(f"❌ Video file not found: {original}")
        return

    scores = np.load(scores_path)
    total_segments = len(scores)

    cap = cv2.VideoCapture(original)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {original}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seg_frames = total_frames // total_segments
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 🧵 Video writer
    out = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # 🎯 Threshold for highlight (top X%)
    threshold = np.percentile(scores, 100 - top_percent * 100)
    highlights = [i for i, s in enumerate(scores) if s >= threshold]

    print(f"🚀 Segments: {total_segments} | Highlights: {len(highlights)}")
    
    for seg_idx in range(total_segments):
        is_highlight = seg_idx in highlights
        start = seg_idx * seg_frames
        end = min((seg_idx + 1) * seg_frames, total_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        frame_idx = start
        while frame_idx < end:
            ret, frame = cap.read()
            if not ret:
                break

            if is_highlight:
                # 🐢 Slow-motion: write same frame multiple times
                for _ in range(slow_factor):
                    out.write(frame)
                frame_idx += 1
            else:
                # ⚡ Hyperlapse: skip frames
                out.write(frame)
                frame_idx += fast_skip
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        print(f"{'🟢' if is_highlight else '⚪'} Segment {seg_idx+1}/{total_segments} processed")

    cap.release()
    out.release()
    print(f"✅ Final styled highlight video → {out_video}")

if __name__ == "__main__":
    make_slowfast_highlight_opencv(
        original="videos/myvideo.mp4",
        scores_path="outputs/scores.npy",
        out_video="outputs/styled_highlight.mp4",
        top_percent=0.3,     # Top 30% considered highlight
        slow_factor=2,       # Write frame twice (0.5x effect)
        fast_skip=3          # Skip 2 out of 3 frames (3x effect)
    )
