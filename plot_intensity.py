#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ================= USER PARAMETERS =================

VIDEOS = [
    "videos/"+str(i)+".mp4" for i in range(1,17)
]

# video1 = cv2.VideoCapture(VIDEOS[0])
# ret, frame = video1.read()

# # ROI format: (x, y, width, height)
# # You need to tune these numbers for your screen recording.
# ALICE_ROI = cv2.selectROI("SELECT ROI 1", frame, fromCenter=False)
# BOB_ROI   = cv2.selectROI("SELECT ROI 2", frame, fromCenter=False)

# print("SARA - ", ALICE_ROI)
# print("GARY - ", BOB_ROI)

# cv2.destroyAllWindows()

# ROI format: (x, y, width, height)
# You need to tune these numbers for your screen recording.
ALICE_ROI = (558, 75, 265, 192)
BOB_ROI   = (559, 376, 280, 187)


THRESHOLD_ALICE = 2.9e4
THRESHOLD_BOB   = 2.2e4

BACKGROUND_SUBTRACT = True
BACKGROUND_FRAMES = 5

SAVE_PLOTS = True
OUTPUT_DIR = "analysis_output"

# ===================================================


def crop_roi(gray, roi):
    x, y, w, h = roi
    return gray[y:y+h, x:x+w]


def read_roi_intensity(video_path, roi):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
        roi_frame = crop_roi(gray, roi)
        frames.append(roi_frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames found in {video_path}")

    if BACKGROUND_SUBTRACT:
        bg = np.mean(frames[:BACKGROUND_FRAMES], axis=0)
    else:
        bg = 0.0

    intensities = []

    for frame in frames:
        img = frame - bg
        img[img < 0] = 0
        intensities.append(np.sum(img))

    return np.array(intensities)


def analyze_roi(video_path, label, roi, threshold):
    intensities = read_roi_intensity(video_path, roi)
    peak_frames = np.where(intensities > threshold)[0]

    print(f"\n{label} in {video_path}")
    print(f"Number of frames: {len(intensities)}")
    print(f"Threshold: {threshold}")
    print(f"Frames above threshold: {len(peak_frames)}")
    print(f"Peak frame indices: {peak_frames.tolist()}")

    plt.plot(intensities, marker="o", label=f"{label} intensity")
    plt.axhline(threshold, linestyle="--", label=f"{label} threshold")

    if len(peak_frames) > 0:
        plt.scatter(
            peak_frames,
            intensities[peak_frames],
            s=70,
            label=f"{label} peaks",
        )

    return intensities, peak_frames


def analyze_video(video_path):
    video_path = Path(video_path)

    plt.figure(figsize=(11, 5))

    alice_int, alice_peaks = analyze_roi(
        video_path,
        "Alice",
        ALICE_ROI,
        THRESHOLD_ALICE,
    )

    bob_int, bob_peaks = analyze_roi(
        video_path,
        "Bob",
        BOB_ROI,
        THRESHOLD_BOB,
    )

    plt.xlabel("Frame number")
    plt.ylabel("Total intensity in ROI")
    plt.title(f"Intensity per frame: {video_path.name}")
    plt.legend()
    plt.tight_layout()

    if SAVE_PLOTS:
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        outname = Path(OUTPUT_DIR) / f"{video_path.stem}_alice_bob_intensity.png"
        plt.savefig(outname, dpi=200)
        print(f"\nSaved plot to {outname}")

    plt.show()


def main():
    for video in VIDEOS:
        analyze_video(video)


if __name__ == "__main__":
    main()