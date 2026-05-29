#!/usr/bin/env python3
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

VIDEOS = [f"videos/{i}.mp4" for i in range(1, 17)]

THRESHOLD_CSV = "jolly_good_data.csv"
ROI_FILE = "rois.json"
OUT_DIR = "saved_threshold_plots_separate"

PEAK_MERGE_FRAMES = 5
NORMALIZE_BY_MAX = True
TARGET_THRESHOLD = 0.4

CROP_PLOT_FRAMES = {
    "videos/10.mp4": 1090,
}

# Angle order:
# a,b ; a,b' ; a,b_p ; a,b'_p ;
# a',b ; a',b' ; ...
ANGLE_VALUES = {
    1:  (-45, -22.5),
    2:  (-45,  22.5),
    3:  (-45,  67.5),
    4:  (-45, 112.5),
    5:  (0,   -22.5),
    6:  (0,    22.5),
    7:  (0,    67.5),
    8:  (0,   112.5),
    9:  (45,  -22.5),
    10: (45,   22.5),
    11: (45,   67.5),
    12: (45,  112.5),
    13: (90,  -22.5),
    14: (90,   22.5),
    15: (90,   67.5),
    16: (90,  112.5),
}

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 1.2,
    "lines.linewidth": 1.5,
    "savefig.dpi": 300,
})


def get_frame_intensity(video_path, roi):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    x, y, w, h = map(int, roi)
    vals = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crop = gray[y:y+h, x:x+w]
        vals.append(np.sum(crop))

    cap.release()

    vals = np.array(vals, dtype=float)

    if len(vals) == 0:
        raise RuntimeError(f"No frames read from video: {video_path}")

    if NORMALIZE_BY_MAX and vals.max() > 0:
        vals = vals / vals.max()

    return vals


def select_rois(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Could not read first frame from {video_path}")

    print(f"\nSelecting ROIs using {video_path}")

    print("Select ALICE ROI, then press ENTER/SPACE.")
    alice_roi = cv2.selectROI("Select ALICE ROI", frame, showCrosshair=True)
    cv2.destroyWindow("Select ALICE ROI")

    print("Select BOB ROI, then press ENTER/SPACE.")
    bob_roi = cv2.selectROI("Select BOB ROI", frame, showCrosshair=True)
    cv2.destroyWindow("Select BOB ROI")

    return {
        "alice": tuple(map(int, alice_roi)),
        "bob": tuple(map(int, bob_roi)),
    }


def load_or_create_rois():
    roi_path = Path(ROI_FILE)

    if roi_path.exists():
        with open(roi_path, "r") as f:
            return json.load(f)

    shared_rois = select_rois(VIDEOS[0])

    rois = {}
    for video_path in VIDEOS:
        rois[video_path] = shared_rois

    with open(roi_path, "w") as f:
        json.dump(rois, f, indent=2)

    return rois


def find_peaks_grouped(y, threshold, merge_frames=5):
    above = np.where(y >= threshold)[0]

    if len(above) == 0:
        return np.array([], dtype=int)

    groups = [[above[0]]]

    for idx in above[1:]:
        if idx - groups[-1][-1] < merge_frames:
            groups[-1].append(idx)
        else:
            groups.append([idx])

    peaks = []
    for g in groups:
        g = np.array(g)
        peak_frame = g[np.argmax(y[g])]
        peaks.append(peak_frame)

    return np.array(peaks, dtype=int)


def load_thresholds():
    df = pd.read_csv(THRESHOLD_CSV)

    required = ["video", "alice_threshold", "bob_threshold"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column in {THRESHOLD_CSV}: {col}")

    thresholds = {}

    for _, row in df.iterrows():
        video = str(row["video"])

        thresholds[video] = {
            "alice": float(row["alice_threshold"]),
            "bob": float(row["bob_threshold"]),
        }

    return thresholds


def normalize_by_threshold(y, threshold):
    if threshold <= 0:
        raise ValueError("Threshold must be positive.")

    return y * (TARGET_THRESHOLD / threshold)


def angle_title(side_name, video_number):
    alpha, beta = ANGLE_VALUES[video_number]
    return rf"{side_name}: $\alpha={alpha}^\circ,\ \beta={beta}^\circ$"


def main():
    out_dir = Path(OUT_DIR)
    alice_dir = out_dir / "alice"
    bob_dir = out_dir / "bob"

    alice_dir.mkdir(parents=True, exist_ok=True)
    bob_dir.mkdir(parents=True, exist_ok=True)

    rois = load_or_create_rois()
    thresholds = load_thresholds()

    global_min = np.inf
    global_max = -np.inf

    cached_data = {}

    for video_path in VIDEOS:
        alice_y = get_frame_intensity(video_path, rois[video_path]["alice"])
        bob_y = get_frame_intensity(video_path, rois[video_path]["bob"])

        alice_thr = thresholds[video_path]["alice"]
        bob_thr = thresholds[video_path]["bob"]

        alice_norm = normalize_by_threshold(alice_y, alice_thr)
        bob_norm = normalize_by_threshold(bob_y, bob_thr)

        plot_until_alice = CROP_PLOT_FRAMES.get(video_path, len(alice_norm))
        plot_until_bob = CROP_PLOT_FRAMES.get(video_path, len(bob_norm))

        alice_for_limits = alice_norm[:plot_until_alice]
        bob_for_limits = bob_norm[:plot_until_bob]

        global_min = min(global_min, alice_for_limits.min(), bob_for_limits.min())
        global_max = max(global_max, alice_for_limits.max(), bob_for_limits.max())

        cached_data[video_path] = {
            "alice_y": alice_y,
            "bob_y": bob_y,
            "alice_thr": alice_thr,
            "bob_thr": bob_thr,
        }

    y_margin = 0.05 * (global_max - global_min)
    global_ymin = global_min - y_margin
    global_ymax = global_max + y_margin

    for video_path in VIDEOS:
        data = cached_data[video_path]

        alice_y = data["alice_y"]
        bob_y = data["bob_y"]

        alice_thr = data["alice_thr"]
        bob_thr = data["bob_thr"]

        video_number = int(Path(video_path).stem)

        alice_out = alice_dir / f"video_{video_number:02d}_alice.png"
        bob_out = bob_dir / f"video_{video_number:02d}_bob.png"

        # ---------------- ALICE ----------------

        y = normalize_by_threshold(alice_y, alice_thr)

        peaks = find_peaks_grouped(
            y,
            TARGET_THRESHOLD,
            PEAK_MERGE_FRAMES
        )

        fig, ax = plt.subplots(figsize=(11, 5))

        plot_until = CROP_PLOT_FRAMES.get(video_path, len(y))
        y_plot = y[:plot_until]

        ax.plot(y_plot, label="intensity")

        ax.axhline(
            TARGET_THRESHOLD,
            linestyle="--",
            label="threshold = 0.4"
        )

        peaks_plot = peaks[peaks < plot_until]

        ax.scatter(
            peaks_plot,
            y[peaks_plot],
            marker="o"
        )

        ax.set_ylim(global_ymin, global_ymax)

        ax.set_title(angle_title("Alice", video_number))
        ax.set_xlabel("Frame")
        ax.set_ylabel("Normalized intensity")

        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        ax.tick_params(direction="in", top=True, right=True)

        fig.tight_layout()
        fig.savefig(alice_out, dpi=300)
        plt.close(fig)

        # ---------------- BOB ----------------

        y = normalize_by_threshold(bob_y, bob_thr)

        peaks = find_peaks_grouped(
            y,
            TARGET_THRESHOLD,
            PEAK_MERGE_FRAMES
        )

        fig, ax = plt.subplots(figsize=(11, 5))

        plot_until = CROP_PLOT_FRAMES.get(video_path, len(y))
        y_plot = y[:plot_until]

        ax.plot(y_plot, label="intensity")

        ax.axhline(
            TARGET_THRESHOLD,
            linestyle="--",
            label="threshold = 0.4"
        )

        peaks_plot = peaks[peaks < plot_until]

        ax.scatter(
            peaks_plot,
            y[peaks_plot],
            marker="o",
        )

        ax.set_ylim(global_ymin, global_ymax)

        ax.set_title(angle_title("Bob", video_number))
        ax.set_xlabel("Frame")
        ax.set_ylabel("Normalized intensity")

        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        ax.tick_params(direction="in", top=True, right=True)

        fig.tight_layout()
        fig.savefig(bob_out, dpi=300)
        plt.close(fig)

        print(f"Saved {alice_out}")
        print(f"Saved {bob_out}")

    print(f"\nDone. Saved 32 plots in: {out_dir}")


if __name__ == "__main__":
    main()