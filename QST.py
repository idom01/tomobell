#!/usr/bin/env python3
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# USER PARAMETERS
# =========================

VIDEO_PATH = "QST_videos/partB50.mp4"

SELECT_ROIS = True

ROIS = {
    "Alice0": (100, 100, 200, 200),
    "Alice1": (350, 100, 200, 200),
    "Bob0":   (100, 400, 200, 200),
    "Bob1":   (350, 400, 200, 200),
}

ALPHA = 0.7

# If pulses are close, reduce this.
# If one physical pulse is detected multiple times, increase this.
MIN_PEAK_DISTANCE = 3

OUT_CSV = "qst_intensities.csv"
OUT_PULSE_CSV = "qst_pulses_only.csv"
OUT_PLOT = "qst_intensity_traces.png"


# =========================
# FUNCTIONS
# =========================

def choose_rois(frame):
    names = ["Alice0", "Alice1", "Bob0", "Bob1"]
    rois = {}

    for name in names:
        print(f"Select ROI for {name}, then press ENTER or SPACE.")
        r = cv2.selectROI(f"Select {name}", frame, showCrosshair=True)
        cv2.destroyWindow(f"Select {name}")
        rois[name] = tuple(map(int, r))

    return rois


def roi_intensity(frame, roi):
    x, y, w, h = roi
    crop = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(np.sum(gray))


def find_local_peaks(y, min_distance=3, min_height=1.0):
    """
    Find local maxima in a 1D signal.

    min_height is applied to y itself.
    """

    y = np.asarray(y)
    candidates = []

    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1] and y[i] >= min_height:
            candidates.append(i)

    if len(candidates) == 0:
        return np.array([], dtype=int)

    candidates = np.array(candidates)

    # Sort candidates by height, strongest first
    candidates = candidates[np.argsort(y[candidates])[::-1]]

    accepted = []

    for idx in candidates:
        if all(abs(idx - j) >= min_distance for j in accepted):
            accepted.append(idx)

    accepted = np.array(sorted(accepted), dtype=int)

    return accepted


# =========================
# READ VIDEO
# =========================

video_path = Path(VIDEO_PATH)
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Could not read first frame.")

if SELECT_ROIS:
    ROIS = choose_rois(first_frame)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

camera_names = ["Alice0", "Alice1", "Bob0", "Bob1"]

records = []
frame_idx = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    row = {"frame": frame_idx}

    for name in camera_names:
        row[name] = roi_intensity(frame, ROIS[name])

    records.append(row)
    frame_idx += 1

cap.release()

df = pd.DataFrame(records)


# =========================
# THRESHOLDS
# =========================

thresholds = {}
baselines = {}

print("\nAdaptive thresholds:")

for name in camera_names:
    I_min = df[name].min()
    I_max = df[name].max()
    I_th = ALPHA * I_max + (1 - ALPHA) * I_min

    baselines[name] = I_min
    thresholds[name] = I_th

    print(
        f"{name}: "
        f"I_min={I_min:.1f}, "
        f"I_max={I_max:.1f}, "
        f"I_th={I_th:.1f}"
    )


# =========================
# NORMALIZED EXCESS SIGNAL
# =========================
#
# For each camera:
#
#   excess = (I - threshold) / (I_max - threshold)
#
# clipped below zero.
#
# Then sum over cameras. A real pulse should make this signal peak.
#

df["pulse_score"] = 0.0

for name in camera_names:
    I = df[name].to_numpy(dtype=float)
    I_max = df[name].max()
    I_th = thresholds[name]

    denom = max(I_max - I_th, 1e-12)

    excess = (I - I_th) / denom
    excess = np.clip(excess, 0.0, None)

    df[f"{name}_excess"] = excess
    df["pulse_score"] += excess


# =========================
# PEAK DETECTION
# =========================

pulse_frames = find_local_peaks(
    df["pulse_score"].to_numpy(),
    min_distance=MIN_PEAK_DISTANCE,
    min_height=0.2,
)

df["is_pulse"] = False
df.loc[pulse_frames, "is_pulse"] = True

pulse_df = df.loc[pulse_frames].copy()

df["total"] = sum(df[name] for name in camera_names)
pulse_df["total"] = sum(pulse_df[name] for name in camera_names)

for name in camera_names:
    pulse_df[f"{name}_frac"] = pulse_df[name] / pulse_df["total"]


# =========================
# SAVE
# =========================

df.to_csv(OUT_CSV, index=False)
pulse_df.to_csv(OUT_PULSE_CSV, index=False)

print(f"\nDetected {len(pulse_frames)} pulses.")
print(f"Saved full trace: {OUT_CSV}")
print(f"Saved pulses only: {OUT_PULSE_CSV}")

print("\nPulse frames:")
print(pulse_frames)


# =========================
# PLOT
# =========================

plt.figure(figsize=(12, 7))

for name in camera_names:
    plt.plot(df["frame"], df[name], label=name)

plt.scatter(
    pulse_frames,
    df.loc[pulse_frames, "total"],
    marker="x",
    label="detected pulses",
)

for name in camera_names:
    plt.axhline(thresholds[name], linestyle=":", alpha=0.5)

plt.xlabel("Frame")
plt.ylabel("Integrated ROI intensity")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=200)
plt.show()


plt.figure(figsize=(12, 4))

plt.plot(df["frame"], df["pulse_score"], label="pulse score")

plt.scatter(
    pulse_frames,
    df.loc[pulse_frames, "pulse_score"],
    marker="x",
    label="detected peaks",
)

plt.xlabel("Frame")
plt.ylabel("Pulse score")
plt.legend()
plt.tight_layout()
plt.savefig("qst_pulse_score.png", dpi=200)
plt.show()