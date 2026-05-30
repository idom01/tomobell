#!/usr/bin/env python3
import cv2
import numpy as np
import pandas as pd
from pathlib import Path


# =========================
# USER PARAMETERS
# =========================

VIDEO_PATH = "QST_videos/partB50.mp4"

OUT_CSV = "qst_pulses_only.csv"

CAMERA_NAMES = ["Alice0", "Alice1", "Bob0", "Bob1"]


# =========================
# LOAD VIDEO FRAMES
# =========================

def load_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("No frames found in video.")

    return frames


# =========================
# MANUALLY SELECT PULSE FRAMES
# =========================

def select_pulse_frames(frames):
    selected = set()
    i = 0
    n = len(frames)

    print("\nFrame selection controls:")
    print("  right / d : next frame")
    print("  left  / a : previous frame")
    print("  space     : toggle selected/unselected")
    print("  q         : finish selection")
    print("\nSelect every frame where you see a pulse.\n")

    while True:
        frame = frames[i].copy()

        status = "SELECTED" if i in selected else "not selected"

        cv2.putText(
            frame,
            f"Frame {i+1}/{n} | {status}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0) if i in selected else (0, 0, 255),
            2,
        )

        cv2.putText(
            frame,
            "SPACE toggle | arrows/a,d move | q finish",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Select pulse frames", frame)

        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break

        elif key == ord(" "):
            if i in selected:
                selected.remove(i)
                print(f"Unselected frame {i}")
            else:
                selected.add(i)
                print(f"Selected frame {i}")

        elif key in [ord("d"), 83]:  # d or right arrow
            i = min(i + 1, n - 1)

        elif key in [ord("a"), 81]:  # a or left arrow
            i = max(i - 1, 0)

    cv2.destroyWindow("Select pulse frames")

    return sorted(selected)


# =========================
# GROUP CONSECUTIVE FRAMES
# =========================

def group_consecutive_frames(selected_frames):
    if len(selected_frames) == 0:
        return []

    groups = []
    current = [selected_frames[0]]

    for f in selected_frames[1:]:
        if f == current[-1] + 1:
            current.append(f)
        else:
            groups.append(current)
            current = [f]

    groups.append(current)

    return groups


# =========================
# ROI SELECTION
# =========================

def choose_rois(frame):
    rois = {}

    print("\nChoose ROIs in this order:")
    print("  Alice0, Alice1, Bob0, Bob1")
    print("After selecting each ROI press ENTER or SPACE.\n")

    for name in CAMERA_NAMES:
        print(f"Select ROI for {name}")
        r = cv2.selectROI(f"Select ROI: {name}", frame, showCrosshair=True)
        cv2.destroyWindow(f"Select ROI: {name}")

        x, y, w, h = map(int, r)

        if w == 0 or h == 0:
            raise RuntimeError(f"Invalid ROI selected for {name}")

        rois[name] = (x, y, w, h)

    return rois


def roi_intensity(frame, roi):
    x, y, w, h = roi
    crop = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(np.sum(gray))


# =========================
# MAIN
# =========================

frames = load_frames(VIDEO_PATH)

selected_frames = select_pulse_frames(frames)

if len(selected_frames) == 0:
    raise RuntimeError("No pulse frames selected.")

pulse_groups = group_consecutive_frames(selected_frames)

print("\nSelected frame groups:")
for k, group in enumerate(pulse_groups):
    print(f"Pulse {k:02d}: frames {group}")

print(f"\nNumber of pulses detected by grouping = {len(pulse_groups)}")

if len(pulse_groups) != 50:
    print("\nWARNING:")
    print(f"You selected {len(pulse_groups)} pulses, not 50.")
    print("The CSV will still be created, but check your frame selection.")


# Use the first selected pulse frame to choose ROIs
first_pulse_frame = frames[pulse_groups[0][0]]
ROIS = choose_rois(first_pulse_frame)


records = []

for pulse_index, group in enumerate(pulse_groups):

    intensities_per_frame = []

    for frame_idx in group:
        frame = frames[frame_idx]

        row = {}

        for name in CAMERA_NAMES:
            row[name] = roi_intensity(frame, ROIS[name])

        intensities_per_frame.append(row)

    # Average over consecutive frames belonging to the same pulse
    avg = {}

    for name in CAMERA_NAMES:
        avg[name] = np.mean([r[name] for r in intensities_per_frame])

    total = sum(avg[name] for name in CAMERA_NAMES)

    record = {
        "pulse": pulse_index,
        "frames": ",".join(str(f) for f in group),
        "n_frames": len(group),
        "Alice0": avg["Alice0"],
        "Alice1": avg["Alice1"],
        "Bob0": avg["Bob0"],
        "Bob1": avg["Bob1"],
        "total": total,
    }

    for name in CAMERA_NAMES:
        record[f"{name}_frac"] = avg[name] / total if total > 0 else np.nan

    records.append(record)


df = pd.DataFrame(records)
df.to_csv(OUT_CSV, index=False)

print(f"\nSaved pulse CSV to: {OUT_CSV}")
print(f"Number of pulses in CSV: {len(df)}")

print("\nMean fractions:")
for name in CAMERA_NAMES:
    print(f"{name}: {df[f'{name}_frac'].mean():.4f}")