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

# If False:
#   read frame groups from INPUT_CSV
#   and open selector with those frames pre-selected
MANUAL_FRAME_SELECTION = False

INPUT_CSV = "qst_pulses_only_old.csv"

CAMERA_NAMES = ["Alice0", "Alice1", "Bob0", "Bob1"]


# =========================
# LOAD VIDEO FRAMES
# =========================

def load_frames(video_path, only_indices=None):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = {}

    wanted = None
    if only_indices is not None:
        wanted = set(only_indices)

    idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if wanted is None or idx in wanted:
            frames[idx] = frame

        idx += 1

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("No frames loaded.")

    return frames


# =========================
# READ FRAME GROUPS FROM CSV
# =========================

def read_groups_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    groups = []

    for s in df["frames"]:
        group = [int(x) for x in str(s).split(",")]
        groups.append(group)

    return groups


# =========================
# GROUP CONSECUTIVE FRAMES
# =========================

def group_consecutive_frames(selected_frames):

    if len(selected_frames) == 0:
        return []

    selected_frames = sorted(selected_frames)

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
# MANUAL FRAME SELECTION
# =========================

def select_pulse_frames(frames_dict,
                        initial_selected=None,
                        initial_reference=None):

    if initial_selected is None:
        initial_selected = []

    selected = set(initial_selected)

    reference_frame = initial_reference

    frame_indices = sorted(frames_dict.keys())

    cursor = 0
    n = len(frame_indices)

    print("\nFrame selection controls:")
    print("  right / d : next frame")
    print("  left  / a : previous frame")
    print("  space     : toggle selected/unselected")
    print("  r         : set reference frame")
    print("  q         : finish selection")

    while True:

        real_idx = frame_indices[cursor]

        frame = frames_dict[real_idx].copy()

        selected_status = (
            "SELECTED"
            if real_idx in selected
            else "not selected"
        )

        ref_status = (
            "REFERENCE FRAME"
            if real_idx == reference_frame
            else ""
        )

        color = (
            (0, 255, 0)
            if real_idx in selected
            else (0, 0, 255)
        )

        cv2.putText(
            frame,
            f"Frame {real_idx} ({cursor+1}/{n})",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
        )

        cv2.putText(
            frame,
            selected_status,
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )

        if ref_status:

            cv2.putText(
                frame,
                ref_status,
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 0),
                3,
            )

        cv2.putText(
            frame,
            "SPACE toggle | R reference | arrows move | q finish",
            (30, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Select pulse frames", frame)

        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break

        elif key == ord(" "):

            if real_idx in selected:

                selected.remove(real_idx)

                if reference_frame == real_idx:
                    reference_frame = None

                print(f"Unselected frame {real_idx}")

            else:

                selected.add(real_idx)

                print(f"Selected frame {real_idx}")

        elif key == ord("r"):

            if real_idx not in selected:

                print(
                    "Reference frame must also be selected."
                )

            else:

                reference_frame = real_idx

                print(
                    f"Reference frame set to {real_idx}"
                )

        elif key in [ord("d"), 83]:

            cursor = min(cursor + 1, n - 1)

        elif key in [ord("a"), 81]:

            cursor = max(cursor - 1, 0)

    cv2.destroyWindow("Select pulse frames")

    if len(selected) == 0:
        raise RuntimeError("No frames selected.")

    if reference_frame is None:

        # fallback:
        # first selected frame
        reference_frame = min(selected)

        print(
            f"\nNo reference frame chosen."
            f"\nUsing frame {reference_frame}."
        )

    return sorted(selected), reference_frame


# =========================
# ROI SELECTION
# =========================

def choose_rois(frame):

    rois = {}

    print("\nChoose ROIs in this order:")
    print("  Alice0, Alice1, Bob0, Bob1")

    for name in CAMERA_NAMES:

        print(f"Select ROI for {name}")

        r = cv2.selectROI(
            f"Select ROI: {name}",
            frame,
            showCrosshair=True
        )

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

if MANUAL_FRAME_SELECTION:

    # Load everything
    frames = load_frames(VIDEO_PATH)

    # Start empty
    selected_frames, reference_frame = select_pulse_frames(frames)

else:

    # Read existing groups
    existing_groups = read_groups_from_csv(INPUT_CSV)

    preselected = []

    for g in existing_groups:
        preselected.extend(g)

    preselected = sorted(set(preselected))

    # Load ONLY relevant frames
    frames = load_frames(VIDEO_PATH, only_indices=preselected)

    print(f"\nLoaded {len(frames)} relevant frames from CSV.")

    # Open selector with frames already selected
    selected_frames, reference_frame = select_pulse_frames(
        frames,
        initial_selected=preselected
    )


if len(selected_frames) == 0:
    raise RuntimeError("No pulse frames selected.")


pulse_groups = group_consecutive_frames(selected_frames)


print("\nSelected frame groups:")

for k, group in enumerate(pulse_groups):
    print(f"Pulse {k:02d}: frames {group}")

print(f"\nNumber of pulses = {len(pulse_groups)}")


# Use first pulse frame for ROI selection
roi_frame = frames[reference_frame]

print(
    f"\nUsing frame {reference_frame}"
    " for ROI selection."
)

ROIS = choose_rois(roi_frame)


records = []

for pulse_index, group in enumerate(pulse_groups):

    intensities_per_frame = []

    for frame_idx in group:

        frame = frames[frame_idx]

        row = {}

        for name in CAMERA_NAMES:
            row[name] = roi_intensity(frame, ROIS[name])

        intensities_per_frame.append(row)

    avg = {}

    for name in CAMERA_NAMES:
        avg[name] = np.mean(
            [r[name] for r in intensities_per_frame]
        )

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

        record[f"{name}_frac"] = (
            avg[name] / total if total > 0 else np.nan
        )

    records.append(record)


df = pd.DataFrame(records)

df.to_csv(OUT_CSV, index=False)

print(f"\nSaved pulse CSV to: {OUT_CSV}")

print("\nMean fractions:")

for name in CAMERA_NAMES:
    print(f"{name}: {df[f'{name}_frac'].mean():.4f}")