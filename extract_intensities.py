#!/usr/bin/env python3

import cv2
import numpy as np
import pandas as pd


PEAKS_CSV = "jolly_good_data.csv"
OUT_CSV = "intensities_by_frame.csv"


ROIS = {
    "videos/1.mp4": {
        "alice": [557, 79, 275, 185],
        "bob": [559, 376, 283, 188],
    },
    "videos/2.mp4": {
        "alice": [557, 79, 275, 185],
        "bob": [559, 376, 283, 188],
    },
    "videos/3.mp4": {
        "alice": [557, 79, 275, 185],
        "bob": [559, 376, 283, 188],
    },
    "videos/4.mp4": {
        "alice": [557, 79, 275, 185],
        "bob": [559, 376, 283, 188],
    },
    "videos/5.mp4": {
        "alice": [557, 79, 275, 185],
        "bob": [559, 376, 283, 188],
    },
    "videos/6.mp4": {
        "alice": [557, 79, 275, 185],
        "bob": [559, 376, 283, 188],
    },
    "videos/7.mp4": {
        "alice": [557, 79, 275, 185],
        "bob": [559, 376, 283, 188],
    },
    "videos/8.mp4": {
        "alice": [557, 79, 275, 185],
        "bob": [559, 376, 283, 188],
    },
    "videos/9.mp4": {
        "alice": [557, 79, 275, 185],
        "bob": [559, 376, 283, 188],
    },
    "videos/10.mp4": {
        "alice": [557, 79, 275, 185],
        "bob": [559, 376, 283, 188],
    },
    "videos/11.mp4": {
        "alice": [557, 79, 275, 185],
        "bob": [559, 376, 283, 188],
    },
    "videos/12.mp4": {
        "alice": [557, 79, 275, 185],
        "bob": [559, 376, 283, 188],
    },
    "videos/13.mp4": {
        "alice": [557, 79, 275, 185],
        "bob": [559, 376, 283, 188],
    },
    "videos/14.mp4": {
        "alice": [557, 79, 275, 185],
        "bob": [559, 376, 283, 188],
    },
    "videos/15.mp4": {
        "alice": [557, 79, 275, 185],
        "bob": [559, 376, 283, 188],
    },
    "videos/16.mp4": {
        "alice": [557, 79, 275, 185],
        "bob": [559, 376, 283, 188],
    },
}


def get_mean_intensity(frame, roi):
    x, y, w, h = roi
    crop = frame[y:y+h, x:x+w]

    if crop.size == 0:
        raise ValueError(f"Empty ROI crop for ROI {roi}")

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def process_video(video_path, angle_combination):
    if video_path not in ROIS:
        raise ValueError(f"No ROI defined for video: {video_path}")

    alice_roi = ROIS[video_path]["alice"]
    bob_roi = ROIS[video_path]["bob"]

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    rows = []
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        alice_intensity = get_mean_intensity(frame, alice_roi)
        bob_intensity = get_mean_intensity(frame, bob_roi)

        rows.append({
            "video": video_path,
            "angle_combination": angle_combination,
            "frame": frame_index,
            "alice_intensity": alice_intensity,
            "bob_intensity": bob_intensity,
        })

        frame_index += 1

    cap.release()

    print(f"{video_path}: extracted {frame_index} frames")
    return rows


def main():
    peaks_df = pd.read_csv(PEAKS_CSV)

    required_columns = ["video", "angle_combination"]
    for col in required_columns:
        if col not in peaks_df.columns:
            raise ValueError(f"Missing column in {PEAKS_CSV}: {col}")

    all_rows = []

    for _, row in peaks_df.iterrows():
        video_path = row["video"]
        angle_combination = row["angle_combination"]

        print(f"Processing {video_path}...")
        rows = process_video(video_path, angle_combination)
        all_rows.extend(rows)

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(OUT_CSV, index=False)

    print(f"\nSaved intensity data to: {OUT_CSV}")


if __name__ == "__main__":
    main()