#!/usr/bin/env python3
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

VIDEOS = [f"videos/{i}.mp4" for i in range(1, 17)]

OUT_CSV = "threshold_counts.csv"
ROI_FILE = "rois.json"

PEAK_MERGE_FRAMES = 5
THRESH_STEP = 0.02
NORMALIZE_BY_MAX = True


def get_frame_intensity(video_path, roi):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    x, y, w, h = roi
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

    print(f"\nSelecting ROIs for {video_path}")

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

    first_video = VIDEOS[0]

    if not Path(first_video).exists():
        raise FileNotFoundError(
            f"Video file not found: {first_video}\n"
            f"Current folder is: {Path.cwd()}"
        )

    shared_rois = select_rois(first_video)

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


def count_agreements(alice_peaks, bob_peaks, tolerance=5):
    used_bob = set()
    agreements = 0

    for ap in alice_peaks:
        candidates = [
            i for i, bp in enumerate(bob_peaks)
            if i not in used_bob and abs(ap - bp) <= tolerance
        ]

        if candidates:
            best = min(candidates, key=lambda i: abs(ap - bob_peaks[i]))
            used_bob.add(best)
            agreements += 1

    return agreements


class ThresholdViewer:
    def __init__(self, data):
        self.data = data
        self.i = 0
        self.active = "alice"

        self.fig, self.ax = plt.subplots(figsize=(11, 5))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.draw()

    def draw(self):
        self.ax.clear()

        item = self.data[self.i]

        self.ax.plot(item["alice_y"], label=f"Alice threshold={item['alice_thr']:.3f}")
        self.ax.plot(item["bob_y"], label=f"Bob threshold={item['bob_thr']:.3f}")

        self.ax.axhline(item["alice_thr"], linestyle="--", label="Alice threshold")
        self.ax.axhline(item["bob_thr"], linestyle=":", label="Bob threshold")

        alice_peaks = find_peaks_grouped(item["alice_y"], item["alice_thr"], PEAK_MERGE_FRAMES)
        bob_peaks = find_peaks_grouped(item["bob_y"], item["bob_thr"], PEAK_MERGE_FRAMES)

        self.ax.scatter(alice_peaks, item["alice_y"][alice_peaks], marker="o")
        self.ax.scatter(bob_peaks, item["bob_y"][bob_peaks], marker="x")

        self.ax.set_title(
            f"Video {self.i + 1}/{len(self.data)}: {item['video']} | "
            f"active threshold: {self.active.upper()}"
        )
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Normalized intensity" if NORMALIZE_BY_MAX else "Intensity")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

        self.fig.canvas.draw_idle()

    def on_key(self, event):
        item = self.data[self.i]

        if event.key == "right":
            self.i = (self.i + 1) % len(self.data)

        elif event.key == "left":
            self.i = (self.i - 1) % len(self.data)

        elif event.key == "tab":
            self.active = "bob" if self.active == "alice" else "alice"

        elif event.key == "a":
            if self.active == "alice":
                item["alice_thr"] -= THRESH_STEP
            else:
                item["bob_thr"] -= THRESH_STEP

        elif event.key == "d":
            if self.active == "alice":
                item["alice_thr"] += THRESH_STEP
            else:
                item["bob_thr"] += THRESH_STEP

        item["alice_thr"] = max(0, item["alice_thr"])
        item["bob_thr"] = max(0, item["bob_thr"])

        self.draw()


def main():
    rois = load_or_create_rois()
    data = []

    for video_path in VIDEOS:
        alice_y = get_frame_intensity(video_path, rois[video_path]["alice"])
        bob_y = get_frame_intensity(video_path, rois[video_path]["bob"])

        video_number = Path(video_path).stem

        data.append({
            "video": video_path,
            "angle_combination": video_number,
            "alice_y": alice_y,
            "bob_y": bob_y,
            "alice_thr": 0.5,
            "bob_thr": 0.5,
        })

    viewer = ThresholdViewer(data)

    print("""
Controls:
  left/right : switch between videos
  tab        : switch active threshold Alice/Bob
  a          : lower active threshold
  d          : raise active threshold
  close plot : save CSV
""")

    plt.show()

    rows = []

    for item in data:
        alice_peaks = find_peaks_grouped(item["alice_y"], item["alice_thr"], PEAK_MERGE_FRAMES)
        bob_peaks = find_peaks_grouped(item["bob_y"], item["bob_thr"], PEAK_MERGE_FRAMES)

        agreements = count_agreements(alice_peaks, bob_peaks, PEAK_MERGE_FRAMES)

        rows.append({
            "video": item["video"],
            "angle_combination": item["angle_combination"],
            "alice_threshold": item["alice_thr"],
            "bob_threshold": item["bob_thr"],
            "alice_peak_count": len(alice_peaks),
            "bob_peak_count": len(bob_peaks),
            "agreements": agreements,
            "alice_only": len(alice_peaks) - agreements,
            "bob_only": len(bob_peaks) - agreements,
            "alice_peak_frames": list(alice_peaks),
            "bob_peak_frames": list(bob_peaks),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()