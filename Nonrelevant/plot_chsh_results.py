#!/usr/bin/env python3

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


THRESHOLD_CSV = "jolly_good_data.csv"
INTENSITIES_CSV = "intensities_by_frame.csv"

OUT_DIR = Path("recreated_threshold_plots")

PEAK_MERGE_FRAMES = 5


def parse_peak_frames(s):
    if pd.isna(s):
        return np.array([], dtype=int)

    nums = re.findall(r"\d+", str(s))
    return np.array([int(x) for x in nums], dtype=int)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    threshold_df = pd.read_csv(THRESHOLD_CSV)
    intensity_df = pd.read_csv(INTENSITIES_CSV)

    for _, row in threshold_df.iterrows():
        video = row["video"]
        angle = row["angle_combination"]

        alice_thr = float(row["alice_threshold"])
        bob_thr = float(row["bob_threshold"])

        alice_peaks = parse_peak_frames(row["alice_peak_frames"])
        bob_peaks = parse_peak_frames(row["bob_peak_frames"])

        data = intensity_df[intensity_df["video"] == video].copy()
        data = data.sort_values("frame")

        frames = data["frame"].to_numpy(dtype=int)
        alice_y = data["alice_intensity"].to_numpy(dtype=float)
        bob_y = data["bob_intensity"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(11, 5))

        ax.plot(frames, alice_y, label=f"Alice threshold={alice_thr:.3f}")
        ax.plot(frames, bob_y, label=f"Bob threshold={bob_thr:.3f}")

        ax.axhline(alice_thr, linestyle="--", label="Alice threshold")
        ax.axhline(bob_thr, linestyle=":", label="Bob threshold")

        frame_to_index = {int(f): i for i, f in enumerate(frames)}

        alice_valid = [p for p in alice_peaks if p in frame_to_index]
        bob_valid = [p for p in bob_peaks if p in frame_to_index]

        if alice_valid:
            alice_idx = [frame_to_index[p] for p in alice_valid]
            ax.scatter(alice_valid, alice_y[alice_idx], marker="o")

        if bob_valid:
            bob_idx = [frame_to_index[p] for p in bob_valid]
            ax.scatter(bob_valid, bob_y[bob_idx], marker="x")

        ax.set_title(f"Video {angle}: {video}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Normalized intensity")
        ax.legend()
        ax.grid(True, alpha=0.3)

        out_path = OUT_DIR / f"video_{angle}_threshold_plot.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close(fig)

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()