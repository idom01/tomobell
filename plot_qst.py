#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


CSV_PATH = "qst_pulses_only50_wow.csv"

df = pd.read_csv(CSV_PATH)

required = ["pulse", "Alice0", "Alice1", "Bob0", "Bob1"]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")


# Normalize each camera by its own max
for cam in ["Alice0", "Alice1", "Bob0", "Bob1"]:
    df[cam + "_norm"] = df[cam] / df[cam].max()


x = np.arange(len(df))
bar_width = 0.35

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)


# Alice plot
axes[0].bar(
    x - bar_width / 2,
    df["Alice1_norm"],   # H = 1
    width=bar_width,
    color="red",
    label="H = 1",
)

axes[0].bar(
    x + bar_width / 2,
    df["Alice0_norm"],   # V = 0
    width=bar_width,
    color="blue",
    label="V = 0",
)

axes[0].set_title("Alice Normalized Intensities")
axes[0].set_ylabel("Normalized Intensity")
axes[0].legend()
axes[0].set_ylim(0, 1.05)


# Bob plot
axes[1].bar(
    x - bar_width / 2,
    df["Bob1_norm"],     # H = 1
    width=bar_width,
    color="red",
    label="H = 1",
)

axes[1].bar(
    x + bar_width / 2,
    df["Bob0_norm"],     # V = 0
    width=bar_width,
    color="blue",
    label="V = 0",
)

axes[1].set_title("Bob Normalized Intensities")
axes[1].set_xlabel("Pulse")
axes[1].set_ylabel("Normalized Intensity")
axes[1].legend()
axes[1].set_ylim(0, 1.05)


# Pulse labels
axes[1].set_xticks(x)
# axes[0].set_xlim(0.5, len(df) + 0.5)
# axes[1].set_xlim(0.5, len(df) + 0.5)
axes[1].set_xticklabels(df["pulse"] + 1)

plt.tight_layout()
plt.show()