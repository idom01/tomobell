import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

# -----------------------------
# User input: put your 4x4 matrix here
# -----------------------------
data = np.array([
    [0.00, 0.00, 0.00, 0.00],
    [0.00, 0.31, 0.46, 0.00],
    [0.00, 0.46, 0.69, 0.00],
    [0.00, 0.00, 0.00, 0.00]
], dtype=float)

title_figure = "HVVH 10"
output_pdf = title_figure + ".pdf"

# -----------------------------
# Checks
# -----------------------------
if data.shape != (4, 4):
    raise ValueError("data must be a 4 by 4 matrix")


x_labels = ["HH", "HV", "VH", "VV"]
y_labels = ["HH", "HV", "VH", "VV"]

# -----------------------------
# 3D bar plot
# -----------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

n = 4
x, y = np.meshgrid(np.arange(n), np.arange(n))

x = x.ravel()
y = y.ravel()
z = np.zeros_like(x)

dx = dy = 0.65
dz = data.ravel()

norm = Normalize(vmin=np.min(dz), vmax=np.max(dz))
colors = cm.viridis(norm(dz))

ax.bar3d(x, y, z, dx, dy, dz, color=colors, shade=True)

# Put ticks at the centers of the bars
ax.set_xticks(np.arange(n) + dx / 2)
ax.set_yticks(np.arange(n) + dy / 2)

ax.set_xticklabels(x_labels, fontsize=12, fontweight="bold")
ax.set_yticklabels(y_labels, fontsize=12, fontweight="bold")

#ax.set_xlabel("State", fontsize=14, fontweight="bold", labelpad=12)
#ax.set_ylabel("Measured state", fontsize=14, fontweight="bold", labelpad=12)
#ax.set_zlabel("Value", fontsize=14, fontweight="bold", labelpad=10)

ax.set_title(title_figure, fontsize=18, fontweight="bold", pad=20)

# Nice z ticks, similar to MATLAB logic
high_num = np.max(data)
ax.set_zticks(np.round(np.linspace(0, high_num, 4), 2))

# Viewing angle; adjust if desired
ax.view_init(elev=25, azim=-55)

plt.tight_layout()
plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
plt.show()
#plt.close()


print(f"Saved figure to {output_pdf}")