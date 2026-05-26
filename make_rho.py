#!/usr/bin/env python3
import numpy as np
import pandas as pd


CSV_PATH = "good_data.csv"

OUT_TXT = "rho_average_state_method.txt"
OUT_NPY = "rho_average_state_method.npy"


df = pd.read_csv(CSV_PATH)

required = ["Alice0", "Alice1", "Bob0", "Bob1"]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")


# =========================
# NORMALIZE EACH CAMERA BY ITS MAX
# =========================

for col in required:
    max_val = df[col].max()
    if max_val <= 0:
        raise ValueError(f"Column {col} has non-positive max intensity.")

    df[col + "_norm"] = df[col] / max_val

print("\nCamera max intensities:")
for col in required:
    print(f"{col}: {df[col].max():.6f}")


# Basis order:
# |HH>, |HV>, |VH>, |VV>
basis_vectors = {
    "HH": np.array([1, 0, 0, 0], dtype=complex),
    "HV": np.array([0, 1, 0, 0], dtype=complex),
    "VH": np.array([0, 0, 1, 0], dtype=complex),
    "VV": np.array([0, 0, 0, 1], dtype=complex),
}


states = []
labels = []

for _, row in df.iterrows():

    A0 = float(row["Alice0_norm"])
    A1 = float(row["Alice1_norm"])
    B0 = float(row["Bob0_norm"])
    B1 = float(row["Bob1_norm"])

    if A0 + A1 <= 0 or B0 + B1 <= 0:
        continue

    # Classify after camera-wise max normalization
    alice_state = "H" if A0 >= A1 else "V"
    bob_state   = "H" if B0 >= B1 else "V"

    label = alice_state + bob_state

    states.append(basis_vectors[label])
    labels.append(label)


if len(states) == 0:
    raise RuntimeError("No valid pulses found.")


states = np.array(states)

# Coherent average of classified state vectors
psi_avg = np.mean(states, axis=0)

norm = np.linalg.norm(psi_avg)

if norm == 0:
    raise RuntimeError("Average state has zero norm.")

psi_avg = psi_avg / norm

rho = np.outer(psi_avg, psi_avg.conj())

eigvals = np.linalg.eigvalsh(rho)
purity = np.real(np.trace(rho @ rho))


counts = pd.Series(labels).value_counts()

print(f"\nUsed {len(states)} pulses.")

print("\nDetected pulse states:")
for label in ["HH", "HV", "VH", "VV"]:
    print(f"{label}: {counts.get(label, 0)}")

print("\nAverage normalized state vector:")
print("basis order: |HH>, |HV>, |VH>, |VV>")
for label, amp in zip(["HH", "HV", "VH", "VV"], psi_avg):
    print(f"{label}: {amp.real:+.6f}{amp.imag:+.6f}j")

print("\nDensity matrix rho:")
for row in rho:
    print("  ".join(f"{z.real:+.6f}{z.imag:+.6f}j" for z in row))

print("\nDiagnostics:")
print("Trace =", np.trace(rho))
print("Eigenvalues =", eigvals)
print("Purity =", purity)


np.save(OUT_NPY, rho)

with open(OUT_TXT, "w") as f:
    f.write("Average-state constrained reconstruction\n\n")
    f.write("Classification: each camera normalized by its max intensity over CSV\n")
    f.write("Basis order: |HH>, |HV>, |VH>, |VV>\n\n")

    f.write("Camera max intensities:\n")
    for col in required:
        f.write(f"{col}: {df[col].max():.8f}\n")

    f.write("\nCounts:\n")
    for label in ["HH", "HV", "VH", "VV"]:
        f.write(f"{label}: {counts.get(label, 0)}\n")

    f.write("\nAverage normalized state vector:\n")
    for label, amp in zip(["HH", "HV", "VH", "VV"], psi_avg):
        f.write(f"{label}: {amp.real:+.8f}{amp.imag:+.8f}j\n")

    f.write("\nDensity matrix rho:\n")
    for row in rho:
        f.write("  ".join(f"{z.real:+.8f}{z.imag:+.8f}j" for z in row))
        f.write("\n")

    f.write("\nDiagnostics:\n")
    f.write(f"Trace = {np.trace(rho)}\n")
    f.write(f"Purity = {purity:.8f}\n")
    f.write("Eigenvalues:\n")
    for x in eigvals:
        f.write(f"{x:+.8f}\n")

print(f"\nSaved: {OUT_TXT}")
print(f"Saved: {OUT_NPY}")