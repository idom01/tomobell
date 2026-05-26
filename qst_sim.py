#!/usr/bin/env python3
import numpy as np
import pandas as pd


# =========================
# PARAMETERS
# =========================

N = 50
I0 = 1.0
NOISE = 0.02

OUT_CSV = "simulated_qst_pulses_only.csv"


# =========================
# SIMULATE RANDOM H/V PULSES
# =========================

rows = []

for k in range(N):

    bit = np.random.choice([0, 1])

    if bit == 0:
        # H pulse -> Alice H, Bob H
        state = "HH"

        Alice0 = I0 + NOISE * np.random.rand()
        Alice1 = NOISE * np.random.rand()

        Bob0 = I0 + NOISE * np.random.rand()
        Bob1 = NOISE * np.random.rand()

    else:
        # V pulse -> Alice V, Bob V
        state = "VV"

        Alice0 = NOISE * np.random.rand()
        Alice1 = I0 + NOISE * np.random.rand()

        Bob0 = NOISE * np.random.rand()
        Bob1 = I0 + NOISE * np.random.rand()

    rows.append({
        "pulse": k,
        "true_state": state,
        "Alice0": Alice0,
        "Alice1": Alice1,
        "Bob0": Bob0,
        "Bob1": Bob1,
    })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print(f"Saved simulated pulses to {OUT_CSV}")


# =========================
# H/V POPULATIONS FROM CAMERAS
# =========================

A0 = df["Alice0"].to_numpy()
A1 = df["Alice1"].to_numpy()
B0 = df["Bob0"].to_numpy()
B1 = df["Bob1"].to_numpy()

pA0 = A0 / (A0 + A1)
pA1 = A1 / (A0 + A1)

pB0 = B0 / (B0 + B1)
pB1 = B1 / (B0 + B1)

P_HH = np.mean(pA0 * pB0)
P_HV = np.mean(pA0 * pB1)
P_VH = np.mean(pA1 * pB0)
P_VV = np.mean(pA1 * pB1)

norm = P_HH + P_HV + P_VH + P_VV

P_HH /= norm
P_HV /= norm
P_VH /= norm
P_VV /= norm

print("\nH/V probabilities inferred from cameras:")
print(f"P_HH = {P_HH:.4f}")
print(f"P_HV = {P_HV:.4f}")
print(f"P_VH = {P_VH:.4f}")
print(f"P_VV = {P_VV:.4f}")


# =========================
# PAPER-STYLE QST ASSUMPTION
# =========================
#
# For the Phi+ emulation:
#
#   |Phi+> = (|HH> + |VV>) / sqrt(2)
#
# H/V data gives:
#   P_HH ≈ 1/2
#   P_VV ≈ 1/2
#
# D/A data is assumed phase-coherent with phi = 0, so:
#   E_XX = +1
#
# With real phase phi = 0:
#   rho_14 = rho_41 = E_XX / 2
#
# In the ideal case E_XX = 1, so rho_14 = rho_41 = 1/2.
#

E_XX = 1.0

rho_paper = np.array([
    [P_HH, 0.0,  0.0,  0.5 * E_XX],
    [0.0,  P_HV, 0.0,  0.0],
    [0.0,  0.0,  P_VH, 0.0],
    [0.5 * E_XX, 0.0, 0.0, P_VV],
], dtype=complex)

rho_paper /= np.trace(rho_paper)


# =========================
# CAMERA-ONLY RECONSTRUCTION
# =========================

rho_camera_only = np.diag([P_HH, P_HV, P_VH, P_VV]).astype(complex)


# =========================
# IDEAL TARGET
# =========================

rho_phi_plus = np.array([
    [0.5, 0.0, 0.0, 0.5],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0, 0.5],
], dtype=complex)


# =========================
# PRINT
# =========================

def print_matrix(name, M):
    print(f"\n{name}:")
    for row in M:
        print("  ".join(f"{z.real:+.4f}{z.imag:+.4f}j" for z in row))
    print("eigenvalues:", np.linalg.eigvalsh(M))


print("\nCounts:")
print(df["true_state"].value_counts())

print_matrix("Camera-only density matrix", rho_camera_only)
print_matrix("Paper-style reconstructed density matrix", rho_paper)
print_matrix("Ideal Phi+ density matrix", rho_phi_plus)