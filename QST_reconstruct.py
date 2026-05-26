#!/usr/bin/env python3
import numpy as np
import pandas as pd


# =========================
# USER PARAMETERS
# =========================

CSV_PATH = "qst_pulses_only.csv"

# Assumed Bell-like phase.
# phi = 0 gives Phi+ = (HH + VV)/sqrt(2)
# phi = pi gives Phi- = (HH - VV)/sqrt(2)
PHI = 0.0

OUT_TXT = "rho_constrained_qst.txt"
OUT_NPY = "rho_constrained_qst.npy"


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(CSV_PATH)

required = ["Alice0", "Alice1", "Bob0", "Bob1"]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

A0 = df["Alice0"].to_numpy(dtype=float)
A1 = df["Alice1"].to_numpy(dtype=float)
B0 = df["Bob0"].to_numpy(dtype=float)
B1 = df["Bob1"].to_numpy(dtype=float)


# =========================
# NORMALIZE EACH PULSE
# =========================

Atot = A0 + A1
Btot = B0 + B1

valid = (Atot > 0) & (Btot > 0)

A0 = A0[valid]
A1 = A1[valid]
B0 = B0[valid]
B1 = B1[valid]

Atot = Atot[valid]
Btot = Btot[valid]

pA_H = A0 / Atot
pA_V = A1 / Atot

pB_H = B0 / Btot
pB_V = B1 / Btot


# =========================
# ESTIMATE JOINT POPULATIONS
# =========================
#
# Basis order:
#   |HH>, |HV>, |VH>, |VV>
#

P_HH = np.mean(pA_H * pB_H)
P_HV = np.mean(pA_H * pB_V)
P_VH = np.mean(pA_V * pB_H)
P_VV = np.mean(pA_V * pB_V)

P_sum = P_HH + P_HV + P_VH + P_VV

P_HH /= P_sum
P_HV /= P_sum
P_VH /= P_sum
P_VV /= P_sum


# =========================
# CONSTRAINED BELL-LIKE RECONSTRUCTION
# =========================
#
# This assumes the state is in the HH/VV coherent family:
#
#   |psi> = sqrt(P_HH)|HH> + exp(i phi) sqrt(P_VV)|VV>
#
# while keeping measured leakage into HV and VH on the diagonal.
#

coherence = np.sqrt(P_HH * P_VV) * np.exp(-1j * PHI)

rho = np.array([
    [P_HH, 0.0,  0.0,  coherence],
    [0.0,  P_HV, 0.0,  0.0],
    [0.0,  0.0,  P_VH, 0.0],
    [np.conj(coherence), 0.0, 0.0, P_VV],
], dtype=complex)

rho = 0.5 * (rho + rho.conj().T)
rho /= np.trace(rho)


# =========================
# DIAGNOSTICS
# =========================

eigvals = np.linalg.eigvalsh(rho)
purity = np.real(np.trace(rho @ rho))

rho_phi_plus = np.array([
    [0.5, 0.0, 0.0, 0.5],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0, 0.5],
], dtype=complex)

fidelity_phi_plus = np.real(np.trace(rho @ rho_phi_plus))


def print_matrix(name, M):
    print(f"\n{name}:")
    for row in M:
        print("  ".join(f"{z.real:+.6f}{z.imag:+.6f}j" for z in row))


print(f"\nUsed {len(A0)} valid pulses.")

print("\nMeasured joint populations:")
print(f"P_HH = {P_HH:.6f}")
print(f"P_HV = {P_HV:.6f}")
print(f"P_VH = {P_VH:.6f}")
print(f"P_VV = {P_VV:.6f}")

print_matrix("Constrained reconstructed rho", rho)

print("\nDiagnostics:")
print("Trace(rho) =", np.trace(rho))
print("Eigenvalues =", eigvals)
print("Purity Tr(rho^2) =", purity)
print("Fidelity with Phi+ =", fidelity_phi_plus)


# =========================
# SAVE
# =========================

np.save(OUT_NPY, rho)

with open(OUT_TXT, "w") as f:
    f.write("Constrained Bell-like QST reconstruction\n\n")
    f.write("Basis order: |HH>, |HV>, |VH>, |VV>\n\n")
    f.write(f"PHI = {PHI:.8f}\n\n")

    f.write("Measured populations:\n")
    f.write(f"P_HH = {P_HH:.8f}\n")
    f.write(f"P_HV = {P_HV:.8f}\n")
    f.write(f"P_VH = {P_VH:.8f}\n")
    f.write(f"P_VV = {P_VV:.8f}\n\n")

    f.write("rho:\n")
    for row in rho:
        f.write("  ".join(f"{z.real:+.8f}{z.imag:+.8f}j" for z in row))
        f.write("\n")

    f.write("\nDiagnostics:\n")
    f.write(f"Trace = {np.trace(rho)}\n")
    f.write(f"Purity = {purity:.8f}\n")
    f.write(f"Fidelity with Phi+ = {fidelity_phi_plus:.8f}\n")
    f.write("Eigenvalues:\n")
    for x in eigvals:
        f.write(f"{x:+.8f}\n")

print(f"\nSaved: {OUT_TXT}")
print(f"Saved: {OUT_NPY}")