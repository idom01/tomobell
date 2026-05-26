#!/usr/bin/env python3
import pandas as pd

CSV_FILE = "jolly_good_data.csv"


def E(hh, hv, vh, vv):
    total = hh + hv + vh + vv

    if total == 0:
        return 0.0

    return (hh + vv - hv - vh) / total


df = pd.read_csv(CSV_FILE)

A = df["agreements"].values

# =========================================================
# ORDER:
#
#  0  (a,    b)
#  1  (a,    b')
#  2  (a,    b_p)
#  3  (a,    b'_p)
#
#  4  (a',   b)
#  5  (a',   b')
#  6  (a',   b_p)
#  7  (a',   b'_p)
#
#  8  (a_p,  b)
#  9  (a_p,  b')
# 10  (a_p,  b_p)
# 11  (a_p,  b'_p)
#
# 12  (a'_p, b)
# 13  (a'_p, b')
# 14  (a'_p, b_p)
# 15  (a'_p, b'_p)
#
# =========================================================

# ===================== (a,b) ======================

N_HH_ab = A[0]
N_HV_ab = A[2]
N_VH_ab = A[8]
N_VV_ab = A[10]

# ===================== (a,b') =====================

N_HH_abp = A[1]
N_HV_abp = A[3]
N_VH_abp = A[9]
N_VV_abp = A[11]

# ===================== (a',b) =====================

N_HH_apb = A[4]
N_HV_apb = A[6]
N_VH_apb = A[12]
N_VV_apb = A[14]

# ===================== (a',b') ====================

N_HH_apbp = A[5]
N_HV_apbp = A[7]
N_VH_apbp = A[13]
N_VV_apbp = A[15]

# ==================================================
# PRINT N VALUES
# ==================================================

print("\n=============== N VALUES ===============\n")

print("(a,b)")
print(f"N_HH = {N_HH_ab}")
print(f"N_HV = {N_HV_ab}")
print(f"N_VH = {N_VH_ab}")
print(f"N_VV = {N_VV_ab}\n")

print("(a,b')")
print(f"N_HH = {N_HH_abp}")
print(f"N_HV = {N_HV_abp}")
print(f"N_VH = {N_VH_abp}")
print(f"N_VV = {N_VV_abp}\n")

print("(a',b)")
print(f"N_HH = {N_HH_apb}")
print(f"N_HV = {N_HV_apb}")
print(f"N_VH = {N_VH_apb}")
print(f"N_VV = {N_VV_apb}\n")

print("(a',b')")
print(f"N_HH = {N_HH_apbp}")
print(f"N_HV = {N_HV_apbp}")
print(f"N_VH = {N_VH_apbp}")
print(f"N_VV = {N_VV_apbp}\n")

# ==================================================
# E VALUES
# ==================================================

E_ab = E(N_HH_ab, N_HV_ab, N_VH_ab, N_VV_ab)

E_abp = E(
    N_HH_abp,
    N_HV_abp,
    N_VH_abp,
    N_VV_abp
)

E_apb = E(
    N_HH_apb,
    N_HV_apb,
    N_VH_apb,
    N_VV_apb
)

E_apbp = E(
    N_HH_apbp,
    N_HV_apbp,
    N_VH_apbp,
    N_VV_apbp
)

print("=============== E VALUES ===============\n")

print(f"E(a,b)   = {E_ab:.6f}")
print(f"E(a,b')  = {E_abp:.6f}")
print(f"E(a',b)  = {E_apb:.6f}")
print(f"E(a',b') = {E_apbp:.6f}")

# ==================================================
# CHSH S VALUE
# ==================================================

S = abs(E_ab - E_abp + E_apb + E_apbp)

print("\n=============== S VALUE ================\n")

print(f"S = {S:.6f}")

print("\n========================================\n")