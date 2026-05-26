import numpy as np

# Based on the main-paper CHSH emulation setup link provided in sources:
# pulsed laser -> HWP/random H/V preparation -> BS -> Alice/Bob polarizers -> cameras.
# Source: :contentReference[oaicite:0]{index=0}

N_PULSES = 20
THRESHOLD = 0.4
SEED = 1

rng = np.random.default_rng()

# CHSH base angles
a_vals = [-np.pi/4, 0]
b_vals = [-np.pi/8, np.pi/8]

# Include perpendicular settings, giving 4 x 4 = 16 measured settings
alice_angles = [-np.pi/4, np.pi/4, 0, np.pi/2]
bob_angles   = [-np.pi/8, 3*np.pi/8, np.pi/8, 5*np.pi/8]


def intensity_after_polarizer(input_bit, theta):
    """
    input_bit = 1 means H polarization
    input_bit = 0 means V polarization

    H intensity after polarizer angle theta: cos^2(theta)
    V intensity after polarizer angle theta: sin^2(theta)
    """
    if input_bit == 1:
        return np.cos(theta)**2
    else:
        return np.sin(theta)**2


def threshold_detector(intensity):
    return 1 if intensity >= THRESHOLD else 0


def run_setting(a, b):
    """
    Simulate 20 pulses for one Alice/Bob polarizer setting.
    The source randomly emits H or V, shared by Alice and Bob after the BS.
    """
    source_bits = rng.integers(0, 2, size=N_PULSES)

    alice_data = []
    bob_data = []

    for bit in source_bits:
        IA = intensity_after_polarizer(bit, a)
        IB = intensity_after_polarizer(bit, b)

        alice_data.append(threshold_detector(IA))
        bob_data.append(threshold_detector(IB))

    alice_data = np.array(alice_data)
    bob_data = np.array(bob_data)

    # N(a,b): number of pulses where both Alice and Bob output 1
    N_ab = np.sum((alice_data == 1) & (bob_data == 1))

    return alice_data, bob_data, N_ab


def E(a, b, counts):
    """
    CHSH-style correlation from the four coincidence counts:

    E(a,b) =
    [N(a,b) + N(a_perp,b_perp) - N(a,b_perp) - N(a_perp,b)]
    / total
    """
    ap = a + np.pi/2
    bp = b + np.pi/2

    N_ab = counts[(a, b)]
    N_apbp = counts[(ap, bp)]
    N_abp = counts[(a, bp)]
    N_apb = counts[(ap, b)]

    total = N_ab + N_apbp + N_abp + N_apb

    if total == 0:
        return np.nan

    return (N_ab + N_apbp - N_abp - N_apb) / total


datasets = {}
counts = {}

print("\n=== Raw binary data for 16 settings ===\n")

for a in alice_angles:
    for b in bob_angles:
        A, B, N_ab = run_setting(a, b)

        datasets[(a, b)] = (A, B)
        counts[(a, b)] = N_ab

        print(f"a = {180/np.pi * a:+.4f}, b = {180/np.pi * b:+.4f}")
        print(f"Alice: {A.tolist()}")
        print(f"Bob:   {B.tolist()}")
        print(f"N(a,b) = # simultaneous 1's = {N_ab}")
        print("-" * 60)


print("\n=== CHSH correlations ===\n")

E_m_m = E(-np.pi/4, -np.pi/8, counts)
E_m_p = E(-np.pi/4,  np.pi/8, counts)
E_0_m = E(0,        -np.pi/8, counts)
E_0_p = E(0,         np.pi/8, counts)

print(f"E(-45, -22.5) = {E_m_m:.4f}")
print(f"E(-45, 22.5) = {E_m_p:.4f}")
print(f"E(0, -22.5) = {E_0_m:.4f}")
print(f"E(0, 22.5) = {E_0_p:.4f}")

S = E_m_m - E_m_p + E_0_m + E_0_p

print("\n=== CHSH S parameter ===\n")
print(f"S = {S:.4f}")
print(f"|S| = {abs(S):.4f}")