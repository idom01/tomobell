import numpy as np

# -----------------------------
# Parameters
# -----------------------------

N_PULSES = 2000

USE_NOISE = True

# If SEED = None, every run is different.
# If SEED = 1, 2, 3, ... the run is reproducible.
SEED = None

# Detection thresholds
THRESHOLD_A = 1
THRESHOLD_B = 1

# Mean gains after beam splitter, camera sensitivity, etc.
GAIN_A = 1.0
GAIN_B = 1.0

# Background camera offset
BACKGROUND_A = 0.02
BACKGROUND_B = 0.02

# Additive camera/readout noise
READ_NOISE_A = 0.03
READ_NOISE_B = 0.03

# Pulse-to-pulse laser energy fluctuations.
# This is multiplicative and common to Alice and Bob.
PULSE_SIGMA = 0.35

rng = np.random.default_rng(SEED)


# -----------------------------
# Angles
# -----------------------------

deg = np.pi / 180

alice_angles = [-45*deg, 45*deg, 0*deg, 90*deg]
bob_angles   = [-22.5*deg, 67.5*deg, 112.5*deg, 22.5*deg]


# -----------------------------
# Physics / detector model
# -----------------------------

def malus_intensity(input_bit, theta):
    """
    input_bit = 0 means H
    input_bit = 1 means V

    Polarizer angle theta measured relative to H.
    """
    if input_bit == 0:
        return np.cos(theta)**2
    else:
        return np.sin(theta)**2


def measured_intensity(ideal_intensity, arm):
    """
    Convert ideal Malus-law intensity into measured camera intensity.
    """

    if not USE_NOISE:
        return ideal_intensity

    # Shared pulse energy should be applied outside this function.
    # This function only adds arm-dependent gain/background/readout noise.
    if arm == "A":
        return (
            GAIN_A * ideal_intensity
            + BACKGROUND_A
            + rng.normal(0, READ_NOISE_A)
        )

    if arm == "B":
        return (
            GAIN_B * ideal_intensity
            + BACKGROUND_B
            + rng.normal(0, READ_NOISE_B)
        )

    raise ValueError("arm must be 'A' or 'B'")


def run_setting(a, b):
    """
    Simulate one pair of polarizer settings.
    Returns Alice binary data, Bob binary data, and coincidence count N(a,b).
    """

    source_bits = rng.integers(0, 2, size=N_PULSES)

    alice_bits = []
    bob_bits = []

    alice_intensities = []
    bob_intensities = []

    for bit in source_bits:

        IA_ideal = malus_intensity(bit, a)
        IB_ideal = malus_intensity(bit, b)

        if USE_NOISE:
            # Mean-normalized lognormal pulse fluctuation.
            # Common to Alice and Bob because both beams come from same laser pulse.
            pulse_energy = rng.lognormal(
                mean=-0.5 * PULSE_SIGMA**2,
                sigma=PULSE_SIGMA
            )

            IA_ideal *= pulse_energy
            IB_ideal *= pulse_energy

        IA = measured_intensity(IA_ideal, "A")
        IB = measured_intensity(IB_ideal, "B")

        A = int(IA > THRESHOLD_A)
        B = int(IB > THRESHOLD_B)

        alice_intensities.append(IA)
        bob_intensities.append(IB)

        alice_bits.append(A)
        bob_bits.append(B)

    alice_bits = np.array(alice_bits)
    bob_bits = np.array(bob_bits)

    N_ab = np.sum((alice_bits == 1) & (bob_bits == 1))

    return alice_bits, bob_bits, N_ab, np.array(alice_intensities), np.array(bob_intensities)


def E(a, b, counts):
    """
    CHSH correlation:

    E(a,b) =
    [N(a,b) + N(a_perp,b_perp) - N(a_perp,b) - N(a,b_perp)]
    /
    [N(a,b) + N(a_perp,b_perp) + N(a_perp,b) + N(a,b_perp)]
    """

    ap = a + np.pi/2
    bp = b + np.pi/2

    N_ab   = counts[(a,  b)]
    N_apbp = counts[(ap, bp)]
    N_apb  = counts[(ap, b)]
    N_abp  = counts[(a,  bp)]

    denom = N_ab + N_apbp + N_apb + N_abp

    if denom == 0:
        return np.nan

    return (N_ab + N_apbp - N_apb - N_abp) / denom


# -----------------------------
# Run simulation
# -----------------------------

datasets = {}
intensity_data = {}
counts = {}

print("\n=== Binary datasets and coincidence counts ===\n")

for a in alice_angles:
    for b in bob_angles:

        A, B, N_ab, IA, IB = run_setting(a, b)

        datasets[(a, b)] = (A, B)
        intensity_data[(a, b)] = (IA, IB)
        counts[(a, b)] = N_ab

        print(f"a = {a/deg:+6.1f} deg, b = {b/deg:+6.1f} deg")
        print(f"Alice bits: {A.tolist()}")
        print(f"Bob bits:   {B.tolist()}")
        print(f"N(a,b) = {N_ab}")

        if USE_NOISE:
            print(f"Alice intensities: {np.round(IA, 3).tolist()}")
            print(f"Bob intensities:   {np.round(IB, 3).tolist()}")

        print("-" * 70)


# -----------------------------
# CHSH values
# -----------------------------

E_m_m = E(-45*deg, -22.5*deg, counts)
E_m_p = E(-45*deg,  22.5*deg, counts)
E_0_m = E(0*deg,   -22.5*deg, counts)
E_0_p = E(0*deg,    22.5*deg, counts)

S = E_m_m - E_m_p + E_0_m + E_0_p

print("\n=== Correlations ===\n")
print(f"E(-45, -22.5) = {E_m_m:.4f}")
print(f"E(-45, +22.5) = {E_m_p:.4f}")
print(f"E(0,   -22.5) = {E_0_m:.4f}")
print(f"E(0,   +22.5) = {E_0_p:.4f}")

print("\n=== CHSH parameter ===\n")
print(f"S   = {S:.4f}")
print(f"|S| = {abs(S):.4f}")