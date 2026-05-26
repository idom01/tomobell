import numpy as np

# -----------------------------
# Parameters
# -----------------------------

N_PAIRS = 20
SEED = None
rng = np.random.default_rng(SEED)

deg = np.pi / 180

# Choose:
# "entangled_phi_plus" = true Bell state (|HH> + |VV>)/sqrt(2)
# "classical_mixture"  = random |HH> or |VV>, no entanglement
STATE_MODEL = "entangled_phi_plus"
#STATE_MODEL = "classical_mixture"

alice_angles = [-45*deg, 45*deg, 0*deg, 90*deg]
bob_angles   = [-22.5*deg, 67.5*deg, 112.5*deg, 22.5*deg]


# -----------------------------
# Basic probabilities
# -----------------------------

def p_plus_given_state(measure_angle, state_angle):
    """
    Probability that a photon in linear polarization |state_angle>
    passes a polarizer / is measured '+' at angle measure_angle.
    """
    return np.cos(measure_angle - state_angle)**2


def measure_single_photon(measure_angle, state_angle):
    """
    Measure one photon in basis {angle, angle+90deg}.

    Returns:
        outcome = 1 for '+', 0 for '-'
        collapsed_angle = angle if '+', angle+90deg if '-'
    """
    p_plus = p_plus_given_state(measure_angle, state_angle)

    if rng.random() < p_plus:
        return 1, measure_angle
    else:
        return 0, measure_angle + np.pi/2


# -----------------------------
# One pair simulation
# -----------------------------

def simulate_pair(a, b):
    """
    Simulate one photon pair.

    Alice is measured first.
    Then Bob's state is assigned according to the chosen model.
    """

    if STATE_MODEL == "classical_mixture":
        # Randomly create either |HH> or |VV>
        initial_angle = 0 if rng.random() < 0.5 else np.pi/2

        alice_state = initial_angle
        bob_state = initial_angle

        A, _ = measure_single_photon(a, alice_state)

        # In a classical mixture, Alice's measurement does NOT physically
        # collapse Bob into Alice's measurement basis.
        B, _ = measure_single_photon(b, bob_state)

        return A, B

    elif STATE_MODEL == "entangled_phi_plus":
        # For |Phi+> = (|HH> + |VV>)/sqrt(2),
        # Alice's result is random in any linear basis.
        A = 1 if rng.random() < 0.5 else 0

        # Collapse rule:
        # If Alice gets '+ at angle a', Bob collapses to |a>.
        # If Alice gets '- at angle a', Bob collapses to |a_perp>.
        if A == 1:
            bob_state = a
        else:
            bob_state = a + np.pi/2

        B, _ = measure_single_photon(b, bob_state)

        return A, B

    else:
        raise ValueError("Unknown STATE_MODEL")


def run_setting(a, b):
    """
    Run 20 photon pairs for one setting.
    Returns Alice bits, Bob bits, and N(a,b), where N is number of ++ events.
    """
    A_bits = []
    B_bits = []

    for _ in range(N_PAIRS):
        A, B = simulate_pair(a, b)
        A_bits.append(A)
        B_bits.append(B)

    A_bits = np.array(A_bits)
    B_bits = np.array(B_bits)

    N_ab = int(np.sum((A_bits == 1) & (B_bits == 1)))

    return A_bits, B_bits, N_ab


# -----------------------------
# CHSH calculation
# -----------------------------

def E(a, b, counts):
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
# Run all 16 settings
# -----------------------------

datasets = {}
counts = {}

print(f"\n=== Quantum CHSH simulation: {STATE_MODEL} ===\n")

for a in alice_angles:
    for b in bob_angles:
        A, B, N_ab = run_setting(a, b)

        datasets[(a, b)] = (A, B)
        counts[(a, b)] = N_ab

        print(f"a = {a/deg:+6.1f} deg, b = {b/deg:+6.1f} deg")
        print(f"Alice bits: {A.tolist()}")
        print(f"Bob bits:   {B.tolist()}")
        print(f"N(a,b) = #(++ events) = {N_ab}")
        print("-" * 70)


# -----------------------------
# Correlations and S
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
print(f"Ideal quantum value: {2*np.sqrt(2):.4f}")