import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------

N_PAIRS = 20
N_RUNS = 10000
SEED = None
rng = np.random.default_rng(SEED)

deg = np.pi / 180

STATE_MODEL = "entangled_phi_plus"
# STATE_MODEL = "classical_mixture"

alice_angles = [-45*deg, 45*deg, 0*deg, 90*deg]
bob_angles   = [-22.5*deg, 67.5*deg, 112.5*deg, 22.5*deg]


# -----------------------------
# Basic probabilities
# -----------------------------

def p_plus_given_state(measure_angle, state_angle):
    return np.cos(measure_angle - state_angle)**2


def measure_single_photon(measure_angle, state_angle):
    p_plus = p_plus_given_state(measure_angle, state_angle)

    if rng.random() < p_plus:
        return 1, measure_angle
    else:
        return 0, measure_angle + np.pi/2


# -----------------------------
# One pair simulation
# -----------------------------

def simulate_pair(a, b):
    if STATE_MODEL == "classical_mixture":
        initial_angle = 0 if rng.random() < 0.5 else np.pi/2

        alice_state = initial_angle
        bob_state = initial_angle

        A, _ = measure_single_photon(a, alice_state)
        B, _ = measure_single_photon(b, bob_state)

        return A, B

    elif STATE_MODEL == "entangled_phi_plus":
        A = 1 if rng.random() < 0.5 else 0

        if A == 1:
            bob_state = a
        else:
            bob_state = a + np.pi/2

        B, _ = measure_single_photon(b, bob_state)

        return A, B

    else:
        raise ValueError("Unknown STATE_MODEL")


# -----------------------------
# Run one setting
# -----------------------------

def run_setting(a, b):
    A_bits = []
    B_bits = []

    for _ in range(N_PAIRS):
        A, B = simulate_pair(a, b)
        A_bits.append(A)
        B_bits.append(B)

    A_bits = np.array(A_bits)
    B_bits = np.array(B_bits)

    N_ab = int(np.sum((A_bits == 1) & (B_bits == 1)))

    return N_ab


# -----------------------------
# Correlation
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
# One full CHSH experiment
# -----------------------------

def run_one_experiment():
    counts = {}

    for a in alice_angles:
        for b in bob_angles:
            counts[(a, b)] = run_setting(a, b)

    E_m_m = E(-45*deg, -22.5*deg, counts)
    E_m_p = E(-45*deg,  22.5*deg, counts)
    E_0_m = E(0*deg,   -22.5*deg, counts)
    E_0_p = E(0*deg,    22.5*deg, counts)

    S = E_m_m - E_m_p + E_0_m + E_0_p

    return S


# -----------------------------
# Run many experiments
# -----------------------------

S_values = []

for i in range(N_RUNS):
    S = run_one_experiment()
    if not np.isnan(S):
        S_values.append(S)

S_values = np.array(S_values)

print(f"\n=== Monte Carlo CHSH simulation ===")
print(f"State model: {STATE_MODEL}")
print(f"Number of experiments: {N_RUNS}")
print(f"Pairs per setting: {N_PAIRS}")
print()
print(f"Mean S      = {np.mean(S_values):.4f}")
print(f"Std S       = {np.std(S_values):.4f}")
print(f"Median S    = {np.median(S_values):.4f}")
print(f"Min S       = {np.min(S_values):.4f}")
print(f"Max S       = {np.max(S_values):.4f}")
print(f"Mean |S|    = {np.mean(np.abs(S_values)):.4f}")
print()
print(f"Fraction with |S| > 2: {np.mean(np.abs(S_values) > 2):.4f}")
print(f"Ideal quantum value:   {2*np.sqrt(2):.4f}")


# -----------------------------
# Plot distribution
# -----------------------------

plt.figure(figsize=(8, 5))

plt.hist(S_values, bins=40, density=True, alpha=0.75)

plt.axvline(np.mean(S_values), linestyle="--", label=f"mean S = {np.mean(S_values):.3f}")
plt.axvline(2, linestyle=":", label="classical bound S = 2")
plt.axvline(-2, linestyle=":")
plt.axvline(2*np.sqrt(2), linestyle="-.", label=r"$2\sqrt{2}$")
plt.axvline(-2*np.sqrt(2), linestyle="-.")

plt.xlabel("S")
plt.ylabel("Probability density")
plt.title(f"Distribution of CHSH S over {N_RUNS} experiments\n"
          f"{N_PAIRS} pairs per setting, model = {STATE_MODEL}")
plt.legend()
plt.tight_layout()

plt.savefig("S_distribution_20_pulses.png", dpi=300)
plt.show()