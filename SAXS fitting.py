"""
saxs_beaucage_twolevel_hard_carbon.py

SAXS analysis for hard carbon using a TWO-level Beaucage unified model.

Data format:
    q (1/Angstrom), I_subtracted (abs.units*thickness), sigma

Two-level Beaucage model (Beaucage 1995):
    I(q) = I_1(q) + I_2(q)

    I_i(q) = G_i * exp(-q^2 * Rg_i^2 / 3)
           + B_i * ( [erf(q*Rg_i/sqrt(6))^3] / q )^P_i

Level-1: large-scale aggregates (low-q curvature)
Level-2: nano-pores / closed pores (mid-q "knee" & power-law tail)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.special import erf
import os

# ================= User Parameters =================
infile   = r"path.txt" # Path to the original SAXS data csv or txt
out_dir  = r"path\to\output_folder"  # Directory to save all output results
out_prefix = "SAXS_beaucage_2level"

# overall intensity scaling (keep 1.0 if data already in abs. units)
intensity_scale = 1.0

# ------------- mid-q power-law window (for level-2 init) -------------

pow_q_min = 0.09   # Å^-1
pow_q_max = 0.4   # Å^-1

# ------------- Maximum q value used for global fitting ----------------

# defines the q-range used for the two-level Beaucage fit
fit_q_max = 0.35   # Å^-1: If set too high, WAXS contributions will be mixed in; decrease/increase as needed.


# ---------- Physical parameters for Porod invariant ----------
# Δρ: scattering length density contrast (cm^-2)
#   hard-carbon skeleton ρ_wall ≈ 2.0 g/cm^3, pores ~ vacuum
#   → ΔSLD ≈ 1.69×10^11 cm^-2 
delta_rho = 1.69e11 # this value depends on the specific materials

# ρ_solid (g/cm^3): used to convert porosity φ into specific pore volume (cm^3/g)
rho_solid = 1.6
# ====================================================

os.makedirs(out_dir, exist_ok=True)
out_prefix_path = os.path.join(out_dir, out_prefix)


# -------------------- I/O --------------------------
def read_three_column(path):
    """Read q, I, sigma from a three-column text/csv file."""
    data = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#") or line[0].isalpha():
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                q = float(parts[0])
                I = float(parts[1])
                s = float(parts[2]) if len(parts) >= 3 else 1.0
                if q > 0 and I > 0:
                    data.append((q, I, s))
            except ValueError:
                continue
    if not data:
        raise ValueError("No valid numeric data found in file.")
    arr = np.array(data)
    arr = arr[np.argsort(arr[:, 0])]  # sort by q
    return arr[:, 0], arr[:, 1], arr[:, 2]


# --------------- Beaucage Unified Model (single level) ------------
def beaucage_level(q, G, Rg, B, P):
    """
    Single-level Beaucage unified function:
        I(q) = G * exp(-q^2 Rg^2 / 3)
             + B * ( [erf(q*Rg/sqrt(6))^3] / q )^P
    """
    q = np.asarray(q)
    q_safe = np.where(q <= 0, 1e-12, q)

    # Guinier-like term
    expo = - (Rg**2) * q**2 / 3.0
    expo = np.clip(expo, -200, 50)
    term1 = G * np.exp(expo)

    # structural power-law term
    arg = q * Rg / np.sqrt(6.0)
    arg = np.clip(arg, -20.0, 20.0)
    pw = (erf(arg)**3) / q_safe
    pw = np.where(pw > 0, pw, 1e-30)
    term2 = B * pw**P

    return term1 + term2


def beaucage_twolevel(q,
                      G1, Rg1, B1, P1,
                      G2, Rg2, B2, P2):
    """Sum of two Beaucage levels."""
    return (beaucage_level(q, G1, Rg1, B1, P1) +
            beaucage_level(q, G2, Rg2, B2, P2))


# ---------  Utility function: find the straightest segment on the log–log curve (for level-2 initialization) ----------
def find_power_law_window(q, I, window_points=60):
    logq = np.log(q)
    logI = np.log(I)

    n = len(q)
    if n < window_points + 5:
        window_points = max(10, n // 4)

    best_R2 = -np.inf
    best_i0, best_i1 = 0, window_points

    for i0 in range(0, n - window_points + 1):
        i1 = i0 + window_points
        x = logq[i0:i1]
        y = logI[i0:i1]
        A = np.vstack([x, np.ones_like(x)]).T
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        y_fit = A @ coef
        ss_res = np.sum((y - y_fit)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        if ss_tot <= 0:
            continue
        R2 = 1.0 - ss_res / ss_tot
        if R2 > best_R2:
            best_R2 = R2
            best_i0, best_i1 = i0, i1

    return best_i0, best_i1, best_R2


# --------- Utility function: low-q Guinier estimation (for level-1 initialization) ----------
def estimate_lowq_guinier(q, I, q_max_for_low=0.03):
     """
    Fit the Guinier region in the linear ln(I) vs q^2 plot to obtain initial
    estimates of G1 and Rg1 for level-1. The parameter q_max_for_low can be
    adjusted depending on the data quality and range.
    """
    mask = (q > 0) & (q <= q_max_for_low)
    if mask.sum() < 8:
        mask = (q > 0) & (q <= q_max_for_low * 1.5)
    if mask.sum() < 5:
        idx = np.argsort(q)[:20]
        mask = np.zeros_like(q, dtype=bool)
        mask[idx] = True

    q_low = q[mask]
    I_low = I[mask]

    x = q_low**2
    y = np.log(I_low)
    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    if m < 0:
        Rg_est = np.sqrt(-3.0 * m)
    else:
        Rg_est = 80.0  # fallback

    G_est = float(np.exp(c))

    # Constrain Rg1 to a reasonable range: for large-scale structures, Rg1 is usually larger than the nanopore Rg2
    Rg_est = float(np.clip(Rg_est, 30.0, 500.0))

    return G_est, Rg_est


# ---------------------- Fitting (two-level) -----------------------------
def fit_beaucage_twolevel(q, I, sigma):
    """
    Two-level Beaucage fit:
    level-1: estimate_lowq_guinier)
    level-2: pow_q_min ~ pow_q_max)
    """

    # ----- Level-1 init from low-q Guinier -----
    G1_init, Rg1_init = estimate_lowq_guinier(q, I, q_max_for_low=0.03)
    P1_init = 4.0       
    B1_init = 1e-3      

    # ----- Level-2 init from mid-q power-law -----
    mask_pow = (q >= pow_q_min) & (q <= pow_q_max)
    if mask_pow.sum() < 80:
        print("Warning: not enough points in mid-q range; using full q-range for level-2 window search.")
        mask_pow = np.ones_like(q, dtype=bool)

    q_pow = q[mask_pow]
    I_pow = I[mask_pow]
    idx_global = np.where(mask_pow)[0]

    i0_loc, i1_loc, R2 = find_power_law_window(q_pow, I_pow, window_points=80)
    i0 = idx_global[i0_loc]
    i1 = idx_global[i1_loc]

    logq = np.log(q)
    logI = np.log(I)
    x_win = logq[i0:i1]
    y_win = logI[i0:i1]
    A = np.vstack([x_win, np.ones_like(x_win)]).T
    (slope, intercept), _, _, _ = np.linalg.lstsq(A, y_win, rcond=None)

    P2_init = -slope
    P2_init = float(np.clip(P2_init, 2.0, 6.0))

    q_knee2 = q[(i0 + i1) // 2]
    Rg2_init = 1.0 / max(q_knee2, 1e-3)   # ~ 1/q_knee
    Rg2_init = float(np.clip(Rg2_init, 5.0, 200.0))

    q_mid = q[(i0 + i1) // 2]
    I_mid = I[(i0 + i1) // 2]
    B2_init = I_mid * (q_mid**P2_init)
    B2_init = float(max(B2_init, 1e-10))
    G2_init = I_mid   

    print(f"Level-2 power-law window (limited to {pow_q_min}-{pow_q_max} Å⁻¹): "
          f"index {i0}–{i1} (R² ≈ {R2:.4f})")
    print(f"  slope ≈ {slope:.3f} → P2_init ≈ {P2_init:.2f}")
    print(f"  q_knee2 ≈ {q_knee2:.4g} Å⁻¹ → Rg2_init ≈ {Rg2_init:.1f} Å")

    # ----- Choose global fit region -----
    q_min_fit = q.min()
    q_max_fit = min(q.max(), fit_q_max)
    mask_fit = (q >= q_min_fit) & (q <= q_max_fit)

    q_fit = q[mask_fit]
    I_fit = I[mask_fit]
    s_fit = sigma[mask_fit]

    # ----- Build initial guess + bounds -----
    # param order: G1, Rg1, B1, P1,   G2, Rg2, B2, P2
    p0 = np.array([G1_init, Rg1_init, B1_init, P1_init,
                   G2_init, Rg2_init, B2_init, P2_init], dtype=float)

    lower_bounds = np.array([0.0, 30.0, 0.0, 2.0,
                             0.0,  5.0, 0.0, 2.0])
    upper_bounds = np.array([np.inf, 500.0, np.inf, 6.0,
                             np.inf, 200.0, np.inf, 6.0])

    p0 = np.clip(p0, lower_bounds, upper_bounds)

    try:
        popt, pcov = curve_fit(
            beaucage_twolevel,
            q_fit,
            I_fit,
            sigma=s_fit,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=40000
        )
        G1, Rg1, B1, P1, G2, Rg2, B2, P2 = popt
    except Exception as e:
        print("Two-level Beaucage fit failed:", e)
        return (np.nan,) * 8, mask_fit

    print("\nTwo-level Beaucage fit parameters:")
    print(f"  Level-1: G1={G1:.3g}, Rg1={Rg1:.2f} Å, B1={B1:.3g}, P1={P1:.2f}")
    print(f"  Level-2: G2={G2:.3g}, Rg2={Rg2:.2f} Å, B2={B2:.3g}, P2={P2:.2f}")
    print(f"  Fit q-range: {q_min_fit:.4g} – {q_max_fit:.4g} Å⁻¹")

    return (G1, Rg1, B1, P1, G2, Rg2, B2, P2), mask_fit


# ---------------- Porod invariant for two-level model -------------
def compute_porod_invariant_twolevel(params, q_data_max):
    """
    Q_A = ∫ q^2 I(q) dq, with q in Å^-1, I in cm^-1.
    Units: Q_A [cm^-1·Å^-3]
    """
    if any(np.isnan(p) for p in params):
        return np.nan

    G1, Rg1, B1, P1, G2, Rg2, B2, P2 = params

    q_upper = max(10.0 * q_data_max, q_data_max + 0.5)

    def integrand(q):
        return q**2 * beaucage_twolevel(q, G1, Rg1, B1, P1, G2, Rg2, B2, P2)

    Q_A, err = quad(integrand, 0.0, q_upper, limit=700)
    return Q_A


def solve_porosity_from_Q(Q_A, delta_rho):
    """
    Two-phase system:
        Q_cm = 2*pi^2 * (Δρ)^2 * φ * (1 - φ)

    Here Q_A = ∫ q^2 I(q) dq with q in Å^-1, I in cm^-1.
    Convert:
        Q_cm = 1e24 * Q_A  (Å^-3 → cm^-3)
    """
    if np.isnan(Q_A) or delta_rho <= 0:
        return np.nan, np.nan

    Q_cm = Q_A * 1.0e24

    A = 2.0 * np.pi**2 * (delta_rho**2)
    x = Q_cm / A
    D = 1.0 - 4.0 * x
    if D < 0:
        return np.nan, Q_cm

    phi1 = (1.0 - np.sqrt(D)) / 2.0
    phi2 = (1.0 + np.sqrt(D)) / 2.0

    if 0.0 <= phi1 <= 1.0:
        return phi1, Q_cm
    elif 0.0 <= phi2 <= 1.0:
        return phi2, Q_cm
    else:
        return np.nan, Q_cm


# ========================= Main =========================
def main():
    q, I, sigma = read_three_column(infile)
    I     = I * intensity_scale
    sigma = sigma * intensity_scale

    print(f"Loaded {len(q)} points from {infile}")
    print(f"q range: {q.min():.4g} – {q.max():.4g} Å⁻¹")

    # ---- Two-level Beaucage fit ----
    params, mask_fit = fit_beaucage_twolevel(q, I, sigma)

    if any(np.isnan(p) for p in params):
        print("Fit failed; aborting invariant calculation.")
        return

    G1, Rg1, B1, P1, G2, Rg2, B2, P2 = params

    # ---- Porod invariant & porosity ----
    Q_A = compute_porod_invariant_twolevel(params, q.max())
    if not np.isnan(Q_A):
        phi, Q_cm = solve_porosity_from_Q(Q_A, delta_rho)
        print(f"\nPorod invariant:")
        print(f"  Q_A  (Å-units) ≈ {Q_A:.4g}  [cm^-1·Å^-3]")
        print(f"  Q_cm          ≈ {Q_cm:.4g}  [cm^-4]")

        if not np.isnan(phi):
            V_pore_mass = phi / rho_solid
            print(f"  Porosity φ          ≈ {phi:.3f}")
            print(f"  V_pore per mass     ≈ {V_pore_mass:.4f} cm³/g")
        else:
            V_pore_mass = np.nan
            print("  Could not derive φ from Q (check Δρ, units).")
    else:
        phi = np.nan
        V_pore_mass = np.nan
        Q_cm = np.nan
        print("\nQ could not be computed from the fit.")

    # ---- Save summary CSV ----
    summary = {
        "G1": G1, "Rg1_A": Rg1, "B1": B1, "P1": P1,
        "G2": G2, "Rg2_A": Rg2, "B2": B2, "P2": P2,
        "Q_A_cm^-1_A^-3": Q_A,
        "Q_cm_cm^-4": Q_cm,
        "delta_rho_cm^-2": delta_rho,
        "porosity_phi": phi,
        "rho_solid_g_cm3": rho_solid,
        "V_pore_per_mass_cm3_g": V_pore_mass,
        "q_min_1_per_A": q.min(),
        "q_max_1_per_A": q.max(),
        "pow_q_min_1_per_A": pow_q_min,
        "pow_q_max_1_per_A": pow_q_max,
        "fit_q_max_1_per_A": fit_q_max,
    }
    df = pd.DataFrame([summary])
    csv_path = out_prefix_path + "_summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("\nSaved summary CSV to:", csv_path)

    # ---- Plot: data + fit ----
    q_fine = np.logspace(np.log10(q.min()*0.9), np.log10(q.max()*1.1), 800)
    I_fit_curve = beaucage_twolevel(q_fine, *params)

    plt.figure(figsize=(8, 6))
    plt.loglog(q, I, ".", ms=3, alpha=0.4, label="data")
    plt.loglog(q_fine, I_fit_curve, "-", lw=2, label="Beaucage 2-level fit")

    # highlight fit region
    q_fit_region = q[mask_fit]
    if q_fit_region.size > 0:
        ymin = I[mask_fit].min() * 0.6
        ymax = I[mask_fit].max() * 1.4
        plt.fill_between(
            q_fit_region,
            ymin,
            ymax,
            color="orange",
            alpha=0.08,
            label="fit region"
        )

    plt.xlabel(r"$q\ (\mathrm{\AA^{-1}})$")
    plt.ylabel(r"$I(q)$ (abs. units × thickness)")
    plt.title(out_prefix + " (Beaucage 2-level unified fit)")
    plt.legend()
    plt.tight_layout()

    fig_path = out_prefix_path + "_fit.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("Saved fit figure to:", fig_path)


if __name__ == "__main__":
    main()
