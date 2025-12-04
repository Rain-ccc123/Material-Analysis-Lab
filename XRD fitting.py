"""

XRD analysis for (002) and (100) peaks of carbon materials
---------------------------------------------------------
Features:
 - Input: two-column (2theta, intensity) text/csv file
 - Smooth and pre-process data
 - Fit (002) and (100) peaks individually (Pseudo-Voigt + linear background)
 - Output (all distances in nm):
        d002_nm,  Lc_nm,  N_layers ≈ Lc/d002 (from 002 peak)
        d100_nm,  La_nm,  a_nm      (from 100 peak, a = 2/sqrt(3) * d100)
 - Save result as ASCII-only CSV (UTF-8-SIG, Excel-safe)
 - Save peak fitting figure
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from lmfit.models import PseudoVoigtModel, LinearModel
from scipy import interpolate
import os

# ================= User Parameters =================
infile = r"path.txt"                     # input XRD file
wavelength = 1.5406                      # Cu Kα (Angstrom)
inst_fwhm = 0.01                         # instrumental broadening (deg)
Kc = 0.9                                 # Scherrer K for Lc (002)
Ka = 1.84                                # Scherrer K for La (100)
out_prefix = "xrd_result"
out_dir = r"path_ouput"                  # output folder

# theta ranges for 002 and 100 peaks (deg)
peak_ranges = {
    "002": (10, 35),
    "100": (35, 55),
}
# ====================================================

os.makedirs(out_dir, exist_ok=True)
out_prefix_path = os.path.join(out_dir, out_prefix)

# ---------------------- I/O ------------------------
def read_two_column(path):
    """Read two-column numeric data: 2theta, intensity."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    x, y = float(parts[0]), float(parts[1])
                    data.append((x, y))
                except Exception:
                    continue
    if not data:
        raise ValueError("No valid numeric data found.")
    arr = np.array(data)
    idx = np.argsort(arr[:, 0])
    return arr[idx, 0], arr[idx, 1]

# ------------------ FWHM extraction ----------------
def numeric_fwhm(x, y):
    """Numerical FWHM from fitted peak."""
    if np.max(y) <= 0:
        return np.nan
    peak_idx = np.argmax(y)
    peak_val = y[peak_idx]
    half = peak_val / 2.0

    above = np.where(y >= half)[0]
    if len(above) < 2:
        return np.nan

    left_idx, right_idx = above[0], above[-1]

    left_idx = max(left_idx, 1)
    right_idx = min(right_idx, len(y) - 2)

    fl = interpolate.interp1d(
        y[:left_idx+1], x[:left_idx+1],
        bounds_error=False, fill_value="extrapolate"
    )
    fr = interpolate.interp1d(
        y[right_idx-1:], x[right_idx-1:],
        bounds_error=False, fill_value="extrapolate"
    )

    return float(fr(half) - fl(half))

# --------------------- Peak Fitting ----------------
def fit_peak_region(x_region, y_region, label):
    """Fit single peak (Pseudo-Voigt + linear background) in a given 2theta range."""
    if x_region.size < 8:
        return None

    # robust initial guess: local maximum
    local_max_idx = np.argmax(y_region)
    center_guess = x_region[local_max_idx]

    # fitting window ~30% of region width, but not too narrow
    window = max((x_region.max() - x_region.min()) * 0.3, 0.8)

    model = PseudoVoigtModel(prefix="p_") + LinearModel(prefix="bkg_")
    pars = model.make_params()

    pars["p_center"].set(value=center_guess,
                         min=center_guess - window,
                         max=center_guess + window)
    pars["p_sigma"].set(value=0.08, min=1e-4)
    pars["p_amplitude"].set(value=y_region.max() - y_region.min(), min=0)
    pars["p_fraction"].set(value=0.5, min=0, max=1)

    pars["bkg_intercept"].set(value=y_region.min())
    pars["bkg_slope"].set(value=0)

    try:
        out = model.fit(y_region, pars, x=x_region)
    except Exception:
        return None

    xfine = np.linspace(x_region.min(), x_region.max(), 600)
    yfit = out.eval(x=xfine)
    bkg = out.params["bkg_intercept"].value + out.params["bkg_slope"].value * xfine
    ypeak = yfit - bkg

    fwhm_deg = numeric_fwhm(xfine, ypeak)

    return {
        "label": label,
        "center_2theta": float(out.params["p_center"].value),
        "fwhm_deg": float(fwhm_deg),
        "x_fine": xfine,
        "y_fit": yfit,
    }

# --------------------- Scherrer --------------------
def scherrer_size_nm(lambda_A, fwhm_deg, two_theta_deg, K):
    """Scherrer grain size D in nm + corrected beta (rad)."""
    if np.isnan(fwhm_deg) or fwhm_deg <= 0:
        return np.nan, np.nan

    beta_rad = np.deg2rad(fwhm_deg)
    inst_rad = np.deg2rad(inst_fwhm)
    beta_corr = np.sqrt(max(0.0, beta_rad**2 - inst_rad**2))

    if beta_corr <= 0:
        return np.nan, beta_corr

    theta_rad = np.deg2rad(two_theta_deg / 2.0)

    # D in Angstrom, then convert to nm
    D_A = (K * lambda_A) / (beta_corr * np.cos(theta_rad))
    D_nm = D_A * 0.1
    return D_nm, beta_corr

def bragg_d_nm(lambda_A, two_theta_deg):
    """Bragg d-spacing in nm."""
    theta_rad = np.deg2rad(two_theta_deg / 2.0)
    s = np.sin(theta_rad)
    if s == 0:
        return np.nan
    d_A = lambda_A / (2 * s)
    return d_A * 0.1   # Å → nm

# ====================== Main ======================
def main():
    x, y = read_two_column(infile)

    # smoothing (ensure odd window length)
    win = max(5, (len(y) // 200) * 2 + 1)
    if win >= len(y):
        win = len(y) if len(y) % 2 == 1 else len(y) - 1
    y_sm = savgol_filter(y, win, 3) if len(y) >= win and win >= 5 else y

    fits = []
    for label, (lo, hi) in peak_ranges.items():
        mask = (x >= lo) & (x <= hi)
        if not np.any(mask):
            print(f"Warning: no data in range for peak {label} ({lo}-{hi} deg).")
            continue
        fit = fit_peak_region(x[mask], y_sm[mask], label)
        if fit is not None:
            fits.append(fit)
        else:
            print(f"Warning: fit failed for peak {label}.")

    # Prepare results (all distances in nm)
    results = []

    d002_nm = Lc_nm = d100_nm = La_nm = a_nm = np.nan
    N_layers = np.nan  # stacked graphite layers N ≈ Lc/d002

    for f in fits:
        label = f["label"]
        tt = f["center_2theta"]
        fwhm_deg = f["fwhm_deg"]

        d_nm = bragg_d_nm(wavelength, tt)

        if label == "002":
            d002_nm = d_nm
            Lc_nm, beta002 = scherrer_size_nm(wavelength, fwhm_deg, tt, Kc)
            # N ≈ Lc / d002
            if (not np.isnan(Lc_nm)) and (not np.isnan(d002_nm)) and d002_nm > 0:
                N_layers = Lc_nm / d002_nm
            else:
                N_layers = np.nan

            results.append({
                "Peak": "002",
                "2theta_deg": tt,
                "d002_nm": d002_nm,
                "d100_nm": np.nan,
                "FWHM_deg": fwhm_deg,
                "FWHM_corr_rad": beta002,
                "Lc_nm": Lc_nm,
                "La_nm": np.nan,
                "a_nm": np.nan,
                "N_layers": N_layers,
            })

        elif label == "100":
            d100_nm = d_nm
            La_nm, beta100 = scherrer_size_nm(wavelength, fwhm_deg, tt, Ka)
            a_nm = (2.0 / np.sqrt(3.0)) * d100_nm if not np.isnan(d100_nm) else np.nan
            results.append({
                "Peak": "100",
                "2theta_deg": tt,
                "d002_nm": np.nan,
                "d100_nm": d100_nm,
                "FWHM_deg": fwhm_deg,
                "FWHM_corr_rad": beta100,
                "Lc_nm": np.nan,
                "La_nm": La_nm,
                "a_nm": a_nm,
                "N_layers": np.nan,   # only defined for 002
            })

    # -------- CSV output (ASCII-only, UTF-8-SIG) --------
    if results:
        df = pd.DataFrame(results)
        csv_path = out_prefix_path + "_002_100_params_nm.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print("\nSaved CSV to:", csv_path)
        print(df.to_string(index=False))
    else:
        print("No peaks fitted. No CSV generated.")
        return

    # -------- Summary print --------
    print("\nSummary (all in nm):")
    print(f"d002_nm: {d002_nm:.4f}" if not np.isnan(d002_nm) else "d002_nm: n/a")
    print(f"Lc_nm:  {Lc_nm:.4f}"  if not np.isnan(Lc_nm)  else "Lc_nm:  n/a")
    print(f"d100_nm:{d100_nm:.4f}" if not np.isnan(d100_nm) else "d100_nm: n/a")
    print(f"La_nm:  {La_nm:.4f}"  if not np.isnan(La_nm)  else "La_nm:  n/a")
    print(f"a_nm:   {a_nm:.4f}"   if not np.isnan(a_nm)   else "a_nm:   n/a")
    print(f"N_layers ≈ Lc/d002: {N_layers:.2f}" if not np.isnan(N_layers) else "N_layers ≈ Lc/d002: n/a")

    # -------- Plot --------
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label="raw", lw=1)
    plt.plot(x, y_sm, label="smoothed", lw=1, alpha=0.6)

    for f in fits:
        plt.plot(f["x_fine"], f["y_fit"], "--", label=f"{f['label']} fit", lw=1)

    plt.xlabel("2θ (deg)")
    plt.ylabel("Intensity (a.u.)")
    plt.legend()
    plt.tight_layout()

    png_path = out_prefix_path + "_002_100_fit.png"
    plt.savefig(png_path, dpi=300)
    plt.close()

    print("Saved figure to:", png_path)


if __name__ == "__main__":
    main()
