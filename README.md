
# Material Analysis Lab: Online Platform for Quantitative TEM, XRD, and SAXS Analysis of Carbon Materials

**Website:** [http://106.53.68.158:5000/](http://106.53.68.158:5000/)

This online platform, referred to as **Material Analysis Lab**, provides an integrated environment for **quantitative microstructural analysis of carbon materials**, with a particular focus on **hard carbon negative electrodes for sodium-ion batteries**. The tool combines **TEM image analysis**, **XRD peak fitting**, and **SAXS unified modeling** into a single web interface to facilitate reproducible data processing and structure–performance correlation studies.

The platform is intended to support both routine data analysis and the construction of a **materials genome–style database** for hard carbons, enabling AI-assisted screening and design of high-performance negative electrodes.

---

## 1. Overview of Available Modules

The web interface currently consists of three main modules:

1. **TEM Microstructure Analysis**
2. **XRD Peak Fitting (002 / 100)**
3. **SAXS Unified Beaucage Fitting**

Each module takes standard experimental data as input and returns both **numerical descriptors (CSV)** and **graphical outputs (PNG)** suitable for direct use in publications and further data mining.

---

## 2. TEM Microstructure Analysis

### 2.1 Purpose

This module provides a **quantitative, TEM-imaging–based analysis** of hard carbon microstructure. It extracts an **orderliness descriptor (R-value, specifically R₉₀)** from high-resolution TEM (HRTEM) images, which correlates with the sodium-storage behavior of hard carbons.

### 2.2 Input Requirements

* **Image type:** High-resolution TEM (HRTEM) image of a carbon material
* **Format:** Common image formats (e.g., PNG, JPG, TIFF)
* **Content requirements:**

  * A clear **scale bar** must be present in the image.
  * The image should include a representative **microstructural region** (graphene-like domains, pores, defects, etc.).
* **Calibration:**

  * The user is asked to draw a box along the scale bar so that the box length **exactly matches the scale bar**.
  * The system uses this calibration to convert pixels into real-space units (nm).

### 2.3 Typical Workflow

1. **Upload the original TEM image** via the web interface.
2. **Draw a box on the scale bar**, matching its length to the scale bar.

   * The system uses this step to calibrate the pixel-to-nanometer scale.
3. **Move the analysis box (ROI)** to a characteristic microstructural region.

   * The ROI should capture representative curved graphene-like domains and pore structures.
4. **Run the analysis**:

   * The platform computes a local coherence/“orderliness” map.
   * It extracts the distribution of the R-parameter over the selected region.

### 2.4 Output

The TEM module returns:

* **Heatmap of microstructural orderliness** within the ROI.
* **Histogram of the R-parameter distribution.**
* **Key statistical descriptors (exported as CSV):**

  * Mean R
  * Standard deviation of R
  * **R₉₀ (90th percentile of R)** — used as the main orderliness descriptor in the manuscript.

These outputs can be directly used to correlate TEM-derived microstructural orderliness with electrochemical performance (e.g., plateau capacity, rate capability).

---

## 3. XRD Peak Fitting Module

### 3.1 Purpose

This module performs **automated peak fitting of the (002) and (100) reflections** of carbon materials to obtain:

* Interplanar spacings **d₀₀₂** and **d₁₀₀**
* Orientation-dependent crystallite sizes **L_c** and **L_a**
* Lattice parameter **a** (from d₁₀₀)

It is optimized for hard carbon and related carbon materials used as negative electrodes.

### 3.2 Input Requirements

* **Data format:** Two-column text or CSV file

  * Column 1: 2θ (degrees)
  * Column 2: Intensity (arbitrary units)
* **File type:** `.txt` or `.csv` with comma or whitespace delimiters
* **Units:**

  * 2θ in degrees
  * X-ray wavelength (typically Cu Kα, λ = 0.15406 nm) assumed or user-specified

### 3.3 Fitting Procedure

* The module fits the (002) and (100) peaks using:

  * **Pseudo-Voigt peak shape**
  * **Linear background**
* Instrumental broadening and Scherrer analysis are used to extract:

  * d₀₀₂, d₁₀₀ (Å or nm)
  * L_c, L_a (nm)
  * a (nm), derived from d₁₀₀
* The fitting algorithm is implemented in Python using standard scientific libraries (e.g., `numpy`, `lmfit`, `matplotlib`).

### 3.4 Output

* **Fitted diffraction profiles** as PNG images.
* **Numerical results** (CSV):

  * d₀₀₂, d₁₀₀
  * Lc, La
  * a
  * FWHM values and other intermediate fitting parameters, if applicable.

These parameters are used in the manuscript to characterize **interlayer spacing**, **crystallite size**, and **graphitic domain evolution** with heat treatment.

---

## 4. SAXS Unified Beaucage Fitting

### 4.1 Purpose

This module performs **two-level Beaucage unified fitting** of SAXS data to quantify:

* Large-scale aggregate structure (level 1)
* Nanopore/mesostructure contributions (level 2)

It provides characteristic sizes (Guinier radii), fractal/Porod exponents, and the **Porod invariant**, which is further used to estimate **porosity and specific pore volume**.

### 4.2 Input Requirements

* **Data format:** Two-column text or CSV file

  * Column 1: q (Å⁻¹)
  * Column 2: I(q) (arbitrary units or absolute units, depending on calibration)
* **q-range:**

  * The code allows definition of a maximum q-value for global fitting.
  * If the upper limit is set too high, WAXS contributions may be mixed in; users can adjust this parameter as needed.

### 4.3 Fitting Procedure

The two-level Beaucage model returns, for each dataset:

* **Level 1 (large-scale aggregates / particle size):**

  * G₁, R_g1, B₁, P₁
  * R_g1 corresponds to large-scale structures (typically larger than nanopore R_g2).

* **Level 2 (nanopores / density fluctuations):**

  * G₂, R_g2, B₂, P₂
  * R_g2 typically reflects length scales on the order of ~1–3 nm for hard carbon nanopores.
  * P₂ (Porod/fractal exponent) is interpreted in terms of mass-fractal or surface-fractal behavior.

* **Porod invariant and porosity:**

  * Porod invariant Q (cm⁻⁴)
  * Porosity φ and specific pore volume V_pore (cm³ g⁻¹), computed using:

    * Scattering length density contrast Δρ (e.g., Δρ = 1.69 × 10¹¹ cm⁻² for carbon vs. vacuum)
    * Solid density ρ_solid (g cm⁻³), used to convert φ into V_pore.

### 4.4 Output

* **Fitted SAXS curves** with decomposed contributions (PNG).
* **Parameter summary** (CSV), including:

  * G₁, R_g1, B₁, P₁
  * G₂, R_g2, B₂, P₂
  * Q, φ, V_pore

These parameters are used to interpret nanopore size, connectivity, and fractal characteristics of hard carbons and related materials.

---

## 5. Implementation

* **Backend:** Python (Flask or similar web framework)
* **Core libraries:** `numpy`, `scipy`, `lmfit`, `matplotlib` and other standard scientific packages
* **Front-end:** Simple web interface for data upload, scale calibration, ROI selection, and result visualization
* **Outputs:**

  * All numerical results exported as **CSV** files.
  * All plots exported as **PNG** images suitable for inclusion in figures and supplementary information.

---

## 6. Reproducibility and Data Use

All code used for quantitative data analysis has been made available via:

* The **Material Analysis Lab** online platform ([http://106.53.68.158:5000/](http://106.53.68.158:5000/))

The platform is primarily optimized for **hard carbon and related carbon materials** but can, in principle, be applied to other porous carbons with similar data formats.

---

## 7. Citation

If you use this platform or its underlying code in your research, please cite the following work:

> Zhiyu Zou *et al.*, “Quantitative TEM Analysis of Hard-Carbon Microstructure for Correlating Structure with Sodium‑Storage Performance,” *submitted to* Science Advances, 2025.


