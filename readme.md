# Fourier Transform in the Complex Plane

**Author:** Shreyash Singh  
**Level:** Advanced Undergraduate / Early Research  
**Domains:** Complex Analysis · Fourier Analysis · PDEs · Numerical Methods · Scientific Visualization

---

## 📌 Project Overview

This project explores the **analytic continuation of the Fourier transform into the complex plane** and provides numerical visualizations of its **magnitude, phase, and spectral deformation**. Unlike the classical Fourier transform defined only for real frequencies, this project extends the frequency variable to complex values:

\[
F(z) = \int_{-\infty}^{\infty} f(x) e^{-izx} \, dx, \quad z \in \mathbb{C}
\]

The project demonstrates:
- Holomorphicity of the Fourier transform
- Exponential decay and instability along the imaginary axis
- Spectral interpretation using PDE eigenfunctions
- High-quality visualizations and animations

This work bridges **complex analysis → Fourier methods → PDE spectral theory → scientific computing**.

---

## 🧠 Key Concepts Covered

- Complex Fourier Transform
- Analytic continuation
- Cauchy–Riemann equations (numerical verification)
- Paley–Wiener-type decay
- Stability wedges in the complex plane
- Heat and Schrödinger equation spectral theory
- Resolvent operator interpretation
- Numerical quadrature and convergence
- Animated spectral slices

---

## 📁 Repository Structure

```
.
├── notebook/
│   └── complex_fourier_transform.ipynb
│
├── report/
│   ├── Complex_Fourier_Transform_Report.tex
│   └── Complex_Fourier_Code_Overview.tex
│
├── figures/
│   ├── magnitude_heatmap.png
│   ├── phase_heatmap.png
│   └── surface_plot.png
│
├── animations/
│   ├── imaginary_sweep.mp4
│   └── real_sweep.mp4
│
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation & Requirements

Create a virtual environment (recommended) and install dependencies:

```bash
pip install -r requirements.txt
```

### Required Python Packages
- `numpy`
- `matplotlib`
- `jupyter`

---

## ▶️ How to Run the Project

1. Launch Jupyter Notebook:
```bash
jupyter notebook
```

2. Open:
```
notebook/complex_fourier_transform.ipynb
```

3. Run cells sequentially to:
- Define test functions
- Construct the complex frequency grid
- Compute the complex Fourier transform
- Generate heatmaps
- Produce animated spectral sweeps

---

## 📊 Visual Outputs

The notebook generates the following main results:

- ✅ Heatmap of **|F(z)|** over the complex plane
- ✅ Heatmap of **arg(F(z))** (phase portrait)
- ✅ 3D surface plot of **|F(z)|**
- ✅ Animated spectral sweeps along:
  - Constant **Im(z)** (stability visualization)
  - Constant **Re(z)** (frequency deformation)

These are saved in the `figures/` and `animations/` directories.

---

## 🧾 Report & Documentation

Two LaTeX documents are included:

- **Main Research Report:**
  - `Complex_Fourier_Transform_Report.tex`
  - Full theory, proofs, PDE links, and visual interpretation

- **Code Overview:**
  - `Complex_Fourier_Code_Overview.tex`
  - Clean explanation of the numerical implementation

To compile:
```bash
pdflatex Complex_Fourier_Transform_Report.tex
pdflatex Complex_Fourier_Code_Overview.tex
```

---

## 🔬 Mathematical Highlights

- Proof of **entire analyticity** for Schwartz-class functions
- Explicit closed-form solution for the Gaussian case:
\[
F(z) = \sqrt{\pi} e^{-z^2/4}
\]
- Numerical verification of the **Cauchy–Riemann equations**
- Visualization of **spectral stability wedges**
- Direct interpretation via:
  - Heat equation
  - Schrödinger equation
  - Resolvent operators

---

## 🎯 Applications

- Quantum tunneling and momentum-space wavefunctions
- Signal stability and damping
- Diffusion processes
- Non-Hermitian spectral theory
- Control systems and resonances
- Research-grade scientific visualization

---

## 🚀 Future Extensions

Planned or possible upgrades include:

- Paley–Wiener bounds with numerical verification
- Non-analytic test functions and singularity tracking
- PDE time-evolution animations
- Operator resolvent pole tracking
- GPU acceleration with CUDA
- Export to journal-ready figures

---

## 👤 Author

**Shreyash Singh**  
B.Tech Mathematics and Computing  
Research Interest: Applied Analysis, PDEs, Scientific Computing, Quantum Models

---

## 📜 License

This project is intended for academic and research use. You are free to fork, modify, and cite with attribution.

---

If you use or build upon this work, a citation or acknowledgment would be appreciated.

