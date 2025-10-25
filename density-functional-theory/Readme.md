# Radial Kohn-Sham Equation Solver for Hydrogen-like Atoms

**Author:** Wang Jianghai @Nanyang Technological University

**Date:** 2025-10-24

---

## Table of Contents

- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
  - [1.1 Preliminary](#11-preliminary)
  - [1.1.1 Density Functional Theory](#111-density-functional-theory)
  - [1.1.2 Kohn-Sham Equations](#112-kohn-sham-equations)
  - [1.1.3 Radial Kohn-Sham Equation](#113-radial-kohn-sham-equation)
- [2. Numerical Implementation](#2-numerical-implementation)
  - [2.1 Discretization](#21-discretization)
  - [2.2 Numerov Method](#22-numerov-method)
  - [2.3 Thomas Algorithm](#23-thomas-algorithm)
  - [2.4 Self-Consistent Field Method](#24-self-consistent-field-method)
  - [2.5 Exchange-Correlation Functional](#25-exchange-correlation-functional)
  - [2.6 Implementation Details](#26-implementation-details)
- [3. Usage](#3-usage)
- [4. Validation](#4-validation)
- [5. Results and Discussion](#5-results-and-discussion)
- [6. Mathematical Appendix](#6-mathematical-appendix)
- [7. References](#7-references)

<div id="abstract" style="border:1px solid #d0d7de; padding:16px; background:#e6ffef; border-radius:6px;">
<strong style="font-size:1.05em;">Abstract</strong>

<p style="margin-top:0.5em;">
This project implements a radial Kohn–Sham (KS) self-consistent field (SCF) solver for spherically symmetric (H-like) systems, focusing on the 1s orbital. The solver uses a fixed radial mesh, constructs the Kohn–Sham potential as
</p>

$$
V_{\text{KS}}(r) = V_{\text{nuc}}(r) + V_{\text{ee}}(r) + V_{\text{xc}}(r),
$$

<p>
solves the radial KS differential equation with Numerov/Thomas tridiagonal propagation, computes energy components, and iterates until the KS eigenvalue ($\varepsilon$) converges. Exchange-correlation is handled in a local density approximation (LDA) with an analytic parameterization. The implementation is pedagogical and suitable for learning numerical Density Functional Theory (DFT) methods and numerical ordinary differential equation (ODE) solvers (Numerov and Thomas algorithms) in spherical coordinates.
</p>
</div>

# 1. Introduction: Density Functional Theory
Following the establishment of quantum mechanics and statistical physics, theoretical studies of condensed matter gradually emerged. It became clear that the macroscopic properties of solids are intrinsically linked to their electronic behavior. In principle, if one could exactly solve the electronic wave functions in a solid, all physical quantities of the system could be obtained through the corresponding quantum-mechanical operators. However, a typical solid contains on the order of $10^{23}$ particles, each with three spatial degrees of freedom, making the many-body Hamiltonian intractably complex and analytically unsolvable. Hence, appropriate approximations are essential.

Density Functional Theory is a quantum mechanical modeling method used in physics, chemistry, and materials science to investigate the electronic structure of many-body systems. The fundamental principle of DFT is that the properties of a many-electron system can be determined using functionals of the spatially-dependent electron density rather than the many-body wavefunction, reducing the problem from a $3N$-dimensional space to only three dimensions. This fundamental reformulation significantly simplifies the calculation of electronic structure while retaining quantum-mechanical accuracy.

## 1.1 Adiabatic and Single-Electron Approximations
Because the mass of a nucleus is roughly 1800 times that of an electron, electrons respond much more rapidly to environmental changes than nuclei do. Thus, the physical problem can be decoupled into two parts: when describing electronic motion, the nuclei can be treated as fixed sources of potential, whereas when describing nuclear motion, the electronic distribution can be considered static. Solving the electronic problem under a fixed nuclear potential yields the lowest-energy configuration, i.e., the electronic ground state.

This separation of electronic and nuclear motion forms the basis of the Born–Oppenheimer approximation, or the adiabatic approximation. It reduces the full many-body problem to a purely electronic one, with the ground-state energy $E(R_1, R_2, ..., R_M)$ depending on the nuclear positions {$R_i$}, thereby defining the adiabatic potential energy surface of the system.

Even after separating nuclear and electronic motion, the electron–electron interaction remains a many-body problem involving $N$ particles. To address this, Hartree proposed averaging the instantaneous interactions between electrons, treating each electron as moving independently in an averaged potential—the Hartree approximation, or single-electron approximation. Although this simplification transformed the many-electron problem into a set of single-electron equations, it neglected electron exchange and correlation effects. Fock later incorporated the exchange interaction, leading to the Hartree–Fock approximation. While Hartree–Fock theory accurately accounts for exchange, it still neglects correlation effects, and its accuracy deteriorates with increasing electron number.

## 1.2 Thomas–Fermi–Dirac Approximation
In 1927, Thomas and Fermi proposed the homogeneous electron gas model, which neglects electron–electron interactions and expresses the total energy as a functional of electron density. In 1930, Dirac introduced a local approximation for exchange interactions, extending the model to account for exchange energy within the density functional framework. The combined formulation is known as the Thomas–Fermi–Dirac (TFD) approximation.

## 1.3 Hohenberg–Kohn Theorems
The theoretical foundation of modern DFT lies in two theorems proved by Hohenberg and Kohn.

1. The ground-state properties of a many-electron system are uniquely determined by the electron density $n(\mathbf{r})$.
2. The electron density that minimizes the energy of the overall functional is the ground-state electron density.

The energy functional can be expressed as:
$$
E[\psi_i] = -\frac{\hbar^2}{m}\int \psi_i^* \nabla^2 \psi_i d^3r + \int V(\mathbf{r}) n(\mathbf{r}) d^3r + \frac{e^2}{2}\iint \frac{n(\mathbf{r}) n(\mathbf{r'})}{|\mathbf{r}-\mathbf{r'}|} d^3r d^3r' + E_{\text{ion}} + E_{\text{XC}}[\psi_i]
$$

which contains the kinetic energy of electrons, electron–nucleus Coulomb interaction, electron–electron Coulomb repulsion, and nucleus–nucleus repulsion. The remaining term, $E_{\text{XC}}[\psi_i]$, is the exchange-correlation energy functional, which accounts for the complex many-body effects of exchange and correlation among electrons.
## 1.4 Kohn-Sham Equations

In practice, DFT calculations are performed using the Kohn-Sham approach, which replaces the original many-body problem by an auxiliary independent-particle problem. The Kohn-Sham equation in atomic units is:

$$\left[-\frac{1}{2}\frac{d^2}{dr^2} + \frac{l(l+1)}{2r^2}+V_\text{KS}(r)\right]P_{nl}(r) = \varepsilon_{nl}P_{nl}(r)$$

where $P_{nl}(r)=r\Psi_{nl}(r)$ is the radial wave function, $V_{\text{KS}}(\mathbf{r})$ is the Kohn-Sham potential, given by:

$$V_{\text{KS}}(\mathbf{r}) = V_\text{nuc}(r) + V_{ee}(r) + V_{xc}(r)$$

comprising the nuclear attraction potential

$$V_\text{nuc}(r)=-\frac{Z}{r}$$

the electron-electron Coulomb repulsion potential

$${V^{ee}}\left( r \right) = 4\pi \int {{{\rho \left( {r'} \right)} \over {\left| {r - r'} \right|}}} {r'^2}dr'$$

and the exchange-correlation potential $V_{\text{xc}}$.

### Radial Kohn-Sham Equation

For atoms with spherical symmetry, we can separate the wavefunction into radial and angular parts:

$$\psi_{nlm}(\mathbf{r}) = \frac{P_{nl}(r)}{r} Y_{lm}(\theta, \phi)$$

where $Y_{lm}$ are spherical harmonics and $P_{nl}(r)$ is the radial wavefunction. For the ground state (1s orbital, with $l=0$), the radial Kohn-Sham equation becomes:

$$\left[ -\frac{1}{2}\frac{d^2}{dr^2} + V_{{KS}}(r) \right] P_{10}(r) = \varepsilon_{10}P_{10}(r)$$

This is the equation we solve numerically in this implementation.

## 1.5 Exchange–Correlation Functionals
The key unknown in DFT is the exchange-correlation functional $E_{XC}[\rho]$. While the Hohenberg–Kohn theorems guarantee its existence, its exact form remains unknown. Practical DFT calculations therefore rely on approximations, among which the **Local Density Approximation (LDA)**, **Generalized Gradient Approximation (GGA)**, and **hybrid functionals** are the most widely used.

### 1.5.1 Local Density Approximation (LDA)
LDA assumes that the exchange–correlation energy at each point depends only on the local electron density, as in a uniform electron gas:
$$
V_{XC}(\mathbf{r}) = V_{XC}^{\text{electron gas}}[n(\mathbf{r})]
$$

Although this approximation neglects spatial variations in the density, it provides a tractable and often surprisingly effective method, especially for systems with slowly varying densities. However, it becomes inaccurate for systems with strong density gradients, such as covalently bonded materials.

### 1.5.2 Generalized Gradient Approximation (GGA)
GGA extends LDA by including the gradient of the density, making $E_{XC}$ a functional of both $n(\mathbf{r})$ and $\nabla n(\mathbf{r})$. This inclusion of local inhomogeneity typically improves accuracy over LDA. Common GGA functionals include **Perdew–Wang (PW91)** and **Perdew–Burke–Ernzerhof (PBE)**, both widely adopted in solid-state calculations.

### 1.5.3 Hybrid Functionals
Hybrid functionals combine the orbital-dependent Hartree–Fock exchange with density-based functionals, controlled by a mixing parameter $\lambda$. For instance, when $\lambda=0$, the exchange is purely Hartree–Fock; when $\lambda=1$, it is entirely LDA or GGA. The **Heyd–Scuseria–Ernzerhof (HSE)** hybrid functional, particularly **HSE06**, mixes 25% of short-range Hartree–Fock exchange with 75% of short-range PBE exchange, while using PBE for correlation and long-range exchange:
$$
E_{XC}^{\text{HSE06}} = \frac{1}{4}E_X^{\text{SR,HF}}(\mu) + \frac{3}{4}E_X^{\text{SR,PBE}}(\mu) + E_X^{\text{LR,PBE}}(\mu) + E_C^{\text{PBE}}
$$
Hybrid functionals often provide improved band-gap predictions and better descriptions of strongly correlated systems, though at significantly higher computational cost and with sensitivity to empirical parameters.

# 2. Numerical Implementation

## 2.1 Discretization

The radial equation is discretized using a uniform grid:

$$r_i = r_0 + i \cdot h, \quad i = 0, 1, ..., N-1$$

where $r_0$ is the starting point (close to the origin), $h$ is the step size, and $N$ is the number of grid points.

## 2.2 Numerov Method

For solving the second-order differential equation, we employ the Numerov method, which provides sixth-order accuracy. The method transforms the equation into a three-point recursion relation:

$$(1 + \frac{h^2}{12} f_{i+1}) y_{i+1} - (2 - \frac{5h^2}{6} f_i) y_i + (1 + \frac{h^2}{12} f_{i-1}) y_{i-1} = 0$$

where $f_i = 2(E - V(r_i))$.

## 2.3 Thomas Algorithm

The Thomas algorithm (also known as the tridiagonal matrix algorithm) is used to efficiently solve the tridiagonal system arising from the Numerov method:

1. Forward elimination phase creates an upper bidiagonal system
2. Backward substitution phase solves for the unknowns

This approach is computationally efficient with O(N) complexity.

## 2.4 Self-Consistent Field Method

The SCF procedure follows these steps:
1. Initialize with an analytical solution for a hydrogen-like atom
2. Calculate the electron density from the wavefunction
3. Construct the Kohn-Sham potential
4. Solve the Kohn-Sham equation to obtain a new wavefunction
5. Mix the old and new potentials to improve convergence stability
6. Check for energy convergence
7. Repeat until convergence

Practical and theoretical remarks on SCF variables and initialization:  
- Use the Kohn–Sham eigenvalue ε (the orbital eigenvalue returned by the KS solver) as the spectral quantity to monitor and to drive eigenvalue updates and convergence tests. Do not substitute the full total energy for the eigenvalue inside the iterative eigenproblem. The total electronic energy is a derived functional of the converged density and should be computed once (or only for diagnostics) after SCF convergence. Using E_tot in place of ε in the eigenvalue loop mixes distinct variational objects and destabilizes convergence.  
- Initial potential: when the initial P(r) is the hydrogenic analytic solution (which is the solution of the Schrödinger equation with V_nuc only), set the initial potential V_old = V_nuc(r) = −Z/r. Choosing V_old = V_KS (full potential) while using a hydrogenic P(r) is inconsistent and frequently causes large first-iteration errors that demand tiny mixing parameters. If oscillations occur, reduce mixing parameter α or use more advanced mixing (DIIS/Pulay).

## 2.5 Exchange-Correlation Functional

We implement the Local Density Approximation (LDA) for the exchange-correlation functional:

- The **correlation potential** uses the Perdew-Zunger parametrization of the Ceperley-Alder quantum Monte Carlo results:
$${V^C}[\rho] = \left\{ \begin{array}{l}A\ln {r_s} + \left( {B - \frac{1}{3}A} \right) + \frac{2}{3}C{r_s}\ln {r_s} + \frac{1}{3}\left( {2D - C} \right){r_s},\;\;\;\;{\rm{if}}\;{r_s} < 1;\\\gamma \frac{{\left( {1 + \frac{7}{6}{\beta _1}\sqrt {{r_s}}  + \frac{4}{3}{\beta _2}{r_s}} \right)}}{{{{\left( {1 + {\beta _1}\sqrt {{r_s}}  + {\beta _2}{r_s}} \right)}^2}}},\;\;\;\;\;\;\;{\rm{if}}\;\;\;{r_s} \ge 1;\end{array} \right.$$
  - For high densities ($r_s < 1$):
    $${V^C}[\rho] = A\ln {r_s} + \left( {B - \frac{1}{3}A} \right) + \frac{2}{3}C{r_s}\ln {r_s} + \frac{1}{3}\left( {2D - C} \right){r_s}$$
  - For low densities ($r_s \geq 1$):
    $${V^C}[\rho] = \gamma \frac{{\left( {1 + \frac{7}{6}{\beta _1}\sqrt {{r_s}}  + \frac{4}{3}{\beta _2}{r_s}} \right)}}{{{{\left( {1 + {\beta _1}\sqrt {{r_s}}  + {\beta _2}{r_s}} \right)}^2}}}$$

where ${r_s} = {\left( {\frac{3}{{4\pi \rho }}} \right)^{\frac{1}{3}}}$ is the Wigner-Seitz radius.

## 2.6 Implementation Details

### 2.6.1 `RadialDFT` Class

The `RadialDFT` class in `DFT.py` encapsulates the numerical solution of the radial Kohn-Sham equation. Key methods include:

- `initialize()`: Sets up the initial wavefunction guess using the analytical solution
- `normalize()`: Normalizes the wavefunction using the trapezoidal rule
- `get_v_nuc()`, `get_v_ee()`, `get_v_xc()`: Calculate the potential components
- `get_v_ks()`: Combines the potential components into the Kohn-Sham potential
- `solve_ks()`: Solves the Kohn-Sham equation using the Numerov method
- `ks_energy()`, `e_xc()`, `total_energy()`: Calculate energy components
- `scf_loop()`: Implements the self-consistent field iteration

### 2.6.2 Boundary Conditions

The radial wavefunction satisfies:
- $P(r \rightarrow 0) \propto r^{l+1}$ (for 1s orbital with $l=0$, so $P(0) = 0$)
- $P(r \rightarrow \infty) = 0$

### 2.6.3 Energy Calculation

The total energy is calculated as:

$$E_{tot} = E_{ks} + E_{ee} + E_{xc} - \int n(r)V_{xc}(r)dr$$

where:
- $E_{ks}$ is the Kohn-Sham eigenvalue
- $E_{ee}$ is the electron-electron repulsion energy
- $E_{xc}$ is the exchange-correlation energy
- The last term corrects for double-counting in the Kohn-Sham potential

Precision on spherical integration for energy components:  
All volume integrals reduce to one-dimensional radial integrals multiplied by 4π. For radial-only functions V(r) and ρ(r):
∫ V( r⃗ ) ρ( r⃗ ) d^3r = 4π ∫_0^∞ V(r) ρ(r) r^2 dr.
Therefore the standard energy component formulas used in post‑SCF evaluation are:

- Hartree / electron–electron contribution (double‑counting handled by prefactor):
  E_ee = + 0.5 * 4π ∫_0^∞ ρ(r) V_ee(r) r^2 dr
  (Check and adopt sign conventions consistent with how V_ee was computed; some implementations store V_ee with opposite sign — reconcile and document.)
- Exchange–correlation energy:
  E_xc = 4π ∫_0^∞ ρ(r) e_xc(r) r^2 dr
- Potential contraction (use for checking consistency):
  ∫ ρ V_xc d^3r = 4π ∫_0^∞ ρ(r) V_xc(r) r^2 dr

Omitting the r^2 Jacobian or the angular factor 4π is a frequent implementation mistake that yields incorrect numerical energies (mismatched units and magnitudes). After implementing these integrals, validate numerically with controlled tests (see Validation).

# 3. Usage

Z = 6       # Nuclear charge for hydrogen-like atom
r0 = 1e-5   # Minimum radius (a.u.)
rf = 20.0   # Maximum radius (a.u.)
N = 10001   # Number of grid points

alpha = 0.1  # Mixing parameter for SCF
prec = 1e-5  # Convergence tolerance
max_iter = 500  # Maximum number of SCF iterations

Execute the calculation:

```bash
python main.py
```

## 3.3 Output Files

The code generates several output files:

1. `wavefunction.dat`: Contains the radial wavefunction $P(r)$ at each iteration
2. `potential.dat`: Contains the different potential components
3. `energy.dat`: Contains the energy components at each iteration
4. `wavefunction_comparison.png`: Plot comparing numerical and analytical wavefunctions
5. `potential_comparison.png`: Plot of the different potential components

# 4. Validation

The implementation is validated by comparing the numerical solution with the analytical solution for hydrogen-like atoms. For a hydrogen-like atom with nuclear charge Z, the analytical 1s wavefunction is:

$$P_\text{analytical}(r) = r\Psi_\text{analytical}(r) = \frac{Z^{\frac32}}{\sqrt{\pi}}re^{-Zr}$$

with energy $E = -\frac{Z^2}{2}$ (in atomic units).

Validation checklist (minimal, reproducible tests):
- One‑electron limit: with V_ee = 0 and V_xc = 0 the solver with V_nuc only must reproduce the analytic eigenvalue ε = −Z^2/2 and the analytic radial function P_1s(r) after normalization with Norm = sqrt(4π ∫ P^2 dr). This test confirms correct treatment of P(r), boundary conditions, Numerov integration, and normalization factors.
- Norm test: assert |4π ∫ P^2 dr − 1| < tol after normalization.
- Energy integral consistency: verify dimensional consistency and relative magnitudes of E_ks, E_ee, E_xc and that E_xc computed from the density integral is consistent (within expected differences) with −∫ ρ V_xc d^3r.
- SCF bookkeeping: during SCF use Δε = |ε_new − ε_old| as convergence metric; compute total energy E_tot only after the density has converged.

# 5. Results and Discussion

## 5.1 Convergence Properties

The convergence of the SCF procedure is monitored by tracking the total energy difference between consecutive iterations. The mixing parameter `alpha` controls the convergence stability; smaller values generally provide more stable convergence but may require more iterations.

## 5.2 Energy Components

The final energy is broken down into components:
- Kohn-Sham eigenvalue energy
- Electron-electron repulsion energy
- Exchange-correlation energy (from density integral)
- Exchange-correlation energy (from potential)

## 5.3 Numerical Accuracy

## 5.4 Theoretical Extensions

This implementation can be extended in several ways:

1. **Multi-electron systems**: Implementing multiple Kohn-Sham orbitals and handling orbital occupations
2. **Alternative XC functionals**: Implementing GGA (Generalized Gradient Approximation) or hybrid functionals
3. **Excited states**: Adapting the solver for excited state calculations
4. **Non-spherical systems**: Extending beyond radial symmetry
5. **Relativistic effects**: Including scalar relativistic corrections

# 6. Mathematical Appendix

## 6.1 Radial Kohn-Sham Equation

Starting from the three-dimensional Schrödinger equation in spherical coordinates:

$$\Psi \left( {r,\theta ,\varphi } \right) = \Psi \left( r \right)Y\left( {\theta ,\varphi } \right)$$

where $Y( {\theta ,\varphi })$ is spherical harmonics. It can be separated on two independent equations for $r$ and $\theta ,\varphi$. The Schrodinger equation in spherical coordinates for part depending on $r$ only, the **radial Schrodinger equation** is

$$\left[-\frac{1}{2r}\frac{d^2}{dr^2}r + \frac{l(l+1)}{2r^2} + V_\text{nuc}(r)\right]\Psi_{nl}(r) = E_{nl}\Psi_{nl}(r)$$

where $V^{NUC}(r)=-\frac{Z}{r}$. For the ground state (1s orbital) with $n=1$, $l=0$, the centrifugal term vanishes, resulting in:

$$\left[-\frac{1}{2}\frac{d^2}{dr^2} + V_\text{nuc}(r)\right]\Psi_{1s}(r) = E\Psi_{1s}(r)$$

## 6.2 Numerov Method

The Numerov method can be used to solve differential equations of the kind

$${{{\partial^2}} \over {\partial{x^2}}}y\left( x \right) + f\left( x \right)y\left( x \right) = F\left( x \right)$$

For the Kohn-Sham equation, we rewrite it in the form:

$$\frac{\partial^2}{\partial r^2} P(r)+\underbrace{2\left(\varepsilon-V^{KS}(r)\right)}_{f(r)} P(r)=\underbrace{0}_{F(r)} .$$

By using Taylor expansion for $y(x_{i+1})$ and $y(x_{i-1})$ around $x_i$ and combining them, we obtain the Numerov formula:

$$\left( {1 + {{{h^2}} \over {12}}{f_{i + 1}}} \right){y_{i + 1}} - \left( {2 - {{5{h^2}} \over 6}{f_i}} \right){y_i} + \left( {1 + {{{h^2}} \over {12}}{f_{i - 1}}} \right){y_{i - 1}} = {{{h^2}} \over {12}}\left( {{F_{i + 1}} + 10{F_i} + {F_{i - 1}}} \right) + {\cal O}(h^6)$$

## 6.3 Thomas Algorithm

The Thomas algorithm solves the system $A\mathbf{x} = \mathbf{d}$ where $A$ is tridiagonal:

1. Forward elimination:
   $$\alpha_{i+1} = -\frac{B_i}{A_i\alpha_i + C_i}$$
   $$\beta_{i+1} = \frac{Z_i - A_i\beta_i}{A_i\alpha_i + C_i}$$

2. Backward substitution:
   $$x_i = \alpha_{i+1}x_{i+1} + \beta_{i+1}$$

# 7. References

1. Pauling, L.; Wilson, E.B. Introduction to Quantum Mechanics, McGraw-Hill, New York, 1935.
2. Hartree, D.R. The calculation of atomic structures. Chapman & Hall, Ltd., London, 1957.
3. Parr, R.G.; Yang, W. Density functional theory of atoms and molecules. Oxford University. Press, New York, 1989.
4. Perdew, J.P.; Zunger, A. Self-interaction correction to density-functional approximations for many-electron systems. Phys. Rev. B 23, 5048, 1981.
5. Salvadori, M.G. Numerical methods in engineering. New York, 1952.
