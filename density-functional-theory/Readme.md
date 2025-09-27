# Radial Kohn-Sham Equation Solver for Hydrogen-like Atoms

## Introduction

This repository implements a numerical solution to the Kohn-Sham equations of Density Functional Theory (DFT) for hydrogen-like atoms. The implementation focuses on the radial component of the electronic wavefunction for atoms with spherical symmetry, solving the one-electron case with arbitrary nuclear charge Z. This provides a foundational example for understanding self-consistent field (SCF) methods in electronic structure calculations.

## Theoretical Background

### Density Functional Theory

Density Functional Theory is a quantum mechanical modeling method used in physics, chemistry, and materials science to investigate the electronic structure of many-body systems. The fundamental principle of DFT is that the properties of a many-electron system can be determined using functionals of the spatially-dependent electron density rather than the many-body wavefunction.

The theoretical foundation of DFT is based on two Hohenberg-Kohn theorems:

1. The ground-state properties of a many-electron system are uniquely determined by the electron density.
2. The electron density that minimizes the energy of the overall functional is the ground-state electron density.

### Kohn-Sham Equations

In practice, DFT calculations are performed using the Kohn-Sham approach, which replaces the original many-body problem by an auxiliary independent-particle problem. The Kohn-Sham equation in atomic units is:

$$\left[-\frac{1}{2}\frac{d^2}{dr^2} + \frac{l(l+1)}{2r^2}+V_\text{KS}(r)\right]P_{nl}(r) = \varepsilon_{nl}P_{nl}(r)$$

where $P_{nl}(r)=r\Psi_{nl}(r)$ is the radial wave function,

whereas $V_{\text{KS}}(\mathbf{r})$ is the Kohn-Sham potential, given by:

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

## Numerical Implementation

### Discretization

The radial equation is discretized using a uniform grid:

$$r_i = r_0 + i \cdot h, \quad i = 0, 1, ..., N-1$$

where $r_0$ is the starting point (close to the origin), $h$ is the step size, and $N$ is the number of grid points.

### Numerov Method

For solving the second-order differential equation, we employ the Numerov method, which provides sixth-order accuracy. The method transforms the equation into a three-point recursion relation:

$$(1 + \frac{h^2}{12} f_{i+1}) y_{i+1} - (2 - \frac{5h^2}{6} f_i) y_i + (1 + \frac{h^2}{12} f_{i-1}) y_{i-1} = 0$$

where $f_i = 2(E - V(r_i))$.

### Thomas Algorithm

The Thomas algorithm (also known as the tridiagonal matrix algorithm) is used to efficiently solve the tridiagonal system arising from the Numerov method:

1. Forward elimination phase creates an upper bidiagonal system
2. Backward substitution phase solves for the unknowns

This approach is computationally efficient with O(N) complexity.

### Self-Consistent Field Method

The SCF procedure follows these steps:
1. Initialize with an analytical solution for a hydrogen-like atom
2. Calculate the electron density from the wavefunction
3. Construct the Kohn-Sham potential
4. Solve the Kohn-Sham equation to obtain a new wavefunction
5. Mix the old and new potentials to improve convergence stability
6. Check for energy convergence
7. Repeat until convergence

### Exchange-Correlation Functional

We implement the Local Density Approximation (LDA) for the exchange-correlation functional:

- The **exchange potential** has a form:
$${V^X}[\rho] =  - {\left( {\frac{3}{\pi }\rho } \right)^{\frac{1}{3}}}$$

where $\rho \left( r \right) = {\left( {{{{P_{nl}}\left( r \right)} \over r}} \right)^2}$ is the electron density.
- The **correlation potential** uses the Perdew-Zunger parametrization of the Ceperley-Alder quantum Monte Carlo results:
$${V^C}[\rho] = \left\{ \begin{array}{l}A\ln {r_s} + \left( {B - \frac{1}{3}A} \right) + \frac{2}{3}C{r_s}\ln {r_s} + \frac{1}{3}\left( {2D - C} \right){r_s},\;\;\;\;{\rm{if}}\;{r_s} < 1;\\\gamma \frac{{\left( {1 + \frac{7}{6}{\beta _1}\sqrt {{r_s}}  + \frac{4}{3}{\beta _2}{r_s}} \right)}}{{{{\left( {1 + {\beta _1}\sqrt {{r_s}}  + {\beta _2}{r_s}} \right)}^2}}},\;\;\;\;\;\;\;{\rm{if}}\;\;\;{r_s} \ge 1;\end{array} \right.$$
  - For high densities ($r_s < 1$):
    $${V^C}[\rho] = A\ln {r_s} + \left( {B - \frac{1}{3}A} \right) + \frac{2}{3}C{r_s}\ln {r_s} + \frac{1}{3}\left( {2D - C} \right){r_s}$$
  - For low densities ($r_s \geq 1$):
    $${V^C}[\rho] = \gamma \frac{{\left( {1 + \frac{7}{6}{\beta _1}\sqrt {{r_s}}  + \frac{4}{3}{\beta _2}{r_s}} \right)}}{{{{\left( {1 + {\beta _1}\sqrt {{r_s}}  + {\beta _2}{r_s}} \right)}^2}}}$$

where ${r_s} = {\left( {\frac{3}{{4\pi \rho }}} \right)^{\frac{1}{3}}}$ is the Wigner-Seitz radius.

## Implementation Details

### `RadialDFT` Class

The `RadialDFT` class in `DFT.py` encapsulates the numerical solution of the radial Kohn-Sham equation. Key methods include:

- `initialize()`: Sets up the initial wavefunction guess using the analytical solution
- `normalize()`: Normalizes the wavefunction using the trapezoidal rule
- `get_v_nuc()`, `get_v_ee()`, `get_v_xc()`: Calculate the potential components
- `get_v_ks()`: Combines the potential components into the Kohn-Sham potential
- `solve_ks()`: Solves the Kohn-Sham equation using the Numerov method
- `ks_energy()`, `e_xc()`, `total_energy()`: Calculate energy components
- `scf_loop()`: Implements the self-consistent field iteration

### Boundary Conditions

The radial wavefunction satisfies:
- $P(r \rightarrow 0) \propto r^{l+1}$ (for 1s orbital with $l=0$, so $P(0) = 0$)
- $P(r \rightarrow \infty) = 0$

### Energy Calculation

The total energy is calculated as:

$$E_{tot} = E_{ks} + E_{ee} + E_{xc} - \int n(r)V_{xc}(r)dr$$

where:
- $E_{ks}$ is the Kohn-Sham eigenvalue
- $E_{ee}$ is the electron-electron repulsion energy
- $E_{xc}$ is the exchange-correlation energy
- The last term corrects for double-counting in the Kohn-Sham potential

## Usage

### Prerequisites

- Python 3.x
- NumPy
- Matplotlib
- Logging (standard library)

### Running the Code

The main script (`main.py`) initializes the DFT calculation with the following parameters:

```python
# Parameters
Z = 6       # Nuclear charge for hydrogen-like atom
r0 = 1e-5   # Minimum radius (a.u.)
rf = 20.0   # Maximum radius (a.u.)
N = 10001   # Number of grid points

alpha = 0.1  # Mixing parameter for SCF
prec = 1e-5  # Convergence tolerance
max_iter = 500  # Maximum number of SCF iterations
```

Execute the calculation:

```bash
python main.py
```

### Output Files

The code generates several output files:

1. `wavefunction.dat`: Contains the radial wavefunction $P(r)$ at each iteration
2. `potential.dat`: Contains the different potential components
3. `energy.dat`: Contains the energy components at each iteration
4. `wavefunction_comparison.png`: Plot comparing numerical and analytical wavefunctions
5. `potential_comparison.png`: Plot of the different potential components

## Validation

The implementation is validated by comparing the numerical solution with the analytical solution for hydrogen-like atoms. For a hydrogen-like atom with nuclear charge Z, the analytical 1s wavefunction is:

$$P_\text{analytical}(r) = r\Psi_\text{analytical}(r) = \frac{Z^{\frac32}}{\sqrt{\pi}}re^{-Zr}$$

with energy $E = -\frac{Z^2}{2}$ (in atomic units).

## Results and Discussion

### Convergence Properties

The convergence of the SCF procedure is monitored by tracking the total energy difference between consecutive iterations. The mixing parameter `alpha` controls the convergence stability; smaller values generally provide more stable convergence but may require more iterations.

### Energy Components

The final energy is broken down into components:
- Kohn-Sham eigenvalue energy
- Electron-electron repulsion energy
- Exchange-correlation energy (from density integral)
- Exchange-correlation energy (from potential)

### Numerical Accuracy

The accuracy of the numerical solution depends on:
1. Grid resolution (number of points and distribution)
2. Convergence threshold
3. Boundary conditions handling
4. Quality of exchange-correlation functional

## Theoretical Extensions

This implementation can be extended in several ways:

1. **Multi-electron systems**: Implementing multiple Kohn-Sham orbitals and handling orbital occupations
2. **Alternative XC functionals**: Implementing GGA (Generalized Gradient Approximation) or hybrid functionals
3. **Excited states**: Adapting the solver for excited state calculations
4. **Non-spherical systems**: Extending beyond radial symmetry
5. **Relativistic effects**: Including scalar relativistic corrections

## Mathematical Appendix

### Radial Kohn-Sham Equation

Starting from the three-dimensional Schrödinger equation in spherical coordinates:

$$\Psi \left( {r,\theta ,\varphi } \right) = \Psi \left( r \right)Y\left( {\theta ,\varphi } \right)$$

where $Y( {\theta ,\varphi })$ is spherical harmonics. It can be separated on two independent equations for $r$ and $\theta ,\varphi$. The Schrodinger equation in spherical coordinates for part depending on $r$ only, the **radial Schrodinger equation** is

$$\left[-\frac{1}{2r}\frac{d^2}{dr^2}r + \frac{l(l+1)}{2r^2} + V_\text{nuc}(r)\right]\Psi_{nl}(r) = E_{nl}\Psi_{nl}(r)$$

where $V^{NUC}(r)=-\frac{Z}{r}$. For the ground state (1s orbital) with $n=1$, $l=0$, the centrifugal term vanishes, resulting in:

$$\left[-\frac{1}{2}\frac{d^2}{dr^2} + V_\text{nuc}(r)\right]\Psi_{1s}(r) = E\Psi_{1s}(r)$$

### Numerov Method

The Numerov method can be used to solve differential equations of the kind

$${{{d^2}} \over {d{x^2}}}y\left( x \right) + f\left( x \right)y\left( x \right) = F\left( x \right)$$

For the Kohn-Sham equation, we rewrite it in the form:

$$\frac{d^2}{d x^2} y(x)+\underbrace{2\left(\varepsilon-V^{K S}(x)\right)}_{f(x)} y(x)=\underbrace{0}_{F(x)} .$$

By using Taylor expansion for $y(x_{i+1})$ and $y(x_{i-1})$ around $x_i$ and combining them, we obtain the Numerov formula:

$$\left( {1 + {{{h^2}} \over {12}}{f_{i + 1}}} \right){y_{i + 1}} - \left( {2 - {{5{h^2}} \over 6}{f_i}} \right){y_i} + \left( {1 + {{{h^2}} \over {12}}{f_{i - 1}}} \right){y_{i - 1}} = {{{h^2}} \over {12}}\left( {{F_{i + 1}} + 10{F_i} + {F_{i - 1}}} \right) + {\cal O}(h^6)$$

### Thomas Algorithm

The Thomas algorithm solves the system $A\mathbf{x} = \mathbf{d}$ where $A$ is tridiagonal:

1. Forward elimination:
   $$\alpha_{i+1} = -\frac{B_i}{A_i\alpha_i + C_i}$$
   $$\beta_{i+1} = \frac{Z_i - A_i\beta_i}{A_i\alpha_i + C_i}$$

2. Backward substitution:
   $$x_i = \alpha_{i+1}x_{i+1} + \beta_{i+1}$$

## References

1. Pauling, L.; Wilson, E.B. Introduction to Quantum Mechanics, McGraw-Hill, New York, 1935.
2. Hartree, D.R. The calculation of atomic structures. Chapman & Hall, Ltd., London, 1957.
3. Parr, R.G.; Yang, W. Density functional theory of atoms and molecules. Oxford University. Press, New York, 1989.
4. Perdew, J.P.; Zunger, A. Self-interaction correction to density-functional approximations for many-electron systems. Phys. Rev. B 23, 5048, 1981.
5. Salvadori, M.G. Numerical methods in engineering. New York, 1952.
