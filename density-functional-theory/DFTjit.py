#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wang Jianghai @NTU
Contact: jianghai001@e.ntu.edu.sg
Date: 2025-xx-xx
Description: [Brief description of the script's purpose]
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wang Jianghai @NTU
Contact: jianghai001@e.ntu.edu.sg
Date: 2025-09-09 (modified)
Description: Radial DFT solver for H-like atoms (1s state) with SCF loop.
             Mixed approach: Python class + Numba-accelerated numeric kernels.
"""

import numpy as np
import logging
from numba import njit, float64, int64

logger = logging.getLogger("RadialDFT")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def initialize_mesh(r0=1e-5, rf=20.0, N=10000):
    r = np.linspace(r0, rf, N)
    h = (rf - r0) / (N - 1)
    return r, h


@njit
def trapezoid_num(y, x):
    """Simple trapezoidal integrator compatible with njit."""
    n = y.shape[0]
    if n < 2:
        return 0.0
    s = 0.0
    for i in range(n - 1):
        s += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return s

@njit
def compute_P_analytical_num(r, Z):
    """Analytical 1s radial P(r) = r * (Z**1.5) * exp(-Z r) / sqrt(pi)"""
    n = r.shape[0]
    P = np.empty(n, dtype=np.float64)
    coef = (Z ** 1.5) / np.sqrt(np.pi)
    for i in range(n):
        P[i] = r[i] * coef * np.exp(-Z * r[i])
    return P

@njit
def compute_norm_num(P, r):
    """Compute norm = sqrt(4π ∫ P^2 dr) and normalize P in-place is not possible here,
       so return norm. (Normalization will be done in Python by dividing the array.)"""
    integrand = np.empty(P.shape[0], dtype=np.float64)
    for i in range(P.shape[0]):
        integrand[i] = P[i] * P[i]
    I = trapezoid_num(integrand, r)
    norm = np.sqrt(4.0 * np.pi * I)
    return norm

@njit
def get_v_nuc_num(Z, r):
    n = r.shape[0]
    v = np.empty(n, dtype=np.float64)
    for i in range(n):
        v[i] = -Z / r[i]
    return v

@njit
def get_v_ee_num(P, r, Z, r0, h):
    """Compute electron-electron potential using discretized linear system (as in your Numerov-like solver).
       Returns v_ee array of length N.
    """
    N = len(r)
    h2 = h * h
    Y = np.zeros(N)
    alpha = np.zeros(N)
    beta = np.zeros(N)

    alpha[1] = 0.0
    beta[1] = Z * r0

    for i in range(1, N - 1):
        Bi, Ai, Ci = 1.0, 1.0, -2.0
        Zi = (-4.0 * np.pi * h2 / 12.0) * (
                (P[i - 1] ** 2) / r[i - 1] + 10.0 * (P[i] ** 2) / r[i] + (P[i + 1] ** 2) / r[i + 1]
        )
        AC = Ai * alpha[i] + Ci
        alpha[i + 1] = -Bi / AC
        beta[i + 1] = (Zi - Ai * beta[i]) / AC

    # Boundary values
    Y[-1] = 1.0
    for i in range(N - 2, 0, -1):
        Y[i] = alpha[i + 1] * Y[i + 1] + beta[i + 1]
    Y[0] = beta[1]
    return Y / r

@njit
def get_v_xc_num(P, r):
    """Local density approximation exchange-correlation potential (Perdew-Zunger like param).
       Implements the formula you used, in njit-friendly form.
    """
    N = r.shape[0]
    vxc = np.zeros(N, dtype=np.float64)

    A, B, C, D = 0.0311, -0.0480, 0.0020, -0.0116
    b1, b2, g = 1.0529, 0.3334, -0.1423

    rho = (P / r) ** 2
    rs = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
    Vx = - (3.0 * rho / np.pi) ** (1.0 / 3.0)

    # correlation potential
    mask = rs < 1.0
    Vc = np.zeros_like(rs)

    Vc[mask] = (
            A * np.log(rs[mask])
            + (B - (1.0 / 3.0) * A)
            + (2.0 / 3.0) * C * rs[mask] * np.log(rs[mask])
            + (1.0 / 3.0) * (2.0 * D - C) * rs[mask]
    )

    Vc[~mask] = (
            g * (1.0 + (7.0 / 6.0) * b1 * np.sqrt(rs[~mask]) + (4.0 / 3.0) * b2 * rs[~mask])
            / (1.0 + b1 * np.sqrt(rs[~mask]) + b2 * rs[~mask]) ** 2
    )

    vxc = Vx + Vc

    return vxc

@njit
def solve_ks_num(P_init, V, h, eps):
    """Numerov + Thomas style solver implemented in njit.
       Inputs:
         P_init: boundary values (array with P[0] and P[-1] meaningful)
         V: potential array
         h: grid spacing
         eps: eigenvalue (energy)
       Returns: P (solution array, same length)
    """
    N = P_init.shape[0]
    h2 = h * h

    Y  = np.zeros(N, dtype=np.float64)
    alpha = np.zeros(N, dtype=np.float64)
    beta  = np.zeros(N, dtype=np.float64)

    alpha[1] = 0.0
    beta[1]  = P_init[0]

    # forward sweep
    for i in range(1, N - 1):
        fi   = 2.0 * (-V[i]   + eps)
        fi1  = 2.0 * (-V[i+1] + eps)
        fi_1 = 2.0 * (-V[i-1] + eps)

        Ai =  1.0 + h2 / 12.0 * fi_1
        Bi =  1.0 + h2 / 12.0 * fi1
        Ci = -2.0 + 5.0 * h2 / 6.0 * fi

        AC = Ai * alpha[i] + Ci
        alpha[i + 1] = - Bi / AC
        beta[i + 1]  = - Ai * beta[i] / AC

    # backward sweep with boundary conditions from P_init
    Y[0]  = P_init[0]
    Y[-1] = P_init[-1]

    for i in range(N - 2, 0, -1):
        Y[i] = alpha[i + 1] * Y[i + 1] + beta[i + 1]

    norm = np.sqrt(4*np.pi * np.trapz(Y**2, r))
    Y /= norm

    return Y

@njit
def E_ks_num(P, V, r):
    """Compute Kohn-Sham eigenvalue by finite-difference second derivative and integration."""
    N = P.shape[0]
    h2 = (r[1] - r[0])**2
    d2P = np.zeros(N, dtype=np.float64)

    if N >= 3:
        for i in range(1, N - 1):
            d2P[i] = (P[i + 1] - 2.0 * P[i] + P[i - 1]) / h2
        # forward/backward one-sided for edges
        d2P[0] = (P[1] - 2.0 * P[0]) / h2
        d2P[-1] = (P[-2] - 2.0 * P[-1]) / h2
    else:
        # fallback
        for i in range(N):
            d2P[i] = 0.0

    integrand = np.empty(N, dtype=np.float64)
    for i in range(N):
        integrand[i] = P[i] * (-0.5 * d2P[i] + V[i] * P[i])

    eps = 4.0 * np.pi * trapezoid_num(integrand, r)
    return eps

@njit
def e_xc_num(P, r):
    """Exchange-correlation energy density (Ex + Ec) per point."""
    N = P.shape[0]
    E_xc = np.zeros(N, dtype=np.float64)

    A, B, C, D = 0.0311, -0.0480, 0.0020, -0.0116
    b1, b2, g = 1.0529, 0.3334, -0.1423

    rho = (P / r) ** 2
    rs = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
    Ex = -0.75 * (3.0 * rho / np.pi) ** (1.0 / 3.0)

    mask = rs < 1.0
    Ec = np.zeros_like(rs)

    Ec[mask] = A * np.log(rs[mask]) + B + C * rs[mask] * np.log(rs[mask]) + D * rs[mask]

    Ec[~mask] = g / (1.0 + b1 * np.sqrt(rs[~mask]) + b2 * rs[~mask])

    E_xc = Ex + Ec

    return E_xc

@njit
def E_tot_num(P, r, V_ee, V_xc):
    """Return E_ee, E_xc_density, E_xc_vxc (all scalars)."""
    N = P.shape[0]
    rho = np.empty(N, dtype=np.float64)
    for i in range(N):
        rho[i] = (P[i] / r[i])**2

    exc_density = e_xc_num(P, r)

    # integrals: use 4π ∫ ρ(r) * f(r) * r^2 dr
    integrand_ee = np.empty(N, dtype=np.float64)
    integrand_exc = np.empty(N, dtype=np.float64)
    integrand_vxc = np.empty(N, dtype=np.float64)
    for i in range(N):
        rr2 = r[i] * r[i]
        integrand_ee[i] = rho[i] * V_ee[i] * rr2
        integrand_exc[i] = rho[i] * exc_density[i] * rr2
        integrand_vxc[i] = rho[i] * V_xc[i] * rr2

    E_ee = -0.5 * 4.0 * np.pi * trapezoid_num(integrand_ee, r)
    E_xc = 4.0 * np.pi * trapezoid_num(integrand_exc, r)
    E_xc1 = -4.0 * np.pi * trapezoid_num(integrand_vxc, r)

    return E_ee, E_xc, E_xc1

class RadialDFT:
    def __init__(self, Z: float, r: np.ndarray, h: float):
        self.Z = float(Z)
        self.r = np.asarray(r, dtype=np.float64)
        self.h = float(h)
        self.N = len(self.r)
        self.r0 = float(self.r[0])

        # numeric arrays
        self.P = np.zeros_like(self.r, dtype=np.float64)                  # numerical solution arrays
        self.P_analytical = np.zeros_like(self.r, dtype=np.float64)       # analytical solution
        self.v_nuc = np.zeros_like(self.r, dtype=np.float64)
        self.v_ee  = np.zeros_like(self.r, dtype=np.float64)
        self.v_xc  = np.zeros_like(self.r, dtype=np.float64)
        self.v_ks  = np.zeros_like(self.r, dtype=np.float64)

        # history (kept in Python)
        self.history = {
            "P": [],
            "V_ks": [],
            "V_nuc": [],
            "V_ee": [],
            "V_xc": [],
            "E_ks": [],
            "TotE": [],
            "E_ee": [],
            "E_xc": [],
            "E_xc1": [],
            "dE": []
        }

    def initialize(self):
        """Analytical solution for H-like 1s orbital as initial guess"""
        self.P_analytical = compute_P_analytical_num(self.r, self.Z)
        self.P = self.P_analytical.copy()
        logger.info(f"Initialized wave function with analytical 1s orbital for Z={self.Z}")

    def norm(self):
        norm = compute_norm_num(self.P, self.r)
        # normalize in Python (arrays are mutable)
        if norm > 0.0:
            self.P /= norm
        return norm

    def get_v_nuc(self):
        self.v_nuc = get_v_nuc_num(self.Z, self.r)
        return self.v_nuc

    def get_v_ee(self):
        self.v_ee = get_v_ee_num(self.P, self.r, self.Z, self.r0, self.h)
        return self.v_ee

    def get_v_xc(self):
        self.v_xc = get_v_xc_num(self.P, self.r)
        return self.v_xc

    def get_v(self):
        """Total Kohn-Sham potential V_KS = V_nuc + V_ee + V_xc"""
        v_nuc = self.get_v_nuc()
        v_ee  = self.get_v_ee()
        v_xc  = self.get_v_xc()
        self.v_ks = v_nuc + v_ee + v_xc
        return self.v_ks, self.v_nuc, self.v_ee, self.v_xc

    def solve_ks(self, V, eps):
        """
        Solve the Kohn-Sham equation using Numerov + Thomas method.
        We keep the eps dependence by using scaling: here we return P and norm.
        For robustness we use the njit linear solver solve_ks_num with P boundary values.
        """
        # Prepare boundary conditions
        P_init = np.zeros(self.N, dtype=np.float64)
        P_init[0] = self.P[0]
        P_init[-1] = self.P[-1]

        # The solve_ks_num implemented earlier assumed eps=0 inside
        # but your original algorithm couples eps into fi terms.
        # Simpler: use solve_ks_num as a stable propagation to get shape, then renormalize and compute eps by E_ks.
        Y = solve_ks_num(P_init, V, self.h, eps)

        self.P = Y
        norm = self.norm()
        return self.P, norm

    def E_ks(self, V):
        eps = E_ks_num(self.P, V, self.r)
        logger.debug(f"Kohn-Sham eigenvalue (energy): {eps}")
        return eps

    def e_xc(self):
        return e_xc_num(self.P, self.r)

    def E_tot(self, V_ee, V_xc):
        E_ee, E_xc, E_xc1 = E_tot_num(self.P, self.r, V_ee, V_xc)
        logger.debug(f"Total energies: E_ee={E_ee}, E_xc={E_xc}, E_xc1={E_xc1}")
        return E_ee, E_xc, E_xc1

    def _save_iteration(self, P, V_KS, V_nuc, V_ee, V_xc, E_ks, E_ee, E_xc, E_xc1, E_tot, dE):
        """Save current iteration data to history (kept in Python)."""
        self.history["P"].append(P.copy())
        self.history["V_ks"].append(V_KS.copy())
        self.history["V_nuc"].append(V_nuc.copy())
        self.history["V_ee"].append(V_ee.copy())
        self.history["V_xc"].append(V_xc.copy())

        self.history["E_ks"].append(E_ks)
        self.history["E_ee"].append(E_ee)
        self.history["E_xc"].append(E_xc)
        self.history["E_xc1"].append(E_xc1)
        self.history["TotE"].append(E_tot)
        self.history["dE"].append(dE)

    def scf_loop(self, prec=1e-5, alpha=0.1, Nmax=100, verbose=True):
        if not verbose:
            logger.setLevel(logging.CRITICAL)

        V_ks, Vnuc, Vee, Vxc = self.get_v()

        # Here we use only nuclear potential as initial guess instead of full V_ks
        V_old = Vnuc.copy()
        V_mixed = V_old.copy()

        eps = - 0.5 * self.Z**2

        E_ee, E_xc, E_xc1 = self.E_tot(Vee, Vxc)
        E_tot = eps + E_ee + E_xc + E_xc1
        d_eps = None

        for it in range(1, Nmax + 1):
            self.P, _ = self.solve_ks(V_mixed, eps=eps)
            self._save_iteration(self.P, V_mixed, self.v_nuc, Vee, Vxc, eps, E_ee, E_xc, E_xc1, E_tot, d_eps)

            logger.info(f"Iter {it}: ε = {eps:.8f}, Δε = {f'{d_eps:.3e}' if d_eps is not None else 'N/A'}")

            V_new, Vnuc, Vee, Vxc = self.get_v()
            V_mixed = alpha * V_new + (1.0 - alpha) * V_old

            eps_new = self.E_ks(V_mixed)

            E_ee, E_xc, E_xc1 = self.E_tot(Vee, Vxc)
            E_tot = eps_new + E_ee + E_xc + E_xc1

            d_eps = abs(eps_new - eps) if eps is not None else None

            if d_eps is not None and d_eps < prec:
                logger.info(f"Iter {it + 1}: ε = {eps_new:.8f}, Δε = {d_eps:.3e}")
                logger.info(f"🎯 SCF converged in {it + 1} iterations on eigenvalue: ε = {eps_new:.8f}")

                self._save_iteration(self.P, V_mixed, self.v_nuc, Vee, Vxc, eps_new, E_ee, E_xc, E_xc1, E_tot, d_eps)
                return self.history, self.P_analytical, E_tot

            eps = eps_new
            V_old = V_mixed.copy()

        logger.critical(f"⚠️ SCF NOT converged after {Nmax} iterations. Last Δε = {d_eps}")
        raise RuntimeError(f"SCF did not converge after {Nmax} iterations. Last Δε = {d_eps}")


if __name__ == "__main__":
    # Parameters
    Z = 6  # Nuclear charge for Hydrogen-like atom
    r0 = 1e-5  # Minimum radius
    rf = 10.0  # Maximum radius
    N = 1000  # Number of mesh points

    alpha = 0.1  # Mixing parameter for SCF
    prec = 1e-5  # Convergence tolerance
    max_iter = 500  # Maximum number of SCF iterations

    # Initialize radial mesh
    r, h = initialize_mesh(r0, rf, N)

    # Initial guess for wave function (1s state)
    solver = RadialDFT(Z, r, h)
    solver.initialize()
    norm = solver.norm()
    history, P_analytical, _ = solver.scf_loop(alpha=alpha, prec=prec, Nmax=max_iter)
