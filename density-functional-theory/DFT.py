#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wang Jianghai @NTU
Contact: jianghai001@e.ntu.edu.sg
Date: 2025-09-09
Description: Radial DFT solver for H-like atoms (1s state) with SCF loop.
"""

import numpy as np
import logging

logger = logging.getLogger("RadialDFT")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def initialize_mesh(r0 = 1e-5, rf = 20.0, N = 10000):
    r = np.linspace(r0, rf, N)
    h = (rf - r0) / (N - 1)

    return r, h


class RadialDFT:
    def __init__(self, Z: float, r: np.array, h: float):
        self.Z = Z
        self.r = r
        self.h = h
        self.N = len(r)
        self.r0 = r[0]
        self.P = np.zeros_like(r, dtype=float)                  # numerical solution arrays
        self.P_analytical = np.zeros_like(r, dtype=float)       # analytical solution
        self.v_nuc = np.zeros_like(r, dtype=float)
        self.v_ee = np.zeros_like(r, dtype=float)
        self.v_xc = np.zeros_like(r, dtype=float)
        self.v_ks = np.zeros_like(r, dtype=float)

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
        self.P_analytical = self.r * (self.Z**1.5) * np.exp(- self.Z * self.r) / np.sqrt(np.pi)
        self.P = self.P_analytical.copy()
        logger.info(f"Initialized wave function with analytical 1s orbital for Z={self.Z}")

    def norm(self):
        norm = np.sqrt(4.0 * np.pi * np.trapezoid(self.P ** 2, self.r))
        self.P /= norm
        return norm

    def get_v_nuc(self):
        """Nuclear potential V_nuc(r) = -Z/r"""
        self.v_nuc = -self.Z / self.r
        return self.v_nuc

    def get_v_ee(self):
        """Electron-electron repulsion potential V_ee(r) using Numerov method"""
        h2 = self.h**2
        assert self.N >= 3, "N must be >= 3 to run the forward/backward sweeps."

        Y  = np.zeros(self.N, dtype=float)
        alpha = np.zeros(self.N, dtype=float)
        beta = np.zeros(self.N, dtype=float)

        # ---- forward sweep ----
        alpha[1] = 0.0
        beta[1] = self.Z * self.r0

        for i in range(1, self.N-1):
            Bi, Ai, Ci = 1.0, 1.0, -2.0
            Zi = (-4.0 * np.pi * h2 / 12.0) * (
                    self.P[i-1]**2 / self.r[i-1] + 10.0 * self.P[i]**2 / self.r[i] + self.P[i+1]**2 / self.r[i+1])

            AC = Ai * alpha[i] + Ci
            alpha[i+1] = - Bi / AC
            beta[i+1] = (Zi - Ai * beta[i]) / AC

        # ---- backward sweep ----
        Y[-1] = 1.0
        for i in range(self.N-2, 0, -1):
            Y[i] = alpha[i+1] * Y[i+1] + beta[i+1]
        Y[0] = beta[1]

        self.v_ee = Y / self.r

        return self.v_ee

    def get_v_xc(self):
        """Exchange-correlation potential (LDA)"""
        A, B, C, D = 0.0311, -0.0480, 0.0020, -0.0116
        b1, b2, g = 1.0529, 0.3334, -0.1423

        # rho = (self.P / self.r)**2
        for i in range(self.N):
            rho = (self.P[i] / self.r[i])**2                         # charge density
            rs = (3.0 / (4.0 * np.pi * rho))**(1.0/3.0)
            Vx = -(3.0 * rho / np.pi)**(1.0/3.0)                     # exchange potential

            # correlation potential
            if rs < 1.0:
                Vc = (A * np.log(rs)
                      + (B - (1.0/3.0)*A)
                      + (2.0/3.0)*C*rs*np.log(rs)
                      + (1.0/3.0)*(2.0*D - C)*rs)
            else:
                Vc = (g * (1.0 + (7.0/6.0)*b1*np.sqrt(rs) + (4.0/3.0)*b2*rs) /
                      (1.0 + b1*np.sqrt(rs) + b2*rs)**2)

            self.v_xc[i] = Vx + Vc

        return self.v_xc

    def get_v(self):
        """Total Kohn-Sham potential V_KS = V_nuc + V_ee + V_xc"""
        v_nuc = self.get_v_nuc()
        v_ee = self.get_v_ee()
        v_xc = self.get_v_xc()
        self.v_ks = v_nuc + v_ee + v_xc

        return self.v_ks, self.v_nuc, self.v_ee, self.v_xc

    def solve_ks(self, V, eps):
        """Solve the Kohn-Sham equation using Numerov + Thomas method"""
        h2 = self.h**2

        Y  = np.zeros(self.N, dtype=float)
        alpha = np.zeros(self.N, dtype=float)
        beta = np.zeros(self.N, dtype=float)

        # ---- boundary condition P(r_0) ----
        # see formula (3) in https://www.dsedu.org/courses/dft/ks_eigenvector
        alpha[1] = 0.0
        beta[1] = self.P[0]

        # ---- forward sweep ----
        for i in range(1, self.N-1):
            fi   = 2.0 * (-V[i]   + eps)
            fi1  = 2.0 * (-V[i+1] + eps)
            fi_1 = 2.0 * (-V[i-1] + eps)

            # see Eq. (7) from Numerov method https://www.dsedu.org/courses/dft/numerov
            Ai =  1.0 + h2 / 12.0 * fi_1
            Bi =  1.0 + h2 / 12.0 * fi1
            Ci = -2.0 + 5.0 * h2 / 6.0 * fi

            # see Eq. (4) from Thomas method https://www.dsedu.org/courses/dft/thomas
            AC = Ai * alpha[i] + Ci
            alpha[i+1] = - Bi / AC
            beta[i+1] = - Ai * beta[i] / AC

        # ---- backward sweep ----
        Y[0] = self.P[0]                                    # the boundary condition P(r_0), see formula (2) in https://www.dsedu.org/courses/dft/ks_eigenvector
        Y[-1] = self.P[-1]                                  # the boundary condition P(r_f), see formula (2) in https://www.dsedu.org/courses/dft/ks_eigenvector

        for i in range(self.N-2, 0, -1):
            Y[i] = alpha[i+1] * Y[i+1] + beta[i+1]          # see Eq. (3) from Thomas method https://www.dsedu.org/courses/dft/thomas

        self.P = Y                                          # P(1) and P(N) are fixed
        norm = self.norm()

        return self.P, norm

    def E_ks(self, V):
        """Calculate eigenvalue (energy e_10) by integration KS equation"""
        h2 = self.h ** 2
        d2P = np.zeros_like(self.P)
        N = self.N
        for i in range(1, N - 1):
            d2P[i] = (self.P[i + 1] - 2.0 * self.P[i] + self.P[i - 1]) / h2     # Finite difference for second derivative

        d2P[0] = (self.P[1] - 2.0 * self.P[0]) / h2

        integrand = self.P * (-0.5 * d2P + V * self.P)

        eps = 4.0 * np.pi * np.trapezoid(integrand, self.r)

        logger.debug(f"Kohn-Sham eigenvalue (energy): {eps}")
        return eps

    def e_xc(self):
        """Exchange-correlation energy density"""
        A, B, C, D = 0.0311, -0.0480, 0.0020, -0.0116
        b1, b2, g = 1.0529, 0.3334, -0.1423

        E_xc = np.zeros_like(self.r, dtype=float)

        for i, pi in enumerate(self.P):
            rho = (pi / self.r[i])**2
            rs = (3.0 / (4.0 * np.pi * rho))**(1.0/3.0)
            Ex = -0.75 * (3.0 * rho / np.pi)**(1.0/3.0)
            if rs < 1.0:
                Ec = A * np.log(rs) + B + C * rs * np.log(rs) + D * rs
            else:
                Ec = g / (1.0 + b1 * np.sqrt(rs) + b2 * rs)

            E_xc[i] = Ex + Ec

        return E_xc


    def E_tot(self, V_ee, V_xc):
        """
        Return components for total energy AFTER SCF converged.
        Returns (E_ee, E_xc_density, E_xc_vxc) where
            E_ee = -0.5 * ∫ ρ V_ee d^3r  (to correct double counting),
            E_xc_density = ∫ ρ e_xc(r) d^3r,
            E_xc_vxc = - ∫ ρ V_xc(r) d^3r
        """
        rho = (self.P / self.r) ** 2
        exc_density = self.e_xc()

        E_ee = -0.5 * 4.0 * np.pi * np.trapezoid(rho * V_ee * self.r**2, self.r)
        E_xc = 4.0 * np.pi * np.trapezoid(rho * exc_density * self.r**2, self.r)
        E_xc1 = -4.0 * np.pi * np.trapezoid(rho * V_xc * self.r**2, self.r)

        logger.debug(f"Total energies: E_ee={E_ee}, E_xc={E_xc}, E_xc1={E_xc1}")
        return E_ee, E_xc, E_xc1

    def _save_iteration(self, P, V_KS, V_nuc, V_ee, V_xc, E_ks, E_ee, E_xc, E_xc1, E_tot, dE):
        """Save current iteration data to history."""
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
        V_mixed = V_old

        eps = - 0.5 * self.Z**2

        E_ee, E_xc, E_xc1 = self.E_tot(Vee, Vxc)
        E_tot = eps + E_ee + E_xc + E_xc1
        d_eps = None

        for it in range(1, Nmax + 1):
            self.P, _ = self.solve_ks(V_mixed, eps=eps)
            self._save_iteration(self.P, V_mixed, self.v_nuc, Vee, Vxc, eps, E_ee, E_xc, E_xc1, E_tot, d_eps)

            logger.info(f"Iter {it}: ε = {eps:.8f}, Δε = {f'{d_eps:.3e}' if d_eps is not None else "N/A"}")

            V_new, Vnuc, Vee, Vxc = self.get_v()
            V_mixed = alpha * V_new + (1.0 - alpha) * V_old

            eps_new = self.E_ks(V_mixed)

            E_ee, E_xc, E_xc1 = self.E_tot(Vee, Vxc)
            E_tot = eps_new + E_ee + E_xc + E_xc1

            d_eps = abs(eps_new - eps)

            # return self.history, self.P_analytical, E_tot
            if d_eps < prec:
                logger.info(f"Iter {it + 1}: ε = {eps_new:.8f}, Δε = {f'{d_eps:.3e}' if d_eps is not None else "N/A"}")
                logger.info(f"🎯 SCF converged in {it + 1} iterations on eigenvalue: ε = {eps_new:.8f}")

                self._save_iteration(self.P, V_mixed, self.v_nuc, Vee, Vxc, eps, E_ee, E_xc, E_xc1, E_tot, d_eps)

                return self.history, self.P_analytical, E_tot

            eps = eps_new
            V_old = V_mixed

        logger.critical(f"⚠️ SCF NOT converged after {Nmax} iterations. Last Δε = {d_eps:.6e}")
        raise RuntimeError(f"SCF did not converge after {Nmax} iterations. Last Δε = {d_eps:.6e}")


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
    history, P_analytical, _= solver.scf_loop(alpha=alpha, prec=prec, Nmax=max_iter)
