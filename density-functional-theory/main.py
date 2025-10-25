#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wang Jianghai @NTU
Contact: jianghai001@e.ntu.edu.sg
Date: 2025-09-09
Description: Main script to run the radial DFT solver for a hydrogen-like atom.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging

from DFT import RadialDFT, initialize_mesh


logger = logging.getLogger("RadialDFT")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def plot(dft: RadialDFT, wavefunc=True, potential=True, savefig=True):
    """
    Plot numerical vs analytical wavefunctions and potentials.
    """
    r = dft.r
    if wavefunc:
        plt.figure(figsize=(12, 5))
        plt.plot(r, dft.P, label='Numerical P(r)', color='blue', lw=2)
        plt.plot(r, dft.P_analytical, label='Analytical P(r)', color='orange', linestyle='--', lw=2)
        plt.xlabel('r (a.u.)')
        plt.ylabel('Radial Wavefunction P(r)')
        plt.xlim([0, 0.25])

        plt.title('Radial Wavefunction Comparison')
        plt.legend()
        plt.grid()
        plt.xlim(0, 0.25)

        plt.tight_layout()

        if savefig:
            plt.savefig("wavefunction.png", dpi=300)
        else:
            plt.show()

    if potential:
        plt.figure(figsize=(12, 5))
        plt.plot(r, dft.v_nuc, label='V_nuc(r)', color='black', lw=2)
        plt.plot(r, dft.v_ee, label='V_ee(r)', color='green', lw=2)
        plt.plot(r, dft.v_xc, label='V_xc(r)', color='purple', lw=2)
        plt.plot(r, dft.v_ks, label='V_KS(r)', color='red', lw=2)
        plt.xlabel('r (a.u.)')
        plt.ylabel('Potential (a.u.)')
        plt.title('Radial Potentials')
        plt.legend()
        plt.grid()
        plt.xlim(0, 0.15)
        plt.ylim(-300, 50)

        plt.tight_layout()

        if savefig:
            plt.savefig("potential.png", dpi=300)
        else:
            plt.show()


def main(Z, r0, rf, N, alpha, prec, max_iter, visualize=True, verbose=True, save=True):
    if not verbose:
        logger.setLevel(logging.CRITICAL)

    # Initialize radial mesh
    r, h = initialize_mesh(r0, rf, N)
    logger.info(f"Mesh initialized: r0={r0}, rf={rf}, N={N}, h={h:.6f}")

    # Initialize DFT solver
    solver = RadialDFT(Z, r, h)
    solver.initialize()            # Analytical initial guess
    logger.info(f"Wavefunction initialized for Z={Z}")
    solver.norm()

    logger.info("Starting SCF loop...")
    data, P_analytical, E_tot = solver.scf_loop(prec, alpha, max_iter)  # Run SCF loop
    logger.info("SCF loop completed.")

    if visualize:
        plot(solver, savefig=save)

    P_history = data['P']
    V_KS = data['V_ks']
    V_nuc = data['V_nuc']
    V_ee = data['V_ee']
    V_xc = data['V_xc']
    E_tot_history = data['TotE']
    E_ks_history = data['E_ks']
    E_ee_history = data['E_ee']
    E_xc_history = data['E_xc']
    E_xc1_history = data['E_xc1']
    dE_history = data['dE']

    logger.info(f"\nTotal Energy: {E_tot_history[-1]:.6f} a.u."
                f"\nKohn-Sham energy: {E_ks_history[-1]:.6f} a.u."
                f"\nElectron-electron energy: {E_ee_history[-1]:.6f} a.u."
                f"\nExchange-correlation energy (density integral): {E_xc_history[-1]:.6f} a.u."
                f"\nExchange-correlation energy (from potential): {E_xc1_history[-1]:.6f} a.u.")

    logger.info("Writing wavefunctions and potentials...")

    if save:
        with open("wavefunction.dat", "w") as f:
            f.write(f"#  Wavefunction Data\n")
            f.write(f"Total Iterations: {len(P_history)}\n\n")
            for i in range(1, len(P_history)+1):
                f.write(f"# Iteration {i}:\n")
                f.write(f"#\tr(a.u.)\tP(r)\tP0(r)\terror (P(i)-P0(i))\n")
                for j in range(N):
                    f.write(f"{r[j]:<20.15f}\t{P_history[i-1][j]:<20.15f}\t{P_analytical[j]:<20.15f}\t{(P_analytical[j] - P_history[i-1][j]):<20.12e}\n")
                f.write("\n")
            f.write(f"# Iteration {len(P_history)} - Final Results:\n")
            f.write(f"#\tr(a.u.)\tP(r)\tP0(r)\terror (P(i)-P0(i))\n")
            for j in range(N):
                f.write(f"{r[j]:<20.15f}\t{P_history[-1][j]:<20.15f}\t{P_analytical[j]:<20.15f}\t{(P_analytical[j] - P_history[-1][j]):<20.12e}\n")

        with open("potential.dat", "w") as f:
            f.write(f"#  Potential Data\n")
            f.write(f"Total Iterations: {len(V_KS)}\n\n")
            for i in range(1, len(V_KS)+1):
                f.write(f"# Iteration {i}:\n")
                f.write(f"#\tr(a.u.)\tV_nuc(r)\tV_ee(r)\tV_xc(r)\tV_ks(r)\n")
                for j in range(N):
                    f.write(f"{r[j]:<20.15f}{V_nuc[i-1][j]:<20.15f}{V_ee[i-1][j]:<20.15f}{V_xc[i-1][j]:<20.15f}{V_KS[i-1][j]:<20.15f}\n")
                f.write("\n")
            f.write(f"# Iteration {len(V_KS)} - Final Results:\n")
            f.write(f"#\tr(a.u.)\tV_nuc(r)\tV_ee(r)\tV_xc(r)\tV_ks(r)\n")
            for j in range(N):
                f.write(f"{r[j]:<20.15f}\t{V_nuc[i-1][j]:<20.15f}\t{V_ee[-1][j]:<20.15f}\t{V_xc[-1][j]:<20.15f}\t{V_KS[-1][j]:<20.15f}\n")

        with open("energy.dat", "w") as f:
            f.write(f"#\tIteration\tTotE\tE_KS\tE_ee\tE_xc\tE_xc1\tdE\n")
            for i in range(len(E_tot_history)):
                f.write(f"\t{i}\t{E_tot_history[i]:<20.15f}\t{E_ks_history[i]:<20.15f}\t{E_ee_history[i]:<20.15f}\t{E_xc_history[i]:<20.15f}\t{E_xc1_history[i]:<20.15f}\t{dE_history[i] if dE_history[i] is not None else 0.0:<20.12e}\n")

    return E_ks_history[-1], len(E_ks_history)


if __name__ == "__main__":
    # Parameters
    Z = 6               # Nuclear charge for Hydrogen-like atom
    r0 = 1e-5           # Minimum radius
    rf = 20.0           # Maximum radius
    N = 8000            # Number of mesh points

    alpha = 0.1         # Mixing parameter for SCF
    prec = 1e-5         # Convergence tolerance
    max_iter = 300      # Maximum number of SCF iterations

    eps, iters = main(Z, r0, rf, N, alpha, prec, max_iter)
    print(f"Final Kohn-Sham energy: {eps:.6f} a.u. after {iters} iterations.")
