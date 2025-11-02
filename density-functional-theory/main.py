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
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(r, dft.P, label='Numerical P(r)', color='blue', lw=2)
        ax1.plot(r, dft.P_analytical, label='Analytical P(r)', color='orange', linestyle='--', lw=2)
        ax1.set_xlabel('r (a.u.)')
        ax1.set_ylabel('Radial Wavefunction P(r)', color='black')
        ax1.set_xlim(0.0, 2.0)
        ax1.set_ylim(0.0, np.max(dft.P) * 1.05)
        ax1.grid(True, linestyle='--', alpha=0.5)

        idx_ana_max = np.argmax(dft.P_analytical)
        r_ana_max, P_ana_max = r[idx_ana_max], dft.P_analytical[idx_ana_max]
        ax1.vlines(r_ana_max, 0, P_ana_max, color='red', linestyle=':', lw=1.5, alpha=0.8)
        ax1.text(r_ana_max, -0.03 * np.max(dft.P), f"r = {r_ana_max:.2f}",
                 color='red', fontsize=9, ha='center', va='top')

        # idx_num_max = np.argmax(dft.P)
        # r_num_max, P_num_max = r[idx_num_max], dft.P[idx_num_max]
        # ax1.vlines(r_num_max, 0, P_num_max, color='red', linestyle=':', lw=1.5, alpha=0.8)
        # ax1.text(r_num_max, -0.03 * np.max(dft.P), f"r = {r_num_max:.2f}",
        #          color='red', fontsize=9, ha='center', va='top')

        diff = dft.P - dft.P_analytical
        ax2 = ax1.twinx()
        ax2.fill_between(r, diff, 0, color='gray', alpha=0.3, label='Difference (Numerical - Analytical)')
        ax2.set_ylabel('Difference ΔP(r)', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        # ax2.set_ylim(-0.0005, np.max(diff) * 1.05)

        idx_max = np.argmax(diff)
        idx_min = np.argmin(diff)
        r_max, diff_max = r[idx_max], diff[idx_max]
        r_min, diff_min = r[idx_min], diff[idx_min]

        ax2.vlines(r_max, 0, diff_max, color='green', linestyle='--', lw=1.0, alpha=0.7)
        ax2.vlines(r_min, 0, diff_min, color='green', linestyle='--', lw=1.0, alpha=0.7)

        ax2.text(
            r_max, -0.02 * np.max(np.abs(diff)),
            f"r = {r_max:.2f}",
            color='red', fontsize=9, ha='center', va='top'
        )

        ax2.text(
            r_min, 0.02 * np.max(np.abs(diff)),
            f"r = {r_min:.2f}",
            color='blue', fontsize=9, ha='center', va='bottom'
        )

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

        plt.title('Radial Wavefunction Comparison')
        plt.tight_layout()

        if savefig:
            plt.savefig("data/wavefunction.png", dpi=300)
            plt.close(fig)
        else:
            plt.show()

    if potential:
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(r, dft.v_nuc, label='V_nuc(r)', color='black', lw=1)
        ax1.plot(r, dft.v_ee, label='V_ee(r)', color='green', lw=1)
        ax1.plot(r, dft.v_xc, label='V_xc(r)', color='purple', lw=1)
        ax1.plot(r, dft.v_ks, label='V_KS(r)', color='red', lw=1)

        ax1.set_xlabel('r (a.u.)')
        ax1.set_ylabel('Potential (a.u.)', color='black')
        ax1.set_xlim(0, 2.0)
        ax1.set_ylim(-20, 10)
        ax1.grid(True, linestyle='--', alpha=0.5)

        v_sc = dft.v_ee + dft.v_xc
        ax2 = ax1.twinx()
        ax2.fill_between(r, v_sc, 0, color='gray', alpha=0.25, label='V_ee + V_xc')
        ax2.set_ylabel('Self-consistent Correction Potential (a.u.)', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.set_ylim(0, np.max(v_sc) * 1.05)

        idx_max = np.argmax(v_sc)
        r_max, v_max = r[idx_max], v_sc[idx_max]

        ax2.vlines(r_max, 0, v_max, color='orange', linestyle='--', lw=1.5, alpha=0.7)

        ax2.text(
            r_max, -0.01 * np.max(np.abs(v_sc)),
            f"r = {r_max:.2f}",
            color='orange', fontsize=9, ha='center', va='top'
        )

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

        plt.title('Radial Potentials')
        plt.tight_layout()

        if savefig:
            plt.savefig("data/potential.png", dpi=300)
            plt.close(fig)
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
        with open("data/wavefunction.dat", "w") as f:
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

        with open("data/potential.dat", "w") as f:
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

        with open("data/energy.dat", "w") as f:
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
