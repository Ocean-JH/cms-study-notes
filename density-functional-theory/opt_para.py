#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wang Jianghai @NTU
Contact: jianghai001@e.ntu.edu.sg
Date: 2025-10-08
Description: [Brief description of the script's purpose]
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from main import main

# Default parameters
Z = 6  # Nuclear charge for Hydrogen-like atom
r0 = 1e-5  # Minimum radius
rf = 20.0  # Maximum radius
N = 8000  # Number of mesh points

alpha = 0.1  # Mixing parameter for SCF
prec = 1e-5  # Convergence tolerance
max_iter = 500  # Maximum number of SCF iterations


def opt_integration_parameters(Z, r0, alpha=0.1, prec=1e-5, max_iter=500):
    # Parameter sweep over rf and N
    rf_values = np.linspace(5, 20, 16)
    n_values  = np.linspace(1000, 10000, 10, dtype=int)
    results = []

    warnings.filterwarnings("error", category=RuntimeWarning)
    for rf in rf_values:
        diverged = False
        for N in n_values:
            if diverged:
                print(f"⚠️ Skipping rf={rf}, N={N} due to previous divergence.")
                results.append((rf, N, np.nan, np.nan, "Skipped"))
                continue

            try:
                print(f"Start running with parameters: rf={rf}, N={N}\t\tRemaining cases: {len(rf_values)*len(n_values) - len(results) - 1}")

                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=RuntimeWarning)
                    eps, iters = main(Z, r0, rf, N, alpha, prec, max_iter, visualize=False, verbose=False, save=False)
                    results.append((rf, N, eps, iters, "Converged"))

                    print(f"  ✅ Kohn-Sham energy: {eps:.6f} a.u. after {iters} iterations.")
            except RuntimeWarning as e:
                results.append((rf, N, np.nan, np.nan, "Diverged"))
                print(f"  ❌ Failed for rf={rf}, N={N}: {e}")
                diverged = True

    df = pd.DataFrame(results, columns=["rf", "n", "eigval", "iterations", "status"])

    md_table = df.to_markdown(index=False, tablefmt="github", floatfmt=".6f")
    with open("int_para_sweep.md", "w") as f:
        f.write(md_table)

    df.loc[df["status"] != "Converged", ["eigval", "iterations"]] = np.nan

    plt.figure(figsize=(12, 5))

    for rf in sorted(df["rf"].unique()):
        subset = df[df["rf"] == rf]
        plt.plot(subset["n"], subset["eigval"], marker="o", label=f"rf={rf:.1f}")

    plt.xlabel("n (number of grid points)")
    plt.ylabel("Final eigenvalue")
    plt.title("Eigenvalue vs n (Converged only)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("int_para_convergence.png", dpi=300)
    # plt.show()

def opt_alpha(Z, r0, rf, N, prec=1e-5, max_iter=500):
    alpha_values = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    results = []

    warnings.filterwarnings("error", category=RuntimeWarning)
    for alpha in alpha_values:
        try:
            print(f"Start running with mixing parameter: alpha={alpha:.3f}\t\tRemaining cases: {len(alpha_values) - len(results) - 1}")

            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning)
                eps, iters = main(Z, r0, rf, N, alpha, prec, max_iter, visualize=False, verbose=False, save=False)
                results.append((alpha, eps, iters, "Converged"))

                print(f"  ✅ Kohn-Sham energy: {eps:.6f} a.u. after {iters} iterations.")
        except (RuntimeWarning, RuntimeError) as e:
            results.append((alpha, np.nan, np.nan, "Diverged"))
            print(f"  ❌ Failed for alpha={alpha:.3f}: {e}")

    df = pd.DataFrame(results, columns=["alpha", "eigval", "iterations", "status"])

    md_table = df.to_markdown(index=False, tablefmt="github", floatfmt=".6f")
    with open("alpha_sweep.md", "w") as f:
        f.write(md_table)

    df.loc[df["status"] != "Converged", ["eigval", "iterations"]] = np.nan

    plt.figure(figsize=(12, 5))


    plt.subplot(1, 2, 1)
    plt.plot(df["alpha"], df["eigval"], marker="o")
    plt.xlabel("Mixing parameter alpha")
    plt.ylabel("Final eigenvalue")
    plt.title("Eigenvalue vs Mixing Parameter")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(df["alpha"], df["iterations"], marker="o")
    plt.xlabel("Mixing parameter alpha")
    plt.ylabel("Number of iterations to converge")
    plt.title("Iterations vs Mixing Parameter")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("alpha_convergence.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    opt_alpha(Z, r0, rf=18.0, N=5000, prec=prec, max_iter=max_iter)