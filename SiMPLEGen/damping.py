#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
damping.py
----------
Step 5 of the Lyman-α emitter pipeline:
Compute the damping transmission (EW decrease ratio) for each halo
by convolving the optical depth profile with a halo-velocity Gaussian
and measuring the fractional area suppression.
"""
import numpy as np
from scipy.stats import norm

from .config import PATHS, Z_REDSHIFT

def run_damping():
    # ── Load inputs ────────────────────────────────────────────────
    # tau_halo: shape (N_halo, N_vel)
    tau = np.load(PATHS["tau_halo"])
    # halomass: shape (N_halo,)
    halomass = np.load(PATHS["halomass"])
    # Muv: not strictly used here, but loaded if needed downstream
    Muv = np.load(PATHS["Muv_grid"])
    z = Z_REDSHIFT

    # ── Compute circular velocity for each halo [km/s] ─────────────
    # v_c = 142.85 * [0.3*(1+z)^3 + 0.7]^(1/3) * (M_h/1e12)^(1/3)
    v_c = 142.85 * (0.3 * (1 + z)**3 + 0.7)**(1/3) * (halomass / 1e12)**(1/3)

    # ── Build velocity grid [km/s] ─────────────────────────────────
    # must match the binning of tau (originally [-500,500] in steps of 2)
    v_grid = np.arange(-500, 501.0, 10.0)

    # ── Construct normalized Gaussian J_ν(v) per halo ───────────────
    means   = 1.5 * v_c
    std_dev = np.full_like(means, 88.0)

    # Gaussian PDFs, shape (N_halo, N_vel)
    Gaussian = np.vstack([
        norm.pdf(v_grid, loc=μ, scale=σ)
        for μ, σ in zip(means, std_dev)
    ])
    # normalize each halo’s Gaussian so peak = 1
    Gaussian /= Gaussian.max(axis=1)[:, None]

    # precompute area under J_ν
    J_area = Gaussian.sum(axis=1)

    # ── Apply damping: F(v) = J_ν(v) * exp[−τ(v)] ────────────────────
    F = Gaussian * np.exp(-tau)
    F_area = F.sum(axis=1)

    # EW decrease ratio = ∫F / ∫J
    ratio = F_area / J_area

    # ── Save output ─────────────────────────────────────────────────
    np.save(PATHS["damping"], ratio)
    print(f"[damping] Saved damping ratio → {PATHS['damping']}")

if __name__ == "__main__":
    run_damping()

