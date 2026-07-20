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

    # Centred LOS peculiar-velocity sightlines: shape (N_halo, N_x)
    v_pec_halo = np.load(PATHS["v_pec_halo"])

    z = Z_REDSHIFT

    # ── Compute circular velocity for each halo [km/s] ─────────────
    # v_c = 142.85 * [0.3*(1+z)^3 + 0.7]^(1/6) * (M_h/1e12)^(1/3)
    v_c = (
        142.85
        * (0.3 * (1 + z)**3 + 0.7)**(1 / 6)
        * (halomass / 1e12)**(1 / 3)
    )

    # ── Build velocity grid [km/s] ─────────────────────────────────
    # Must exactly match the grid used in spec.py.
    v_grid = np.arange(-2000.0, 2000.0 + 5.0, 5.0)

    if tau.shape[1] != len(v_grid):
        raise ValueError(
            f"tau has {tau.shape[1]} velocity bins, but damping.py "
            f"expects {len(v_grid)} bins."
        )

    # ── Halo bulk LOS velocity [km/s] ──────────────────────────────
    # gen.py centres each halo at the midpoint of its velocity sightline.
    i_center = v_pec_halo.shape[1] // 2
    v_halo_los = v_pec_halo[:, i_center]

    # ── Construct normalized Gaussian J_nu(v) per halo ─────────────
    # The intrinsic profile is shifted by both:
    #   1. the halo bulk LOS velocity, and
    #   2. the intrinsic redward Ly-alpha offset, 1.5 * v_c.
    means = v_halo_los + 1.5 * v_c
    std_dev = np.full_like(means, 88.0)

    # Gaussian PDFs, shape (N_halo, N_vel)
    Gaussian = np.vstack([
        norm.pdf(v_grid, loc=mu, scale=sigma)
        for mu, sigma in zip(means, std_dev)
    ])

    # Normalize each halo's Gaussian so its peak is 1.
    # This normalization cancels in the transmission ratio.
    Gaussian /= Gaussian.max(axis=1)[:, None]

    # Area under the intrinsic Ly-alpha line profile.
    J_area = Gaussian.sum(axis=1)

    # ── Apply IGM transmission: F(v) = J_nu(v) exp[-tau(v)] ───────
    F = Gaussian * np.exp(-tau)
    F_area = F.sum(axis=1)

    # EW and Ly-alpha luminosity transmission ratio.
    ratio = F_area / J_area

    # ── Save output ────────────────────────────────────────────────
    np.save(PATHS["damping"], ratio)
    print(f"[damping] Saved damping ratio → {PATHS['damping']}")


if __name__ == "__main__":
    run_damping()
