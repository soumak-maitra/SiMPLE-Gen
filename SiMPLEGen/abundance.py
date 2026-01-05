#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
abundance.py
------------
Step 3 of the Lyman-α emitter pipeline:
Perform abundance matching of halo masses to UV luminosities,
including a duty cycle correction.
"""
import numpy as np
import matplotlib.pyplot as plt
from hmf import MassFunction
from scipy.interpolate import interp1d
from astropy.cosmology import Planck18 as cosmo

from .config import PATHS, Z_REDSHIFT

def run_abundance():
    # ── Parameters ────────────────────────────────────────────────────────
    Redshift   = Z_REDSHIFT
    Halo_Mmin  = 8.0    # log10(M_min / (M⊙/h))
    Halo_Mmax  = 15.0   # log10(M_max / (M⊙/h))
    dlog10m    = 0.1

    # ── Step 1: Halo Mass Function (Sheth–Tormen) ─────────────────────────
    mf = MassFunction(
        hmf_model="SMT",
        z=Redshift,
        Mmin=Halo_Mmin,
        Mmax=Halo_Mmax,
        dlog10m=dlog10m
    )
    halo_masses = mf.m  # array of halo masses [M⊙/h]
    # differential mass bin widths
    logm_edges = np.linspace(Halo_Mmin, Halo_Mmax, len(halo_masses)+1)
    dM = (10**logm_edges[1:] - 10**logm_edges[:-1])

    # ── Step 2: UV Luminosity Function (Schechter; Bouwens+2015) ─────────
    phi_star = 0.47 * (10.0**(-0.27*(Redshift-6))) * 1e-3  # [Mpc⁻³]
    M_star   = -20.95 + 0.01*(Redshift-6)
    alpha    = -1.87 - 0.1*(Redshift-6)

    # M_uv range (bright → faint)
    M_uv = np.linspace(-26, -12, 100)[::-1]

    def schechter_uv(M):
        x = 10**(0.4*(M_star - M))
        return (0.4 * np.log(10) * phi_star) * (x**(alpha+1)) * np.exp(-x)

    phi_uv = schechter_uv(M_uv)
    # cumulative UV number density (descending M_uv)
    dMuv = np.abs(M_uv[1] - M_uv[0])
    cumulative_uv = np.cumsum(phi_uv[::-1])[::-1] * dMuv

    # ── Step 3: Duty Cycle Correction ────────────────────────────────────
    # time interval ~ 200 Myr → Δz ≈ 0.39 at z≈7
    deltaz = 0.39
    mf_prev = MassFunction(
        hmf_model="SMT",
        z=Redshift + deltaz,
        Mmin=Halo_Mmin,
        Mmax=Halo_Mmax,
        dlog10m=dlog10m
    )
    # build cumulative HMF at z and z+Δz
    cum_hmf = np.cumsum(mf.dndm[::-1] * dM[::-1])[::-1]
    cum_prev = np.cumsum(mf_prev.dndm[::-1] * dM[::-1])[::-1]
    # duty cycle fraction ε_dc
    epsilon_dc = (cum_hmf - cum_prev) / cum_hmf
    cum_hmf_dc = cum_hmf * epsilon_dc

    # ── Step 4: Abundance Matching ───────────────────────────────────────
    # map cumulative number densities ↔ M_uv and halo mass
    Dis_to_Muv = interp1d(cumulative_uv, M_uv, fill_value="extrapolate")
    HM_to_Dis  = interp1d(halo_masses, cum_hmf_dc, fill_value="extrapolate")

    # ── Load your halo masses from the gen step ──────────────────────────
    halomass = np.load(PATHS["halomass"])  # array of selected halo masses

    # match each halo mass → UV magnitude
    Muv_output = Dis_to_Muv(HM_to_Dis(halomass))

    # ── Save result ─────────────────────────────────────────────────────
    np.save(PATHS["Muv_grid"], Muv_output)
    print(f"[abundance] Saved Muv_grid → {PATHS['Muv_grid']}")


if __name__ == "__main__":
    run_abundance()

