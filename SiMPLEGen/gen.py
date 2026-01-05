#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen.py
------
Step 1: Identify halos in the simulation box and extract
1D sightlines of n_HI, temperature, and v_pec around each halo.
"""
import numpy as np
import time
from astropy import units as u
from astropy.constants import c, m_p, m_e, k_B
from astropy.cosmology import Planck18 as cosmo

from .config import RAW, PATHS, Z_REDSHIFT, MH_CUT, BOX_SIZE

def run_gen():
    t0 = time.time()

    # ── Load raw grids ───────────────────────────────────────────
    Dbox   = np.load(RAW["density_cube"])     # [comoving units]
    Tbox   = np.load(RAW["temperature"])      # [K]
    Xbox   = np.load(RAW["ionization_cube"])  # dimensionless
    Vbox   = np.load(RAW["velocity"])         # [comoving km/s]
    halo_cm  = np.load(RAW["halo_positions"]) # shape (N_halo, 3)
    halomass = np.load(RAW["halo_masses"])    # shape (N_halo,)

    # ── Mass cut ─────────────────────────────────────────────────
    mask = halomass >= 10.0**MH_CUT
    halo_cm  = halo_cm[mask]
    halomass = halomass[mask]

    # ── Build x & z grids ────────────────────────────────────────
    N = Dbox.shape[0]
    # central index
    i_center = N//2
    # comoving coordinate [Mpc/h]
    x_sim = np.linspace(0, BOX_SIZE, N+1)[:-1]
    # phys cm
    x = x_sim * u.Mpc / cosmo.h

    # build z-grid via dz = dx * H(z)/c
    H_z   = cosmo.H(Z_REDSHIFT).to(u.km/u.s/u.Mpc).value
    dzdx  = H_z / c.to(u.km/u.s).value
    z = np.zeros_like(x_sim)
    z[0] = Z_REDSHIFT - (x_sim[i_center] - x_sim[0]) * dzdx
    for i in range(len(z)-1):
        z[i+1] = z[i] + (x_sim[i+1] - x_sim[i]) * dzdx

    # ── Compute neutral‐H density sightlines ─────────────────────
    # total H number density n_H = Dbox / (physical cell volume)
    nHbox  = Dbox / (u.m.to(u.cm)/(1+Z_REDSHIFT))**3
    nHIbox = nHbox * (1 - Xbox)   # neutral fraction

    # grid‐index of each halo
    halo_idx = (halo_cm * (N-1)/BOX_SIZE).astype(int)

    # extract LOS at each halo (along x-axis)
    n_HI_halo  = nHIbox[:, halo_idx[:,1], halo_idx[:,2]].T
    T_halo     = Tbox[:,   halo_idx[:,1], halo_idx[:,2]].T
    v_pec_halo = Vbox[:,   halo_idx[:,1], halo_idx[:,2]].T

    # roll so each halo’s x-coordinate is at center
    shifts = i_center - halo_idx[:,0]
    n_HI_halo  = np.roll(n_HI_halo,  shifts, axis=1)
    T_halo     = np.roll(T_halo,     shifts, axis=1)
    v_pec_halo = np.roll(v_pec_halo, shifts, axis=1)

    # ── Save for next step ───────────────────────────────────────
    np.save(PATHS["n_HI_halo"],  n_HI_halo)
    np.save(PATHS["T_halo"],     T_halo)
    np.save(PATHS["v_pec_halo"], v_pec_halo)
    np.save(PATHS["halomass"],   halomass)
    np.save(PATHS["x_sim"],      x_sim)
    np.save(PATHS["z_grid"],     z)

    dt = time.time() - t0
    print(f"[gen] done in {dt:.1f}s")
    print(f"     n_HI_halo → {PATHS['n_HI_halo']}")
    print(f"     T_halo    → {PATHS['T_halo']}")
    print(f"     v_pec_halo→ {PATHS['v_pec_halo']}")
    print(f"     halomass  → {PATHS['halomass']}")
    print(f"     x_sim     → {PATHS['x_sim']}")
    print(f"     z_grid    → {PATHS['z_grid']}")

if __name__ == "__main__":
    run_gen()

