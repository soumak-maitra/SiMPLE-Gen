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
    Dbox     = np.load(RAW["density_cube"])     # comoving hydrogen number density [m^-3]
    Tbox     = np.load(RAW["temperature"])      # [K]
    Xbox     = np.load(RAW["ionization_cube"])  # ionized fraction x_HII
    Vbox     = np.load(RAW["velocity"])         # [comoving km/s]
    halo_cm  = np.load(RAW["halo_positions"])   # shape (N_halo, 3), [Mpc/h]
    halomass = np.load(RAW["halo_masses"])      # shape (N_halo,)

    # ── Mass cut ─────────────────────────────────────────────────
    mask = halomass >= 10.0**MH_CUT
    halo_cm  = halo_cm[mask]
    halomass = halomass[mask]

    # ── Build x & z grids ────────────────────────────────────────
    N = Dbox.shape[0]

    # central index
    i_center = N // 2

    # comoving coordinate [Mpc/h]
    x_sim = np.linspace(0, BOX_SIZE, N + 1)[:-1]

    # convert h^-1 cMpc -> cMpc with Astropy units
    x = x_sim * u.Mpc / cosmo.h

    # build z-grid via dz = dchi / (c / H)
    H_z = cosmo.H(Z_REDSHIFT)
    dx_dz = c / H_z

    z = np.zeros_like(x_sim)
    z[0] = Z_REDSHIFT - (x[i_center] - x[0]) / dx_dz

    for i in range(len(z) - 1):
        z[i + 1] = z[i] + (x[i + 1] - x[i]) / dx_dz

    # ── Compute neutral-H density sightlines ─────────────────────
    # Dbox is assumed to contain the comoving hydrogen number
    # density in m^-3. Convert it to physical number density in cm^-3:
    #
    # n_H,physical = n_H,comoving * (1 + z)^3
    #
    # and 1 m^3 = (100 cm)^3.
    nHbox = (
        Dbox
        * (1.0 + Z_REDSHIFT)**3
        / u.m.to(u.cm)**3
    )

    # Xbox is the ionized-hydrogen fraction x_HII.
    nHIbox = nHbox * (1.0 - Xbox)

    # Grid index of each halo.
    # For a periodic grid spanning [0, BOX_SIZE), positions map to
    # floor(position * N / BOX_SIZE). The modulo safely wraps any
    # position lying exactly on the periodic box boundary.
    halo_idx = (
        np.floor(halo_cm * N / BOX_SIZE).astype(np.int64) % N
    )

    # extract LOS at each halo (along x-axis)
    n_HI_halo  = nHIbox[:, halo_idx[:, 1], halo_idx[:, 2]].T
    T_halo     = Tbox[:,   halo_idx[:, 1], halo_idx[:, 2]].T
    v_pec_halo = Vbox[:,   halo_idx[:, 1], halo_idx[:, 2]].T

    # roll so each halo's x-coordinate is at center
    shifts = i_center - halo_idx[:, 0]

    def roll_rows(arr, shifts):
        n_col = arr.shape[1]
        cols = (np.arange(n_col)[None, :] - shifts[:, None]) % n_col
        return np.take_along_axis(arr, cols, axis=1)

    n_HI_halo  = roll_rows(n_HI_halo, shifts)
    T_halo     = roll_rows(T_halo, shifts)
    v_pec_halo = roll_rows(v_pec_halo, shifts)

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
