#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spec.py
-------
Step 2: Compute the optical depth τ(z) along each halo sightline.

MODIFIED (minimal-necessary) to reproduce the behavior of your batch script:
- Double the LOS sampling via linear interpolation (2×N_x).
- Zero out n_HI above the central LOS index (same masking logic).
- Compute τ in halo chunks to control memory.
- Interpolate τ(LOS) onto a fixed velocity grid vpos_halos_final (km/s) around z_centre.
- Save final array shaped (N_halo, N_v).

Interface/usage preserved: run_spec() loads from PATHS and saves to PATHS["tau_halo"].
"""

import numpy as np
from scipy import special
from scipy.interpolate import interp1d
from astropy import units as u
from astropy.constants import c, m_p, m_e, k_B
from astropy.cosmology import Planck18 as cosmo
import time

from .config import PATHS


def tau(x, z, n_HI, T, v_pec):
    """
    Optical depth τ as a function of LOS pixel index (returned array is (N_halo, N_x)).
    Mirrors your original expression and the batch script behavior.
    """
    I         = 4.45e-18
    x_cm      = u.Mpc.to(u.cm) * x / cosmo.h   # match your batch code convention
    lambda_lu = 1215.67e-8
    nu_lu     = c.to(u.cm/u.s).value / lambda_lu
    gamma_ul  = 6.262e8
    m_H       = (m_p + m_e).to(u.g).value

    b    = np.sqrt(2.0 * k_B.to(u.erg/u.K).value * T / m_H)
    dx   = np.mean(x_cm[1:] - x_cm[:-1])
    term1 = (c.to(u.cm/u.s).value * I / np.sqrt(np.pi)) * dx * (n_HI / (b * (1.0 + z)))

    z_exp = z[None, :, None]
    b_exp = b[:, :, None]

    term2 = np.real(
        special.wofz(
            1j * gamma_ul * c.to(u.cm/u.s).value
              / (4.0 * np.pi * nu_lu * b_exp)
            + c.to(u.cm/u.s).value
              * (z_exp - z[None, None, :])
              / (b_exp * (1.0 + z[None, None, :]))
            + v_pec[:, :, None] * 1e5 / b_exp
        )
    )

    # integrate over the second axis
    tau_z = np.sum(term1[:, :, None] * term2, axis=1)
    return tau_z


def run_spec():
    # ── Load sightline data ───────────────────────────────────────
    n_HI_halo  = np.load(PATHS["n_HI_halo"])   # (N_halo, N_x)
    T_halo     = np.load(PATHS["T_halo"])      # (N_halo, N_x)
    v_pec_halo = np.load(PATHS["v_pec_halo"])  # (N_halo, N_x)
    x_sim      = np.load(PATHS["x_sim"])       # (N_x,)
    z_grid     = np.load(PATHS["z_grid"])      # (N_x,)

    # ---- Settings mirroring your batch script ----
    halo_chunk = int(PATHS.get("halo_chunk", 32768)) if isinstance(PATHS, dict) else 32768

    # If you have a saved velocity grid, use it; otherwise default to your batch grid.
    if "vpos_grid" in PATHS:
        vpos_halos_final = np.load(PATHS["vpos_grid"]).astype(float)
    else:
        vpos_halos_final = np.arange(-500.0, 501.0, 10.0)  # km/s

    # z_centre: if explicitly provided as a file, load it; else use LOS midpoint (robust default)
    if "z_centre" in PATHS:
        z_centre = float(np.load(PATHS["z_centre"]))
    else:
        z_centre = float(z_grid[len(z_grid)//2])

    # ── Interpolation doubling (2× LOS sampling) ───────────────────
    x_sim_new = np.linspace(np.min(x_sim), np.max(x_sim), 2 * len(x_sim))
    z_new     = np.linspace(np.min(z_grid), np.max(z_grid), 2 * len(z_grid))

    n_HI_halo = interp1d(x_sim, n_HI_halo, kind="linear", axis=1, bounds_error=False, fill_value="extrapolate")(x_sim_new)
    T_halo    = interp1d(x_sim, T_halo,    kind="linear", axis=1, bounds_error=False, fill_value="extrapolate")(x_sim_new)
    v_pec_halo= interp1d(x_sim, v_pec_halo,kind="linear", axis=1, bounds_error=False, fill_value="extrapolate")(x_sim_new)

    # central index (same convention as your batch code)
    i_center = n_HI_halo.shape[1] // 2 - 1

    # ── Allocate final output (N_halo, N_v) ────────────────────────
    N_halo = n_HI_halo.shape[0]
    F_halo_final = np.zeros((N_halo, len(vpos_halos_final)), dtype=np.float32)

    print(f"[spec] halos={N_halo}, LOS_Nx={len(x_sim)} → {len(x_sim_new)}, vgrid={len(vpos_halos_final)}")
    print(f"[spec] z_centre={z_centre:.5f}, halo_chunk={halo_chunk}")

    # ── Chunk loop over halos ──────────────────────────────────────
    n_chunks = int(np.ceil(N_halo / halo_chunk))
    for sec in range(n_chunks):
        start = sec * halo_chunk
        end   = min((sec + 1) * halo_chunk, N_halo)
        if start >= end:
            break

        t_chunk0 = time.time()
        print(f"[spec] chunk {sec+1}/{n_chunks}: halos {start}:{end}")

        n_HI_sec  = n_HI_halo[start:end, :].copy()   # copy because we mask in-place
        T_sec     = T_halo[start:end, :]
        v_pec_sec = v_pec_halo[start:end, :]

        # zero out above central index (exact mask behavior)
        I_idx = (i_center * np.ones(end - start)).astype(np.int32)
        mask = np.arange(n_HI_sec.shape[1]) > I_idx[:, None]
        n_HI_sec[mask] = 0.0

        # compute τ along LOS grid (returns (chunk_halos, N_x_new))
        F_halo_los = tau(x_sim_new, z_new, n_HI_sec, T_sec, v_pec_sec)

        # convert z grid to velocity grid for each halo (same formula)
        z_rep = np.repeat(z_new[np.newaxis, :], repeats=(end - start), axis=0)
        vpos_halos = c.to(u.km/u.s).value * (z_rep - z_centre) / (1.0 + z_centre)

        # interpolate τ(v) onto vpos_halos_final for each halo
        F_interp = np.array(
            [
                np.interp(
                    vpos_halos_final,
                    vpos_halos[i, :],
                    F_halo_los[i, :],
                    left=np.nan,
                    right=np.nan
                )
                for i in range(end - start)
            ],
            dtype=np.float32
        )

        F_halo_final[start:end, :] = F_interp
        print(f"[spec]    τ computed+interp for {end-start} halos in {(time.time()-t_chunk0):.1f}s")

    # ── Save results ──────────────────────────────────────────────
    np.save(PATHS["tau_halo"], F_halo_final)
    print(f"[spec] τ saved → {PATHS['tau_halo']}")


if __name__ == "__main__":
    run_spec()

