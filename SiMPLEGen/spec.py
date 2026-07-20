#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spec.py
-------
Step 2: Compute the optical depth τ along each halo sightline.

Minimal necessary modifications:
- Double the LOS sampling by periodic linear interpolation.
- Retain only the foreground side of each centred sightline.
- Evaluate τ directly from -2000 to +2000 km/s in 5 km/s bins.
- Compute τ in halo chunks to control memory.
- Save the final array with shape (N_halo, N_velocity).

Interface preserved: run_spec() loads from PATHS and saves to
PATHS["tau_halo"].
"""

import time

import numpy as np
from scipy import special
from astropy import units as u
from astropy.constants import c, m_p, m_e, k_B
from astropy.cosmology import Planck18 as cosmo

from .config import PATHS


def tau(x_abs, z_abs, z_eval, n_HI, T, v_pec):
    """
    Compute optical depth at the requested evaluation redshifts.

    Parameters
    ----------
    x_abs : array
        Comoving absorber coordinates in Mpc/h.
    z_abs : array
        Redshifts of the absorber cells.
    z_eval : array
        Redshifts at which τ is evaluated.
    n_HI, T, v_pec : arrays
        Foreground neutral-hydrogen density, temperature and peculiar
        velocity with shape (N_halo, N_absorber).

    Returns
    -------
    tau_z : array
        Optical depth with shape (N_halo, N_eval).
    """
    I         = 4.45e-18
    x_cm      = u.Mpc.to(u.cm) * x_abs / cosmo.h
    lambda_lu = 1215.67e-8
    nu_lu     = c.to(u.cm / u.s).value / lambda_lu
    gamma_ul  = 6.262e8
    m_H       = (m_p + m_e).to(u.g).value
    c_cms     = c.to(u.cm / u.s).value

    b = np.sqrt(
        2.0 * k_B.to(u.erg / u.K).value * T / m_H
    )

    dx = np.mean(np.diff(x_cm))

    term1 = (
        c_cms
        * I
        / np.sqrt(np.pi)
        * dx
        * n_HI
        / (b * (1.0 + z_abs[None, :]))
    )

    z_abs_exp  = z_abs[None, :, None]
    z_eval_exp = z_eval[None, None, :]
    b_exp      = b[:, :, None]
    v_pec_exp  = v_pec[:, :, None]

    term2 = np.real(
        special.wofz(
            1j
            * gamma_ul
            * c_cms
            / (4.0 * np.pi * nu_lu * b_exp)
            + c_cms
            * (z_abs_exp - z_eval_exp)
            / (b_exp * (1.0 + z_eval_exp))
            + v_pec_exp * 1e5 / b_exp
        )
    )

    # Integrate over the absorber-cell axis.
    tau_z = np.sum(term1[:, :, None] * term2, axis=1)

    return tau_z


def run_spec():
    # ── Load sightline data ───────────────────────────────────────
    n_HI_halo  = np.load(PATHS["n_HI_halo"])
    T_halo     = np.load(PATHS["T_halo"])
    v_pec_halo = np.load(PATHS["v_pec_halo"])
    x_sim      = np.load(PATHS["x_sim"])
    z_grid     = np.load(PATHS["z_grid"])

    # A small chunk is required because the Voigt calculation creates
    # an array with shape (N_chunk, N_absorber, N_velocity).
    halo_chunk = int(PATHS.get("halo_chunk", 32))

    # Final velocity grid: -2000 to +2000 km/s in 5 km/s bins.
    vpos_halos_final = np.arange(
        -2000.0,
        2000.0 + 5.0,
        5.0
    )

    # Halo centre in the original centred sightline.
    i_center_native = len(x_sim) // 2
    z_centre = float(z_grid[i_center_native])

    # ── Double the LOS sampling ───────────────────────────────────
    N_native = len(x_sim)
    dx_native = np.mean(np.diff(x_sim))

    # Original samples occur at every second point. Intermediate
    # points are halfway between adjacent periodic cells.
    x_sim_new = (
        x_sim[0]
        + np.arange(2 * N_native) * dx_native / 2.0
    )

    def double_periodic(arr):
        """
        Double LOS sampling using periodic linear interpolation.
        """
        result = np.empty(
            (arr.shape[0], 2 * arr.shape[1]),
            dtype=arr.dtype
        )

        result[:, ::2] = arr
        result[:, 1::2] = 0.5 * (
            arr + np.roll(arr, -1, axis=1)
        )

        return result

    n_HI_halo  = double_periodic(n_HI_halo)
    T_halo     = double_periodic(T_halo)
    v_pec_halo = double_periodic(v_pec_halo)

    # The native halo-centre cell remains exactly represented after
    # doubling because native cells occupy the even indices.
    i_center = 2 * i_center_native

    # The original z-grid is linear in comoving LOS distance.
    dz_dx = np.mean(
        np.diff(z_grid) / np.diff(x_sim)
    )

    z_new = (
        z_centre
        + (x_sim_new - x_sim_new[i_center]) * dz_dx
    )

    # ── Keep only foreground absorber cells ───────────────────────
    # This retains the halo-centre cell and all cells on the
    # foreground side, while avoiding unnecessary calculations for
    # the zero-contribution background side.
    x_abs = x_sim_new[:i_center + 1]
    z_abs = z_new[:i_center + 1]

    n_HI_halo  = n_HI_halo[:, :i_center + 1]
    T_halo     = T_halo[:, :i_center + 1]
    v_pec_halo = v_pec_halo[:, :i_center + 1]

    # Convert the requested velocity grid directly to evaluation
    # redshifts around the halo redshift.
    c_kms = c.to(u.km / u.s).value

    z_eval = (
        z_centre
        + vpos_halos_final
        * (1.0 + z_centre)
        / c_kms
    )

    # ── Allocate final output ─────────────────────────────────────
    N_halo = n_HI_halo.shape[0]

    tau_halo_final = np.zeros(
        (N_halo, len(vpos_halos_final)),
        dtype=np.float32
    )

    print(
        f"[spec] halos={N_halo}, "
        f"LOS_Nx={N_native} → {2 * N_native}, "
        f"foreground_Nx={len(x_abs)}, "
        f"vgrid={len(vpos_halos_final)}"
    )
    print(
        f"[spec] z_centre={z_centre:.5f}, "
        f"halo_chunk={halo_chunk}"
    )

    # ── Chunk loop over halos ─────────────────────────────────────
    n_chunks = int(np.ceil(N_halo / halo_chunk))

    for sec in range(n_chunks):
        start = sec * halo_chunk
        end = min((sec + 1) * halo_chunk, N_halo)

        if start >= end:
            break

        t_chunk0 = time.time()

        print(
            f"[spec] chunk {sec + 1}/{n_chunks}: "
            f"halos {start}:{end}"
        )

        n_HI_sec = n_HI_halo[start:end]
        T_sec = T_halo[start:end]
        v_pec_sec = v_pec_halo[start:end]

        tau_halo_final[start:end] = tau(
            x_abs,
            z_abs,
            z_eval,
            n_HI_sec,
            T_sec,
            v_pec_sec
        ).astype(np.float32)

        print(
            f"[spec]    τ computed for {end - start} halos "
            f"in {time.time() - t_chunk0:.1f}s"
        )

    # ── Save results ──────────────────────────────────────────────
    np.save(PATHS["tau_halo"], tau_halo_final)

    print(f"[spec] τ saved → {PATHS['tau_halo']}")


if __name__ == "__main__":
    run_spec()
