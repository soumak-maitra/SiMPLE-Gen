#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
assign.py
---------
Step 4 of the Lyman-α emitter pipeline:
Assign intrinsic Lyman-α luminosities (LLya) and
rest-frame equivalent widths (REW) to each halo.

REW is Monte Carlo sampled once per halo.
LLya is then computed from the same REW draw.
"""
import numpy as np

from .config import PATHS, Z_REDSHIFT

def P_REW(Muv, z):
    """
    Returns:
      REW_vals : 1D array of possible REW values [Å]
      p_REW    : corresponding probability weights
    """
    # set REW_min as function of Muv
    if Muv < -21.5:
        REW_min = -20.0
    elif Muv > -19.0:
        REW_min = 17.5
    else:
        REW_min = -20.0 + 6 * (Muv + 21.5)**2
    REW_max = 300.0

    # characteristic scale
    REW_c = 23 + 7 * (Muv + 21.9) + 6 * (z - 4)

    # normalization
    REW_vals = np.arange(REW_min, REW_max + 0.01, 0.01)
    p = np.exp(-REW_vals / REW_c)
    norm = (1 / REW_c) * (np.exp(-REW_min / REW_c) - np.exp(-REW_max / REW_c))**(-1)
    p_REW = norm * p

    return REW_vals, p_REW

def run_assign():
    # ── load Muv array from abundance matching ────────────────────────────
    Muv = np.load(PATHS["Muv_grid"])   # shape (N_halo,)
    z_sim = Z_REDSHIFT

    N = len(Muv)
    LLya = np.zeros(N, dtype=np.float64)
    REW  = np.zeros(N, dtype=np.float64)

    # conversion factor from REW and UV continuum to Lyα luminosity
    factor = (2.47e15 / 1215.67) * (1700 / 1215.67)**(1.7 - 2)

    # ── Monte Carlo sample each halo’s REW, then compute LLya ─────────────
    for i in range(N):
        # sample REW once
        rew_vals, p_rew = P_REW(Muv[i], z_sim)
        p_rew /= p_rew.sum()
        REW[i] = np.random.choice(rew_vals, p=p_rew).astype(np.float64)

        # compute LLya from the same sampled REW
        Luv_nu = 10.0**((51.6 - Muv[i]) / 2.5)
        LLya[i] = factor * Luv_nu * REW[i]

        if i % 10000 == 0:
            print(f"[assign] sampled {i}/{N} halos")

    # ── save outputs ──────────────────────────────────────────────────────
    np.save(PATHS["LLya_grid"], LLya)
    np.save(PATHS["REW_grid"],  REW)
    print(f"[assign] Saved LLya → {PATHS['LLya_grid']}")
    print(f"[assign] Saved REW  → {PATHS['REW_grid']}")

if __name__ == "__main__":
    run_assign()
