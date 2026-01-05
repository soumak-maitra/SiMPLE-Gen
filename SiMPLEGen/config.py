# config.py
import os

PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))

# ── INPUT FILES ───────────────────────────────────────────────
# Point this at wherever your raw .npy or .dat files live.
INPUT_DIR = "/home/soumak/My_Files/Test_Sim/seed1000_MTURN9.0_FESC-1.00_z7.14"

GRID_TAG = "017"

RAW = {
    "density_cube":     os.path.join(INPUT_DIR, f"density_grid_{GRID_TAG}.npy"),
    "temperature":      os.path.join(INPUT_DIR, f"temperature_grid_{GRID_TAG}.npy"),
    "ionization_cube":  os.path.join(INPUT_DIR, f"xHII_grid_{GRID_TAG}.npy"),
    "velocity":         os.path.join(INPUT_DIR, f"velocity_vx_grid_{GRID_TAG}.npy"),
    "halo_positions":   os.path.join(INPUT_DIR, f"halo_coords_grid_{GRID_TAG}.npy"),
    "halo_masses":      os.path.join(INPUT_DIR, f"halo_masses_grid_{GRID_TAG}.npy"),
}

# ── OUTPUT FILES ──────────────────────────────────────────────
DATA_DIR = os.path.join(PACKAGE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

PATHS = {
    "n_HI_halo":   os.path.join(DATA_DIR, "n_HI_halo.npy"),
    "T_halo":      os.path.join(DATA_DIR, "T_halo.npy"),
    "v_pec_halo":  os.path.join(DATA_DIR, "v_pec_halo.npy"),
    "halomass":    os.path.join(DATA_DIR, "halomass.npy"),
    "x_sim":       os.path.join(DATA_DIR, "x_sim.npy"),
    "z_grid":      os.path.join(DATA_DIR, "z_grid.npy"),
    "tau_halo":    os.path.join(DATA_DIR, "tau_halo.npy"),
    "Muv_grid":    os.path.join(DATA_DIR, "Muv_grid.npy"),
    "LLya_grid":    os.path.join(DATA_DIR, "LLya_grid.npy"),
    "REW_grid":    os.path.join(DATA_DIR, "REW_grid.npy"),
    "damping":    os.path.join(DATA_DIR, "damping.npy"),
}

# ── GLOBAL PARAMS ─────────────────────────────────────────────
Z_REDSHIFT = 7.14
MH_CUT     = 9.5
BOX_SIZE   = 80.0    # Mpc/h, used to build x_sim

