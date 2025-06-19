import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# === SELECT PLOT MODE HERE ===
# 0 = Classical only
# 1 = Machine-Learned only
# 2 = Both (side-by-side)
PLOT_MODE = 2
SAVE_FIG = False
# =============================

base_paths = {
    "classical": Path("./Classical/msd"),
    "ml": Path("./Machine-Learned/msd")
}

msd_file = "msd_self_CO2.s0.txt"

def load_msd_data(mode):
    path = base_paths[mode] / msd_file
    try:
        data = np.loadtxt(path, comments='#', usecols=(0, 1, 2, 3, 4))
        return {
            "time": data[:, 0],
            "xyz": data[:, 1],
            "x": data[:, 2],
            "y": data[:, 3],
            "z": data[:, 4]
        }
    except Exception as e:
        print(f"Could not load {path}: {e}")
        return None

def plot_msd(ax, msd_data, title):
    ax.plot(msd_data["time"], msd_data["xyz"], label='MSD total (xyz)', linewidth=2)
    ax.plot(msd_data["time"], msd_data["x"], label='MSD x')
    ax.plot(msd_data["time"], msd_data["y"], label='MSD y')
    ax.plot(msd_data["time"], msd_data["z"], label='MSD z')
    ax.set_title(title)
    ax.set_xlabel("Time [ps]")
    ax.set_ylabel(r"MSD [$\mathrm{Å}^2$]")
    ax.legend()
    ax.grid(True)

# === PLOT ===
if PLOT_MODE in [0, 1]:
    mode = "classical" if PLOT_MODE == 0 else "ml"
    msd_data = load_msd_data(mode)
    if msd_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_msd(ax, msd_data, f"MSD — {'Classical' if mode == 'classical' else 'Machine-Learned'}")
        plt.tight_layout()
        if SAVE_FIG:
            plt.savefig(f"msd_plot_{mode}.png", dpi=300)
        plt.show()

elif PLOT_MODE == 2:
    msd_classical = load_msd_data("classical")
    msd_ml = load_msd_data("ml")

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    if msd_classical:
        plot_msd(axs[0], msd_classical, "MSD — Classical")
    if msd_ml:
        plot_msd(axs[1], msd_ml, "MSD — Machine-Learned")

    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig("msd_comparison.png", dpi=300)
    plt.show()
