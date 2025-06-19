from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# === SELECT PLOT MODE HERE ===
# 0 = Classical only
# 1 = Machine-Learned only
# 2 = Both (side-by-side)
PLOT_MODE = 2
SAVE_FIG = False
# =============================

base_paths = {
    "classical": Path("./Classical/vacf"),
    "ml": Path("./Machine-Learned/vacf")
}

vacf_file = "vacf_self_CO2.s0.txt"

def load_vacf_data(mode):
    path = base_paths[mode] / vacf_file
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

def plot_vacf(ax, vacf_data, title):
    ax.plot(vacf_data["time"], vacf_data["xyz"], label='VACF total (xyz)', linewidth=2)
    ax.plot(vacf_data["time"], vacf_data["x"], label='VACF x')
    ax.plot(vacf_data["time"], vacf_data["y"], label='VACF y')
    ax.plot(vacf_data["time"], vacf_data["z"], label='VACF z')
    ax.set_title(title)
    ax.set_xlabel("Time [ps]")
    ax.set_ylabel(r"VACF [$\mathrm{\AA}^2$]")
    ax.legend()
    ax.grid(True)

# === PLOT ===
if PLOT_MODE in [0, 1]:
    mode = "classical" if PLOT_MODE == 0 else "ml"
    vacf_data = load_vacf_data(mode)
    if vacf_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_vacf(ax, vacf_data, f"VACF — {'Classical' if mode == 'classical' else 'Machine-Learned'}")
        plt.tight_layout()
        if SAVE_FIG:
            plt.savefig(f"vacf_plot_{mode}.png", dpi=300)
        plt.show()

elif PLOT_MODE == 2:
    vacf_classical = load_vacf_data("classical")
    vacf_ml = load_vacf_data("ml")

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    if vacf_classical:
        plot_vacf(axs[0], vacf_classical, "VACF — Classical")
    if vacf_ml:
        plot_vacf(axs[1], vacf_ml, "VACF — Machine-Learned")

    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig("vacf_comparison.png", dpi=300)
    plt.show()
