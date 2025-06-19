from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# === SELECT PLOT MODE HERE ===
# 0 = Classical only
# 1 = Machine-Learned only
# 2 = Both (side-by-side)
PLOT_MODE = 1
SAVE_FIG = False
# =============================

base_paths = {
    "classical": Path("./Classical/conventional_rdf"),
    "ml": Path("./Machine-Learned/conventional_rdf")
}

rdf_files = {
    "C–C (CO₂–CO₂)":      "rdf_C_co2_C_co2.s0.txt",
    "C–C (CO₂–Framework)": "rdf_C_co2_C.s0.txt",
    "C–H (CO₂–Framework)": "rdf_C_co2_H.s0.txt",
    "C–N (CO₂–Framework)": "rdf_C_co2_N.s0.txt",
    "C–O (CO₂–CO₂)":       "rdf_C_co2_O_co2.s0.txt",
    "C–O (CO₂–Framework)": "rdf_C_co2_O.s0.txt",
    "O–C (CO₂–Framework)": "rdf_O_co2_C.s0.txt",
    "O–N (CO₂–Framework)": "rdf_O_co2_N.s0.txt",
    "O–O (CO₂–CO₂)":       "rdf_O_co2_O_co2.s0.txt",
    "O–O (CO₂–Framework)": "rdf_O_co2_O.s0.txt"
}

def load_rdf_data(mode):
    data_dict = {}
    for label, filename in rdf_files.items():
        path = base_paths[mode] / filename
        try:
            data = np.loadtxt(path)
            r = data[:, 0]
            g_r = data[:, 1]
            g_r = gaussian_filter1d(g_r, sigma=1.2)
            data_dict[label] = (r, g_r)
        except Exception as e:
            print(f"Could not load {path}: {e}")
    return data_dict

def plot_single_rdf(ax, data_dict, title):
    for label, (r, g_r) in data_dict.items():
        ax.plot(r, g_r, label=label)
    ax.set_title(title)
    ax.set_xlabel("Distance $r$ [Å]")
    ax.set_ylabel("$g(r)$")
    ax.legend()
    ax.grid(True)

# === PLOT ===
if PLOT_MODE in [0, 1]:
    mode = "classical" if PLOT_MODE == 0 else "ml"
    data = load_rdf_data(mode)
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_single_rdf(ax, data, f"RDFs — {'Classical' if mode == 'classical' else 'Machine-Learned'}")
    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig(f"rdf_{mode}.png", dpi=300)
    plt.show()

elif PLOT_MODE == 2:
    classical_data = load_rdf_data("classical")
    ml_data = load_rdf_data("ml")

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    plot_single_rdf(axs[0], classical_data, "RDFs — Classical")
    plot_single_rdf(axs[1], ml_data, "RDFs — Machine-Learned")

    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig("rdf_comparison.png", dpi=300)
    plt.show()
