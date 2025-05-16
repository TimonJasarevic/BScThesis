import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Dictionary of RDF labels and filenames
rdf_files = {
    "C–O (CO₂–CO₂)": "conventional_rdf/rdf_C_co2_O_co2.s0.txt",
    "O–O (CO₂–CO₂)": "conventional_rdf/rdf_O_co2_O_co2.s0.txt",
    "C–C (CO₂–CO₂)": "conventional_rdf/rdf_C_co2_C_co2.s0.txt",
    "C–B (CO₂–Framework)": "conventional_rdf/rdf_C_co2_B.s0.txt",
    "O–B (CO₂–Framework)": "conventional_rdf/rdf_O_co2_B.s0.txt",
}
abs_path = "/home/timon/Documents/Thesis/Simulations/Classical/raspa3/"


plt.figure(figsize=(8, 6))

# Plot each RDF with Gaussian smoothing
for label, filepath in rdf_files.items():
    data = np.loadtxt(abs_path + filepath)
    r = data[:, 0]
    g_r = gaussian_filter1d(data[:, 1], sigma=1.2)
    plt.plot(r, g_r, label=label)

# Axis labels and legend
plt.title("RDF's of important atom pairs")
plt.xlabel("Distance $r$ [Å]")
plt.ylabel("$g(r)$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rdf_combined.png", dpi=300)
plt.show()
