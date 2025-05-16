import matplotlib.pyplot as plt
import numpy as np

# Load the data from txt
filename = "/home/timon/Documents/Thesis/Simulations/Classical/raspa3/msd/msd_onsager_CO2_CO2.s0.txt"
data = np.loadtxt(filename, comments='#', usecols=(0, 1, 2, 3, 4))

time = data[:, 0]
msd_xyz = data[:, 1]
msd_x = data[:, 2]
msd_y = data[:, 3]
msd_z = data[:, 4]

# Plot
plt.figure(figsize=(8, 5))
plt.plot(time, msd_xyz, label='MSD total (xyz)', linewidth=2)
plt.plot(time, msd_x, label='MSD x')
plt.plot(time, msd_y, label='MSD y')
plt.plot(time, msd_z, label='MSD z')

plt.xlabel("Time [ps]")
plt.ylabel(r"MSD [$\mathrm{Å}^2$]")
plt.title("MSD over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("msd_plot.png", dpi=300)
plt.show()
