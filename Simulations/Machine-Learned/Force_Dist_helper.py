import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

Z2SYM = {1: "H", 6: "C", 7: "N", 8: "O"} 

class ForceLogger:
    def __init__(self, fw_tree, fw_Z_np=None, co2_Z_all=None):
        self.fw_tree = fw_tree
        self.fw_Z_np = fw_Z_np
        self.co2_Z_all = co2_Z_all

        self.force_log = {
            "distance": [],          # nearest-framework distance  (Å)
            "force_classical": [],   # |F_classical|               (kJ mol-1 Å-1)
            "force_ani": []          # |F_ani|                     (kJ mol-1 Å-1)
        }

        self.log_by_pair = {
            "distance": [],
            "force": [],
            "source": [],
            "pair": []
        }

    def record_pseudomol(self,
                        lj_forces: np.ndarray,
                        ani_forces: np.ndarray,
                        coords: np.ndarray) -> None:
        """
        Log one value per molecule instead of per atom.
        For a single CO₂ this is a single point.
        """
        com = coords.mean(axis=0)
        dist_com_fw = self.fw_tree.query(com.reshape(1, 3), k=1)[0][0]
        lj_net_mag  = np.linalg.norm(lj_forces.sum(axis=0))
        ani_net_mag = np.linalg.norm(ani_forces.sum(axis=0))

        self.force_log["distance"].append(dist_com_fw)
        self.force_log["force_classical"].append(lj_net_mag)
        self.force_log["force_ani"].append(ani_net_mag)

    def record_atomwise_forces(self,
                                lj_f: np.ndarray,
                                ani_f: np.ndarray,
                                dist_fw: np.ndarray,
                                nearest_idx: np.ndarray) -> None:
        if self.fw_Z_np is None or self.co2_Z_all is None:
            raise ValueError("fw_Z_np and co2_Z_all must be provided to log atomwise forces by pair.")

        for i, d in enumerate(dist_fw):
            self_Z = self.co2_Z_all[i]
            fw_Z = self.fw_Z_np[nearest_idx[i]]
            pair = f"{Z2SYM[self_Z]}-{Z2SYM[fw_Z]}"

            self.log_by_pair["distance"].append(d)
            self.log_by_pair["force"].append(np.linalg.norm(lj_f[i]))
            self.log_by_pair["source"].append("Classical")
            self.log_by_pair["pair"].append(pair)

            self.log_by_pair["distance"].append(d)
            self.log_by_pair["force"].append(np.linalg.norm(ani_f[i]))
            self.log_by_pair["source"].append("ANI")
            self.log_by_pair["pair"].append(pair)

    def save_csv(self, filename: str = "force_vs_distance.csv") -> None:
        df = pd.DataFrame(self.force_log)
        df.to_csv(filename, index=False)
        print(f"Logged data written to {filename}")

    def plot_force_pseudomol(self, filename: str = "force_vs_distance.png") -> None:
        df = pd.DataFrame(self.force_log)

        fig, ax = plt.subplots()
        ax.scatter(df["distance"], df["force_classical"],
                s=8, alpha=0.4, marker="o", label="Classical")
        ax.scatter(df["distance"], df["force_ani"],
                s=8, alpha=0.4, marker="x", label="ANI-2x")

        ax.set_xlabel("distance to nearest framework atom (Å)")
        ax.set_ylabel("|F| (kJ mol$^{-1}$ Å$^{-1})$")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title("Classical & ANI-2x force magnitudes on CO₂ (COM) vs.\ndistance to nearest framework atom")

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
        print(f"Plot saved to {filename}")

    def plot_force_by_pair(self, filename: str = "force_vs_distance_compare.png") -> None:
        df = pd.DataFrame(self.log_by_pair)
        df.to_csv("force_vs_distance_compare.csv", index=False)

        fig, ax = plt.subplots()
        pairs = sorted(df["pair"].unique())
        colors = plt.cm.tab10(range(len(pairs)))
        cmap = dict(zip(pairs, colors))
        bins = np.arange(0, 5.3, 0.05)

        cl = df[df["source"] == "Classical"].copy()
        for pair, grp in cl.groupby("pair"):
            grp["bin"] = pd.cut(grp["distance"], bins)
            med = grp.groupby("bin")["force"].median()
            ax.plot([b.mid for b in med.index], med,
                    lw=1.8, ls="--", color=cmap[pair], label=f"Classical {pair}")

        ani = df[df["source"] == "ANI"].copy()
        for pair, grp in ani.groupby("pair"):
            grp["bin"] = pd.cut(grp["distance"], bins)
            med = grp.groupby("bin")["force"].median()
            ax.plot([b.mid for b in med.index], med,
                    lw=1.8, ls="-", color=cmap[pair], label=f"ANI-2x {pair}")

        ax.set_yscale("log")
        ax.set_xlabel("distance to nearest framework atom (Å)")
        ax.set_ylabel("|F| (kJ mol$^{-1}$ Å$^{-1}$)")
        ax.set_title("Classical & ANI-2x force magnitudes as \na function of nearest atom–framework pair distance")
        ax.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(filename, dpi=400)
        plt.show()
        print(f"Pairwise plot saved to {filename}")
