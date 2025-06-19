# --------------------------------------------------------------------------- #
#  Environment Setup                                                          #
# --------------------------------------------------------------------------- #
import torch
import torchani
import numpy as np
import raspalib
import gc

import shutil
from pathlib import Path

# --- overwrite simulation.json with MolSim.json ---------------------------- #
root_dir = Path(__file__).resolve().parent
src = root_dir / "saved_simulations" / "MolSim.json"
dst = root_dir / "simulation.json"
shutil.copy(src, dst)

# Determine compute device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# Load ANI-2x neural network model
model = torchani.models.ANI2x(periodic_table_index=True).to(device).eval()

# Initialize RASPA system
reader = raspalib.InputReader("simulation.json")
md = raspalib.MolecularDynamics(reader)
md.initialize()
md.equilibrate()

# --------------------------------------------------------------------------- #
#  Constants and Atom Mappings                                                #
# --------------------------------------------------------------------------- #
HARTREE_TO_KJMOL = 2625.49962          # 1 Hartree = 2625.5 kJ/mol
ANI_CUTOFF       = 5.2                 # ANI-2x cutoff distance (Å)

MolIdxToSpecies = {0: 6, 1: 8}         # CO₂ pseudo-types → atomic numbers

# CO₂ atom types and species
co2_types_all = md.getSpecies()
co2_Z_all     = np.vectorize(MolIdxToSpecies.get)(co2_types_all)

# Define molecular slices (CO₂ has 3 atoms)
N_CO2_atoms = co2_Z_all.size
N_mol       = N_CO2_atoms // 3
mol_slices  = [slice(3*i, 3*(i+1)) for i in range(N_mol)]

# Simulation box
box_len     = np.asarray(md.boxLengths(), dtype=float)
cell_vectors = torch.diag(torch.tensor(box_len, dtype=torch.float32, device=device))
pbc_mask     = torch.tensor([True, True, True], dtype=torch.bool, device=device)

# --------------------------------------------------------------------------- #
#  Machine-Learned CO₂–CO₂ Force Evaluation (ANI-2x)                          #
# --------------------------------------------------------------------------- #
def ml_co2_co2_forces() -> np.ndarray:
    """
    Compute ANI-2x forces from CO₂–CO₂ interactions only.
    """
    pos_np = md.getPositions()  # Å
    species_t = torch.tensor(co2_Z_all, dtype=torch.long, device=device)
    coords_t  = torch.tensor(pos_np, dtype=torch.float32, device=device, requires_grad=True)

    energy = model((species_t.unsqueeze(0), coords_t.unsqueeze(0))).energies
    grad, = torch.autograd.grad(energy, coords_t)

    forces = (-grad * HARTREE_TO_KJMOL).detach().cpu().numpy()
    return forces

# --------------------------------------------------------------------------- #
#  Hybrid MD Step: ANI (CO₂–CO₂)                                              #
# --------------------------------------------------------------------------- #
def run_step() -> None:
    """
    One integration step with hybrid forces:
    - Velocity Verlet propagation.
    - Use ANI-2x for CO₂–CO₂.
    """
    md.runFirstHalfStep()
    ani_forces = ml_co2_co2_forces()
    md.setForces(ani_forces)
    md.runSecondHalfStep()

# --------------------------------------------------------------------------- #
#  Main Production Loop                                                       #
# --------------------------------------------------------------------------- #
NumberOfCycles = 30_000
print("Starting hybrid production…")

for step in range(NumberOfCycles):
    run_step()
    if step % 100 == 0:
        print(f"[Hybrid] Step {step}/{NumberOfCycles}")

print("Writing output…")
md.output()
print("Simulation complete.")

# Clean up
gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()
