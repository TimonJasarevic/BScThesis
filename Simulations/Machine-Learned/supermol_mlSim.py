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

# --- overwrite simulation.json with FWSim.json ---------------------------- #
root_dir = Path(__file__).resolve().parent
src = root_dir / "saved_simulations" / "FWSim.json"
dst = root_dir / "simulation.json"
shutil.copy(src, dst)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# Load ANI-2x model
model = torchani.models.ANI2x(periodic_table_index=True).to(device).eval()

# Load and initialize RASPA system
reader = raspalib.InputReader("simulation.json")
md = raspalib.MolecularDynamics(reader)
md.initialize()
md.equilibrate()

# --------------------------------------------------------------------------- #
#  Constants and Mappings                                                     #
# --------------------------------------------------------------------------- #
HARTREE_TO_KJMOL = 2625.49962 # Conversion factor: 1 Hartree = 2625.5 kJ/mol
FwIdxToSpecies  = {2: 6, 3: 8, 4: 1, 5: 7}
MolIdxToSpecies = {0: 6, 1: 8}

# Setup PBC
box_lengths = np.asarray(md.boxLengths(), dtype=float)
cell = torch.diag(torch.tensor(box_lengths, dtype=torch.float32, device=device))
pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)

# --------------------------------------------------------------------------- #
#  Force Function: ANI-2x (CO₂–framework + CO₂-CO₂)                           #
# --------------------------------------------------------------------------- #
def compute_ani_forces() -> np.ndarray:
    # Get framework data
    fw_pos_np = md.getPositions(framework=True)
    fw_types = md.getSpecies(framework=True)
    fw_Z_np = np.vectorize(FwIdxToSpecies.get)(fw_types)

    # Get CO₂ data
    co2_pos_np = md.getPositions(framework=False)
    co2_types = md.getSpecies(framework=False)
    co2_Z_np = np.vectorize(MolIdxToSpecies.get)(co2_types)

    # Convert to torch
    fw_tensor = torch.tensor(fw_pos_np, dtype=torch.float32, device=device)
    fw_Z = torch.tensor(fw_Z_np, dtype=torch.long, device=device)

    co2_tensor = torch.tensor(co2_pos_np, dtype=torch.float32, device=device, requires_grad=True)
    co2_Z = torch.tensor(co2_Z_np, dtype=torch.long, device=device)

    # Full ANI input
    coords = torch.cat([fw_tensor, co2_tensor], dim=0).unsqueeze(0)  # (1, N, 3)
    species = torch.cat([fw_Z, co2_Z], dim=0).unsqueeze(0)           # (1, N)

    # Energy and CO₂ forces
    energy = model((species, coords), cell=cell, pbc=pbc).energies
    (grad_co2,) = torch.autograd.grad(energy.sum(), [co2_tensor], create_graph=False)
    forces_co2 = (-grad_co2 * HARTREE_TO_KJMOL).detach().cpu().numpy()

    return forces_co2

# --------------------------------------------------------------------------- #
#  MD Step: Run one hybrid integration step                                   #
# --------------------------------------------------------------------------- #
def run_step():
    md.runFirstHalfStep()
    ani_forces = compute_ani_forces()
    md.setForces(ani_forces)
    md.runSecondHalfStep()

# --------------------------------------------------------------------------- #
#  Main Loop                                                                  #
# --------------------------------------------------------------------------- #
NumberOfCycles = 30000
print("Starting hybrid simulation...")

for step in range(NumberOfCycles):
    run_step()
    if step % 100 == 0:
        print(f"[Step {step}/{NumberOfCycles}]")

print("Writing output...")
md.output()
print("Simulation complete.")

# Clean-up
gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()
