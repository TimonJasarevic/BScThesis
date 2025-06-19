# --------------------------------------------------------------------------- #
#  Environment Setup                                                          #
# --------------------------------------------------------------------------- #
import torch
import torchani
import numpy as np
import raspalib
import gc
from scipy.spatial import cKDTree  # Efficient periodic neighbour search

import shutil
from pathlib import Path

# --- overwrite simulation.json with FWSim.json ---------------------------- #
root_dir = Path(__file__).resolve().parent
src = root_dir / "saved_simulations" / "FWSim.json"
dst = root_dir / "simulation.json"
shutil.copy(src, dst)

# Determine compute device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# Load ANI-2x model
model = torchani.models.ANI2x(periodic_table_index=True).to(device).eval()

# Load simulation system
reader = raspalib.InputReader("simulation.json")
md = raspalib.MolecularDynamics(reader)
md.initialize()
md.equilibrate()

# --------------------------------------------------------------------------- #
#  Constants and Simulation Metadata                                          #
# --------------------------------------------------------------------------- #
HARTREE_TO_KJMOL = 2625.49962
ANI_CUTOFF = 5.2

FwIdxToSpecies  = {2: 6, 3: 8, 4: 1, 5: 7}
MolIdxToSpecies = {0: 6, 1: 8}

# Framework metadata
fw_pos_np = md.getPositions(framework=True)
fw_types = md.getSpecies(framework=True)
fw_Z_np = np.vectorize(FwIdxToSpecies.get)(fw_types)
fw_tree = cKDTree(fw_pos_np, boxsize=md.boxLengths())

fw_coords_t_all = torch.tensor(fw_pos_np, dtype=torch.float32, device=device)
fw_species_t_all = torch.tensor(fw_Z_np, dtype=torch.long, device=device)

# CO₂ metadata
co2_pos_np = md.getPositions()
co2_types_all = md.getSpecies()
co2_Z_all = np.vectorize(MolIdxToSpecies.get)(co2_types_all)

N_CO2_atoms = co2_Z_all.size
N_molecules = N_CO2_atoms // 3
mol_slices = [slice(3*i, 3*(i+1)) for i in range(N_molecules)]

# Box and PBC setup
box_len = np.asarray(md.boxLengths(), dtype=float)
cell_vectors = torch.diag(torch.tensor(box_len, dtype=torch.float32, device=device))
pbc_mask = torch.tensor([True, True, True], dtype=torch.bool, device=device)

# --------------------------------------------------------------------------- #
#  Machine-Learned Forces (CO₂–Framework only)                                #
# --------------------------------------------------------------------------- #

def ml_co2_framework_forces() -> np.ndarray:
    co2_pos_np = md.getPositions()
    forces_co2 = np.zeros_like(co2_pos_np)

    for sl in mol_slices:
        mol_coords_np = co2_pos_np[sl]
        idx_sets = fw_tree.query_ball_point(mol_coords_np, r=ANI_CUTOFF)
        if not any(idx_sets):
            continue

        unique_idx = np.unique(np.concatenate(idx_sets))
        fw_coords_sub = fw_coords_t_all[unique_idx]
        fw_species_sub = fw_species_t_all[unique_idx]

        mol_coords_t = torch.tensor(mol_coords_np, dtype=torch.float32,
                                    device=device, requires_grad=True)
        mol_species_t = torch.tensor(co2_Z_all[sl], dtype=torch.long, device=device)

        species = torch.cat([fw_species_sub, mol_species_t]).unsqueeze(0)
        coords = torch.cat([fw_coords_sub, mol_coords_t]).unsqueeze(0)

        energy = model((species, coords), cell=cell_vectors, pbc=pbc_mask).energies
        grad_mol, = torch.autograd.grad(energy, mol_coords_t)
        forces_mol = (-grad_mol * HARTREE_TO_KJMOL).detach().cpu().numpy()
        forces_co2[sl] = forces_mol

    return forces_co2

# --------------------------------------------------------------------------- #
#  Hybrid MD Step (cutoff blending logic)                                     #
# --------------------------------------------------------------------------- #

def run_step() -> None:
    md.runFirstHalfStep()
    md.explicitGradients()
    lj_forces = md.getForces()
    ani_forces = ml_co2_framework_forces()

    co2_pos_np = md.getPositions()
    dist_to_fw = fw_tree.query(co2_pos_np, k=1)[0]
    use_ani = dist_to_fw < ANI_CUTOFF

    total_forces = np.where(use_ani[:, None], ani_forces, lj_forces)
    md.setForces(total_forces)
    md.runSecondHalfStep()

# --------------------------------------------------------------------------- #
#  Main Simulation Loop                                                       #
# --------------------------------------------------------------------------- #

print("Starting hybrid simulation...")
NumberOfCycles = 30_000

for step in range(NumberOfCycles):
    run_step()
    if step % 100 == 0:
        print(f"[Hybrid] Step {step}/{NumberOfCycles}")

print("Writing output...")
md.output()
print("Simulation complete.")

gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()
