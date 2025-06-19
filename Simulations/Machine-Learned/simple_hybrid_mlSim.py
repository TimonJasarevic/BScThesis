# --------------------------------------------------------------------------- #
#  Environment Setup                                                          #
# --------------------------------------------------------------------------- #
import torch
import torchani
import numpy as np
import raspalib
import gc
from scipy.spatial import cKDTree  # For efficient periodic neighbour search

# Determine compute device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# Load pre-trained ANI-2x neural network model
model = torchani.models.ANI2x(periodic_table_index=True).to(device).eval()

# Read simulation parameters and initialise simulation system
reader = raspalib.InputReader("simulation.json")
md = raspalib.MolecularDynamics(reader)
md.initialize()
md.equilibrate()

# --------------------------------------------------------------------------- #
#  Constants and Mappings                                                     #
# --------------------------------------------------------------------------- #
HARTREE_TO_KJMOL = 2625.49962  # Conversion factor: 1 Hartree = 2625.5 kJ/mol
ANI_CUTOFF = 5.2               # ANI-2x radial cutoff distance (Å)

# Atom-type index mappings for framework and CO₂ molecules
FwIdxToSpecies  = {2: 6, 3: 8, 4: 1, 5: 7}  # force field atoms → atomic number
MolIdxToSpecies = {0: 6, 1: 8}              # CO₂: C = 6, O = 8

# Framework atom positions and species
fw_pos0   = md.getPositions(framework=True)
fw_types  = md.getSpecies(framework=True)
fw_Z_np   = np.vectorize(FwIdxToSpecies.get)(fw_types)

# CO₂ atom species
co2_types_all = md.getSpecies()
co2_Z_all     = np.vectorize(MolIdxToSpecies.get)(co2_types_all)

# Simulation box dimensions and PBC setup
box_len     = np.asarray(md.boxLengths(), dtype=float)
cell_vectors = torch.diag(torch.tensor(box_len, dtype=torch.float32, device=device))
pbc_mask    = torch.tensor([True, True, True], dtype=torch.bool, device=device)

# Precompute framework KD-tree for fast local environment queries
fw_tree = cKDTree(fw_pos0, boxsize=box_len)
fw_coords_t_all  = torch.tensor(fw_pos0, dtype=torch.float32, device=device)
fw_species_t_all = torch.tensor(fw_Z_np,  dtype=torch.long,   device=device)


# Define molecular slices (each CO₂ has 3 atoms)
N_CO2_atoms = co2_Z_all.size
N_molecules = N_CO2_atoms // 3
mol_slices  = [slice(3*i, 3*(i+1)) for i in range(N_molecules)]


# --------------------------------------------------------------------------- #
#  Machine-Learned CO₂–Framework Force Evaluation (ANI-2x)                   #
# --------------------------------------------------------------------------- #

def ml_co2_framework_forces() -> np.ndarray:
    """
    Compute forces on CO₂ atoms due to their local interaction with the framework,
    using the ANI-2x machine-learned potential.

    For each CO₂ molecule:
    1. Identify framework atoms within 5.2 Å (periodic) of any CO₂ atom.
    2. Construct species and coordinates tensor for ANI input.
    3. Compute the energy and extract forces on the CO₂ atoms.
    """
    co2_pos_np: np.ndarray = md.getPositions()
    forces_co2 = np.zeros_like(co2_pos_np)

    for sl in mol_slices:
        mol_coords_np = co2_pos_np[sl]

        # Step 1: Periodic neighbour search
        idx_sets   = fw_tree.query_ball_point(mol_coords_np, r=ANI_CUTOFF)
        if not any(idx_sets):
            continue

        unique_idx = np.unique(np.concatenate(idx_sets))
        fw_coords_sub  = fw_coords_t_all[unique_idx]
        fw_species_sub = fw_species_t_all[unique_idx]

        # Step 2: Prepare input tensors for ANI
        fw_coords_sub.requires_grad_(False)
        mol_coords_t = torch.tensor(mol_coords_np, dtype=torch.float32,
                                    device=device, requires_grad=True)
        mol_species_t = torch.tensor(co2_Z_all[sl], dtype=torch.long, device=device)

        species = torch.cat([fw_species_sub, mol_species_t]).unsqueeze(0)
        coords  = torch.cat([fw_coords_sub, mol_coords_t]).unsqueeze(0)

        # Step 3: Evaluate energy and compute gradient (forces)
        energy = model((species, coords), cell=cell_vectors, pbc=pbc_mask).energies
        grad_mol, = torch.autograd.grad(energy, mol_coords_t)   # shape (3, 3)
        forces_mol = (-grad_mol * HARTREE_TO_KJMOL).detach().cpu().numpy()
        forces_co2[sl] = forces_mol

    return forces_co2


# --------------------------------------------------------------------------- #
#  Hybrid MD Step: ANI (CO₂–Framework) + LJ (CO₂–CO₂)                         #
# --------------------------------------------------------------------------- #

def run_step() -> None:
    """
    Perform one time integration step using hybrid force evaluation:
    - Velocity Verlet for propagation.
    - Classical Lennard-Jones forces for CO₂–CO₂.
    - ANI-2x machine-learned forces for CO₂–framework.
    """
    md.runFirstHalfStep()

    md.explicitGradients(includeFrameworkMolecule=False) # Classical LJ forces: CO₂–CO₂ only
    lj_forces = md.getForces()

    ani_forces = ml_co2_framework_forces()
    md.setForces(ani_forces + lj_forces)  # Combine ANI and LJ forces
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
