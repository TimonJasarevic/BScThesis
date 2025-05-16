import raspalib
import numpy as np

# Load the force field
ff = raspalib.ForceField("force_field.json")

# Define minimal MC move probabilities (still required)
mcmoves = raspalib.MCMoveProbabilities(translationProbability=1.0)

# Define CO2 component
co2 = raspalib.Component(
    componentId=0,
    forceField=ff,
    componentName="CO2",
    fileName="CO2.json",
    particleProbabilities=mcmoves,
)

# Load COF-1 framework
cof1 = raspalib.Framework(
    frameworkId=0,
    forceField=ff,
    componentName="COF-1",
    fileName="COF-1.cif",
    numberOfUnitCells=raspalib.int3(2, 2, 6),
    useChargesFrom=raspalib.Framework.UseChargesFrom.PseudoAtoms,
)

# Define the simulation system
system = raspalib.System(
    systemId=0,
    forceField=ff,
    simulationBox=None,
    externalTemperature=298.15,
    externalPressure=101300.0,
    heliumVoidFraction=0.0,
    framework=cof1,
    components=[co2],
    initialNumberOfMolecules=[20],
    numberOfBlocks=5,
    systemProbabilities=mcmoves,
    sampleMoviesEvery=None,
)

# Run Molecular Dynamics
nCycles = int(1e4)
md = raspalib.MolecularDynamics(
    numberOfCycles=nCycles,
    numberOfInitializationCycles=1000,
    systems=[system],
)

md.initialize()
md.equilibrate()

# # Machine learned model
# for i in range(nCycles):
#     md.runFirstHalfStep()
#     # positions, species = md.getPositionsSpecies()
#     # energy = model(positions, species)
#     # forces = -torch.autograd(model, positions)
#     # md.setForces(forces)
#     md.runSecondHalfStep()
# md.output()