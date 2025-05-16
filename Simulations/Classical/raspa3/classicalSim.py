import raspalib
import numpy as np

# Load the force field
ff = raspalib.ForceField("force_field.json")

# Define minimal MC move probabilities (still required)
mcmoves = raspalib.MCMoveProbabilities(translationProbability=1.0)

# Define CO2 component1
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
reader = raspalib.InputReader("simulation.json")
md = raspalib.MolecularDynamics(reader)
md.initialize()
md.equilibrate()

# Classical MD loop
nCycles = int(1e4)
for _ in range(nCycles):
    md.runFirstHalfStep()
    md.explicitGradients()
    md.runSecondHalfStep()
md.output()
