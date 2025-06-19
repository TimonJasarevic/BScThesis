import raspalib

# Define the simulation system
reader = raspalib.InputReader("simulation.json")
md = raspalib.MolecularDynamics(reader)
md.run()
