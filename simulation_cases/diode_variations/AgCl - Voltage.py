#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:31:49 2023

@author: max
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:31:49 2023

@author: max
"""

import sys
import os.path

sys.path.append(os.path.abspath("../"))

import diode_bases.anion_reaction
import datetime
import numpy as np


simulation = diode_bases.anion_reaction.Simulation()

datafile_group_equilibrate = f"Equilibration"

#filepath = "../../Python output databases/Reaction Boundary/Variation - AgCl - Voltage 15.hdf5"
# filepath = "../../Python output databases/Reaction Boundary/Variation - AgCl - Rate Constant - Higher Ionomer Width - Higher D.hdf5"
filepath = None

D = [5e-12, 5e-12]

simulation.D = D
simulation.plotting = True
simulation.iterations_cumulative_plot = 10

## Equilibrate without the reactions
simulation.reaction_surface_count = 0

# Equilibrate the system
npp_equilibrate = simulation.equilibrate(filepath_datafile=filepath, datafile_group=datafile_group_equilibrate)
state_equilibrated = npp_equilibrate.get_state()
mesh = npp_equilibrate.mesh

voltage_variations = np.linspace(0.05, 0.1, 10)

# %%
# Iterate through the concentrations and run a simulation starting from equilibrium
for i, V_pp in enumerate(voltage_variations):
    # if i < 5:
    #     continue

    simulation = diode_bases.anion_reaction.Simulation()

    simulation.D = D
    simulation.rate_constant = 1e-7
    simulation.t_end = 10
    #simulation.t_end_equilibrate = 50

    simulation.vpp = V_pp

    simulation.plotting = True
    simulation.iterations_cumulative_plot = 1

    datafile_group = f"V_pp = {V_pp:.3}"

    simulation.additional_state = {"voltage_peak_peak": V_pp}

    # Use the equilibrated state for the DOF intial values
    params = simulation.get_params()
    params["potential_electrical"] = state_equilibrated["potential_electrical"]
    params["concentrations"] = state_equilibrated["concentrations"]

    # Run the voltage steps simulation
    simulation.run(params, filepath_datafile=filepath, datafile_group=datafile_group)
