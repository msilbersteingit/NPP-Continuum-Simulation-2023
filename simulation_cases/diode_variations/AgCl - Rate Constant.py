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

import diode_bases.anion_reaction
import datetime
import numpy as np


simulation = diode_bases.anion_reaction.Simulation()

datafile_group_equilibrate = f"Equilibration"

filepath = "../../Python output databases/Reaction Boundary/Variation - AgCl - Rate Constant.hdf5"
# filepath = "../../Python output databases/Reaction Boundary/Variation - AgCl - Rate Constant - Higher Ionomer Width - Higher D.hdf5"

D = [5e-12, 5e-12]

simulation.D = D

# Equilibrate the system
npp_equilibrate = simulation.equilibrate(filepath_datafile=filepath, datafile_group=datafile_group_equilibrate)
state_equilibrated = npp_equilibrate.get_state()
mesh = npp_equilibrate.mesh

rate_constant_variations = np.logspace(-2, -7, 10)

# %%
# Iterate through the concentrations and run a simulation starting from equilibrium
for k in rate_constant_variations:
    simulation = diode_bases.anion_reaction.Simulation()

    simulation.D = D
    simulation.rate_constant = k

    datafile_group = f"k = {k:.3}"

    simulation.additional_state = {"rate_constant": k}

    # Use the equilibrated state for the DOF intial values
    params = simulation.get_params()
    params["potential_electrical"] = state_equilibrated["potential_electrical"]
    params["concentrations"] = state_equilibrated["concentrations"]

    # Run the voltage steps simulation
    simulation.run(state_equilibrated, filepath_datafile=filepath, datafile_group=datafile_group)
