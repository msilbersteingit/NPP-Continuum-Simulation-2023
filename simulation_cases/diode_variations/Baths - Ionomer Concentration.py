#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:31:49 2023

@author: max
"""

import sys

import diode_bases.baths_concentration as dd

import numpy as np
import datetime

dd.D = [5e-12, 5e-12]

concentrations_ionomer_variations = np.logspace(3, 0, 10)

dd.plotting = False

# Iterate through the concentrations and run a whole equilibration and simulation per
for c in concentrations_ionomer_variations:
    # Compute the concentrations from the lower bath concentration
    dd.concentrations_ionomer = [c + 0.1 * c, c + 0.1 * c]
    dd.concentrations_baths = [0.1 * c, 0.1 * c]

    filepath = (
        "/home/max/Research/Electrode Influence Letter/Python output"
        " databases/Concentration Boundary/Variation - Baths - Ionomer"
        " Concentration.hdf5"
    )

    now = datetime.datetime.now()

    dd.data_file_group = f"{now} c_ionomer = {c:.3}"
    dd.data_file_group_equilibrate = f"{now} c_ionomer = {c:.3} Equilibration"

    # Construct the inputs and the additional swept state
    mesh, params = dd.construct_inputs()
    params["additional_state"] = {"ionomer_concentration": c}

    # Equilibrate the system
    npp_equilibrated = dd.equilibrate(mesh, params, filepath)

    # Get the equilibrated state and apply the voltage boundary conditions for the main simulation
    state_equilibrated = npp_equilibrated.get_state()
    state_equilibrated["voltage_boundary"] = params["voltage_boundary"]
    state_equilibrated["additional_state"] = params["additional_state"]

    # Run the voltage steps simulation
    dd.run(mesh, state_equilibrated, filepath)
