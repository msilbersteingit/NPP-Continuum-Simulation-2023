#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 17:15:11 2022

@author: max
"""
import sys
import os

import numpy as np
import diode_bases.microporous as dd

import datetime

# Set the changed parameters

widths = np.linspace(2e-6, 5e-5, 10)

dd.D = [5e-11, 5e-11]
# dd.D =  [5E-10, 5E-10]

# Iterate through the concentrations and run a whole equilibration and simulation per
for width in widths:
    # Compute the concentrations from the lower bath concentration
    dd.length_ionomer = width

    # filepath = "../../Python output databases/Microporous Boundary/Variation - Microporous - Ionomer Width - Higher D.hdf5"
    filepath = (
        "../../Python output databases/Microporous Boundary/Variation - Microporous - Ionomer Width - Medium D.hdf5"
    )
    # filepath = "../../Python output databases/Microporous Boundary/Variation - Microporous - Ionomer Width.hdf5"

    now = datetime.datetime.now()

    dd.data_file_group = f"{now} width = {width:.3}"
    dd.data_file_group_equilibrate = f"{now} width = {width:.3} Equilibration"

    # Construct the inputs and the additional swept state
    mesh, params = dd.construct_inputs()
    params["additional_state"] = {"ionomer_width": width}

    # Equilibrate the system
    npp_equilibrated = dd.equilibrate(mesh, params, filepath)

    # Get the equilibrated state and apply the voltage boundary conditions for the main simulation
    state_equilibrated = npp_equilibrated.get_state()

    # Set the OCV to the equilibrated voltage:
    dd.voc = state_equilibrated["potential_electrical"][-1]

    state_equilibrated["voltage_boundary"] = params["voltage_boundary"]
    state_equilibrated["additional_state"] = params["additional_state"]

    # Run the voltage steps simulation
    dd.run(mesh, state_equilibrated, filepath)
