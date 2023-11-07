#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:31:49 2023

@author: max
"""

import sys
import os.path

import diode_bases.baths_concentration
import datetime
import numpy as np

#filepath = "../../Python output databases/Concentration Boundary/Variation - Baths - Bath Concentration.hdf5"
filepath = "../../Python output databases/Concentration Boundary/Variation - Baths - Bath Concentration - Medium D.hdf5"


concentrations_baths_variations = np.linspace(0.05, 0.5, 10)

# Iterate through the concentrations and run a whole equilibration and simulation per
for n,c in enumerate (concentrations_baths_variations):
    simulation = diode_bases.baths_concentration.Simulation()

    simulation.concentrations_baths = [c, c]
    simulation.D_ionomer = [5e-11, 5e-11]

    now = datetime.datetime.now()
    datafile_group_equilibrate = f"{n:02} c_bath = {c:.3} Equilibration"
    now = datetime.datetime.now()
    datafile_group = f"{n:02} c_bath = {c:.3}"

    simulation.additional_state = {"bath_concentration": c}

    # simulation.plotting = True

    # Equilibrate
    simulation.equilibrate(filepath_datafile=filepath, datafile_group=datafile_group_equilibrate)

    # Run the voltage steps simulation
    simulation.run(filepath_datafile=filepath, datafile_group=datafile_group)
