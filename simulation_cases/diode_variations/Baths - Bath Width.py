#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:31:49 2023

@author: max
"""

import sys
import os

import diode_bases.baths_concentration
import datetime
import numpy as np

#filepath = "../../Python output databases/Concentration Boundary/Variation - Baths - Bath Width - High V.hdf5"
filepath = "../../Python output databases/Concentration Boundary/Variation - Baths - Bath Width.hdf5"

bath_widths = np.linspace(1e-6, 5e-5, 10)
#bath_widths = [3.91e-5]

for width in bath_widths:
    simulation = diode_bases.baths_concentration.Simulation()

    simulation.length_bath = width

    now = datetime.datetime.now()
    datafile_group_equilibrate = f"{now} width = {width:.3} Equilibration"
    datafile_group = f"{now} width = {width:.3}"

    simulation.additional_state = {"bath_width": width}

    simulation.plotting = True
    simulation.iterations_cumulative_plot = 4
    #simulation.solver_type = 'fixed_point'

    #simulation.residual_max = 1e-5
    #simulation.vpp = 1

    # Equilibrate
    simulation.equilibrate(filepath_datafile=filepath, datafile_group=datafile_group_equilibrate)

    # Run the voltage steps simulation
    simulation.run(filepath_datafile=filepath, datafile_group=datafile_group)
