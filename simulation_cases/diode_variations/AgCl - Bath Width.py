#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 23:54:21 2023

@author: max
"""

import sys
import os.path

import diode_bases.anion_reaction
import datetime
import numpy as np

filepath = "../../Python output databases/Reaction Boundary/Variation - AgCl - Bath Width.hdf5"

#bath_widths = np.linspace(1e-6, 5e-5, 10)
bath_widths = [3.91e-5]

for width in bath_widths:
    simulation = diode_bases.anion_reaction.Simulation()

    simulation.length_bath = width

    datafile_group_equilibrate = f"width = {width:.3} Equilibration"
    datafile_group = f"width = {width:.3}"

    simulation.additional_state = {"bath_width": width}

    simulation.plotting = False
    simulation.iterations_cumulative_plot = 1
    #simulation.solver_type = 'fixed_point'

    #simulation.residual_max = 1e-9

    # Equilibrate
    simulation.equilibrate(filepath_datafile=filepath, datafile_group=datafile_group_equilibrate)

    # Run the voltage steps simulation
    simulation.run(filepath_datafile=filepath, datafile_group=datafile_group)
