#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 20:13:56 2023

@author: max
"""

import sys
import os.path

sys.path.append(os.path.abspath("../"))

import diode_bases.baths_concentration
import datetime
import numpy as np

now = datetime.datetime.now()
simulation = diode_bases.baths_concentration.Simulation()

# Make both space and time finer so that we get nice plots
simulation.spacing_coarse = 2e-7
simulation.t_step_max = 1e-2

filepath = "../../Python output databases/Concentration Boundary/Baseline.hdf5"

simulation.plotting = False
simulation.iterations_cumulative_plot = 5

# Equilibrate
simulation.equilibrate(filepath_datafile=filepath, datafile_group=f"{now} Equilibration")

# Run the voltage steps simulation
simulation.run(filepath_datafile=filepath, datafile_group=f"{now} Main")
