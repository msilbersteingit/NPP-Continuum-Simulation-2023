#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 23:54:21 2023

@author: max
"""

import sys
import os.path

import diode_bases.baths_concentration
import datetime
import numpy as np

filepaths = [
    "../../Python output databases/Concentration Boundary/Variation - Baths - Ionomer Width.hdf5",
    "../../Python output databases/Concentration Boundary/Variation - Baths - Ionomer Width - Medium D.hdf5",
    "../../Python output databases/Concentration Boundary/Variation - Baths - Ionomer Width - Higher D.hdf5",
]

Ds = [5e-12, 5e-11, 5e-10]

widths = np.linspace(1e-6, 5e-5, 20)
t_step_maxs = [1e-1, 1e-1, 1e-2]

for n in range(len(Ds)):
    filepath = filepaths[n]
    D = [Ds[n], Ds[n]]

    for width in widths:
        simulation = diode_bases.baths_concentration.Simulation()

        simulation.D_ionomer = D
        simulation.length_ionomer = width
        simulation.t_step_max = t_step_maxs[n]

        now = datetime.datetime.now()
        datafile_group_equilibrate = f"{now} width = {width:.3} Equilibration"
        now = datetime.datetime.now()
        datafile_group = f"{now} width = {width:.3}"

        simulation.additional_state = {"ionomer_width": width}

        # simulation.plotting = True

        # Equilibrate
        simulation.equilibrate(filepath_datafile=filepath, datafile_group=datafile_group_equilibrate)

        # Run the voltage steps simulation
        simulation.run(filepath_datafile=filepath, datafile_group=datafile_group)
