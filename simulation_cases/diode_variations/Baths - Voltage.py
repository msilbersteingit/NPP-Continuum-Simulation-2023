#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:31:49 2023

@author: max
"""

import sys

sys.path.append("/home/max/Research/Electrode Influence Letter/Python Simulation/")

import diode_bases.baths_concentration
import datetime
import numpy as np

voltage_variations = np.logspace(np.log10(0.05), np.log10(2), 10)

simulation = diode_bases.baths_concentration.Simulation()

datafile_group_equilibrate = f"Equilibration"


# filepath = "../../Python output databases/Concentration Boundary/Variation - Baths - Voltage.hdf5"
# filepath = "../../Python output databases/Concentration Boundary/Variation - Baths - Voltage - Higher Ionomer Length.hdf5"
# filepath = "../../Python output databases/Concentration Boundary/Variation - Baths - Voltage - Higher Ionomer Length - Higher D.hdf5"
filepath = "../../Python output databases/Concentration Boundary/Variation - Baths - Voltage - Even Longer Time.hdf5"

D = [5e-12, 5e-12]
period_voltage = 80
t_end = 80 * 4

simulation.D = D
simulation.period_voltage = period_voltage
simulation.t_end = t_end

# Equilibrate the system
npp_equilibrate = simulation.equilibrate(filepath_datafile=filepath, datafile_group=datafile_group_equilibrate)
state_equilibrated = npp_equilibrate.get_state()


# Iterate through the concentrations and run a simulation starting from equilibrium
for v in voltage_variations:
    simulation = diode_bases.baths_concentration.Simulation()

    simulation.D = D
    simulation.period_voltage = period_voltage
    simulation.t_end = t_end

    simulation.vpp = v

    datafile_group = f"v = {v:.3}"

    simulation.additional_state = {"voltage_peak_peak": v}

    # Run the voltage steps simulation
    simulation.run(state_equilibrated, filepath_datafile=filepath, datafile_group=datafile_group)
