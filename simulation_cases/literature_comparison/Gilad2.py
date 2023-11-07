#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 20:13:56 2023

@author: max
"""

import sys
import os.path

import diode_bases.baths_concentration
import diode_bases.base as base
import datetime
import numpy as np

k = 1e-7
K = 1

spacing_intermediate = 1.0 * k
spacing_intermediate_width = 25 * spacing_intermediate

class Gilad(diode_bases.baths_concentration.Simulation):
    t_end = 1500
    t_step_max = 1.0
    t_step_initial_equilibrate = 1e-4
    period_voltage = 500

    length_bath = 8e-6
    length_ionomer = 150e-6
    # spacing_fine = 1E-8
    # spacing_fine_width = 3E-7

    use_activity_coeffs = False
    activity_coeffs_ionomer_majority = [0.01, 0.01]
    activity_coeffs_ionomer_minority = [0.01, 0.01]
    activity_coeffs_baths = [0.88, 0.88]

    spacing_coarse = 1e-5
    spacing_fine = k / 10
    spacing_fine_width = spacing_fine * 20
    use_debye_for_spacing = False

    # ssimulation.length_neutral_center = 50e-6

    residual_max = 1e-8

    iterations_grow = 40

    # simulation.dielectric_ratio_ionomer = 1e3
    D_ionomer = [1.0e-10, 1.0e-10]

    concentrations_ionomer = [5000, 4200]
    concentrations_baths = [10, 10]

    vpp = 2
    dielectric_ratio_ionomer = 78

    def compute_mesh_spacing(self, x):
        if self.use_debye_for_spacing:
            spacing_fine = self.L_d
            spacing_fine_width = spacing_fine * 20
        else:
            spacing_fine = self.spacing_fine
            spacing_fine_width = self.spacing_fine_width

        density_fine = 1 / spacing_fine
        density_course = 1 / self.spacing_coarse
        density_intermediate = 1 / spacing_intermediate

        if self.length_neutral_center == 0:
            density = np.maximum.reduce(
                [
                    density_course * np.ones(x.shape),
                    base.gaussian(x, density_fine, spacing_fine_width, 0),
                    base.gaussian(x, density_intermediate, spacing_intermediate_width, 0),
                    base.gaussian(
                        x,
                        density_fine * K,
                        spacing_fine_width / K,
                        self.length_bath - 2 * k / (K),
                    ),
                    base.gaussian(
                        x,
                        density_intermediate * K,
                        spacing_intermediate_width / K,
                        self.length_bath,
                    ),
                    base.gaussian(
                        x,
                        density_fine * 2,
                        spacing_fine_width,
                        self.coordinate_junction,
                    ),
                    base.gaussian(
                        x,
                        density_intermediate,
                        spacing_intermediate_width,
                        self.coordinate_junction,
                    ),
                    base.gaussian(
                        x,
                        density_fine * K,
                        spacing_fine_width / K,
                        self.L - self.length_bath + 2 * k / (K),
                    ),
                    base.gaussian(
                        x,
                        density_intermediate * K,
                        spacing_intermediate_width / K,
                        self.L - self.length_bath,
                    ),
                    base.gaussian(x, density_fine, spacing_fine_width, self.L),
                    base.gaussian(x, density_intermediate, spacing_intermediate_width, self.L),
                ]
            )

            spacing = 1 / density
        else:
            density = np.maximum.reduce(
                [
                    density_course * np.ones(x.shape),
                    base.gaussian(x, density_fine, spacing_fine_width, 0),
                    base.gaussian(
                        x,
                        density_fine * K,
                        spacing_fine_width / K,
                        self.length_bath - k / (2 * K),
                    ),
                    base.gaussian(
                        x,
                        density_intermediate,
                        spacing_intermediate_width,
                        self.length_bath,
                    ),
                    base.gaussian(
                        x,
                        density_fine,
                        spacing_fine_width,
                        self.coordinate_junction - self.length_neutral_center / 2,
                    ),
                    base.gaussian(
                        x,
                        density_intermediate,
                        spacing_intermediate_width,
                        self.coordinate_junction - self.length_neutral_center / 2,
                    ),
                    base.gaussian(
                        x,
                        density_fine,
                        spacing_fine_width,
                        self.coordinate_junction + self.length_neutral_center / 2,
                    ),
                    base.gaussian(
                        x,
                        density_intermediate,
                        spacing_intermediate_width,
                        self.coordinate_junction + self.length_neutral_center / 2,
                    ),
                    base.gaussian(
                        x,
                        density_fine * K,
                        spacing_fine_width / K,
                        self.L - self.length_bath + k / (2 * K),
                    ),
                    base.gaussian(
                        x,
                        density_intermediate,
                        spacing_intermediate_width,
                        self.L - self.length_bath,
                    ),
                    base.gaussian(x, density_fine, spacing_fine_width, self.L),
                ]
            )

            spacing = 1 / density

        return spacing

    @property
    def concentration_initial(self):
        # Note, only coded for two species diodes at the moment.
        concentrations_ionomer_lower = self.concentrations_ionomer_lower

        concentrations_ionomer_upper = [None] * self.ion_count
        for n in range(self.ion_count):
            concentrations_ionomer_upper[n] = (
                self.concentrations_ionomer[n] + self.concentrations_ionomer_lower[(n + 1) % 2]
            )

        def concentration(mesh, n):
            x = mesh.x
            concentration = (
                self.concentrations_baths[n]
                + base.sigmoid(
                    x,
                    concentrations_ionomer_lower[n] - self.concentrations_baths[n],
                    np.sign(self.valences[n]) * k / K,
                    self.L * (1 - n) - (1 - 2 * n) * self.length_bath,
                )
                + base.sigmoid(
                    x,
                    concentrations_ionomer_upper[n] - concentrations_ionomer_lower[n],
                    np.sign(self.valences[n]) * k,
                    self.L / 2 + np.sign(self.valences[n]) * self.length_neutral_center / 2,
                )
                + base.sigmoid(
                    x,
                    self.concentrations_baths[n] - concentrations_ionomer_upper[n],
                    np.sign(self.valences[n]) * k / K,
                    self.L * n + (1 - 2 * n) * self.length_bath,
                )
            )
            return concentration

        return concentration

    @property
    def activity_coeffs(self):
        # Note, only coded for two species diodes at the moment.
        def activity_coeffs(mesh, n):
            x = mesh.x
            activity_coeffs = 1/(
                1/self.activity_coeffs_baths[n]
                + base.sigmoid(
                    x,
                    1/self.activity_coeffs_ionomer_minority[n] - 1/self.activity_coeffs_baths[n],
                    np.sign(self.valences[n]) * k / K,
                    self.L * (1 - n) - (1 - 2 * n) * self.length_bath,
                )
                + base.sigmoid(
                    x,
                    1/self.activity_coeffs_ionomer_majority[n] - 1/self.activity_coeffs_ionomer_minority[n],
                    np.sign(self.valences[n]) * k,
                    self.L / 2 + np.sign(self.valences[n]) * self.length_neutral_center / 2,
                )
                + base.sigmoid(
                    x,
                    1/self.activity_coeffs_baths[n] - 1/self.activity_coeffs_ionomer_majority[n],
                    np.sign(self.valences[n]) * k / K,
                    self.L * n + (1 - 2 * n) * self.length_bath,
                )
            )


            return activity_coeffs

        return activity_coeffs

    @property
    def voltage_boundary(self):
        def voltage(t):
            return base.square_wave(t, a=self.vpp / 2, T=self.period_voltage, k=0.0001)

        return voltage

    # @property
    # def voltage_boundary(self):
    #     def voltage(t):
    #         return base.square_wave(t+150, a=self.vpp/2, T = self.period_voltage, k = 0.01)

    #     return voltage


now = datetime.datetime.now()

simulation = Gilad()

simulation.plotting = True
simulation.iterations_cumulative_plot = 1

# simulation.residual_max = 1

filepath = "../../Python output databases/Literature Comparison/Gilad 45.hdf5"

# Construct the inputs and the additional swept state
# mesh,params = dd.construct_inputs(first_step = 1e-12, atol=1e-15, rtol = 1e-13, max_step=0.001)

now = datetime.datetime.now()

# Equilibrate
simulation.equilibrate(filepath_datafile=filepath, datafile_group=f"{now} Equilibration")

# Run the voltage steps simulation
simulation.run(filepath_datafile=filepath, datafile_group=f"{now} Main")
