#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 2023

@author: max
"""
import diode_bases.base as base
import fipy.tools.numerix as np


class Simulation(base.Simulation):
    concentrations_baths = [0.1, 0.1]
    length_bath = 4e-6

    activity_coeffs_ionomer_majority = [1, 1]
    activity_coeffs_ionomer_minority = [1, 1]
    activity_coeffs_baths = [1, 1]


    # Define all the properties that are computed
    @property
    def L(self):
        return self.length_ionomer * 2 + self.length_bath * 2

    @property
    def concentrations_ionomer_lower(self):
        concentrations = [None] * self.ion_count
        for n in range(self.ion_count):
            concentrations[n] = (
                self.concentrations_baths[n] * self.concentrations_baths[n] / self.concentrations_ionomer[n]
            )

        return concentrations

    # Functions to compute initial conditions

    @property
    def concentration_initial(self):
        # Note, only coded for two species diodes at the moment.
        concentrations_ionomer_lower = self.concentrations_ionomer_lower

        concentrations_ionomer_upper = [None] * self.ion_count
        for n in range(self.ion_count):
            concentrations_ionomer_upper[n] = (
                self.concentrations_ionomer[n] + self.concentrations_ionomer_lower[(n + 1) % 2]
            )

        k = self.L_d

        def concentration(mesh, n):
            x = mesh.x
            concentration = (
                self.concentrations_baths[n]
                + base.sigmoid(
                    x,
                    concentrations_ionomer_lower[n] - self.concentrations_baths[n],
                    np.sign(self.valences[n]) * k,
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
                    np.sign(self.valences[n]) * k,
                    self.L * n + (1 - 2 * n) * self.length_bath,
                )
            )

            return concentration

        return concentration

    @property
    def activity_coeffs(self):
        # Note, only coded for two species diodes at the moment.

        self.activity_coeffs_ionomer_majority
        self.activity_coeffs_ionomer_minority
        self.activity_coeffs_baths

        k = self.L_d

        def activity_coeffs(mesh, n):
            x = mesh.x
            activity_coeffs = (
                self.activity_coeffs_baths[n]
                + base.sigmoid(
                    x,
                    self.activity_coeffs_ionomer_minority[n] - self.activity_coeffs_baths[n],
                    np.sign(self.valences[n]) * k,
                    self.L * (1 - n) - (1 - 2 * n) * self.length_bath,
                )
                + base.sigmoid(
                    x,
                    self.activity_coeffs_ionomer_majority[n] - self.activity_coeffs_ionomer_minority[n],
                    np.sign(self.valences[n]) * k,
                    self.L / 2 + np.sign(self.valences[n]) * self.length_neutral_center / 2,
                )
                + base.sigmoid(
                    x,
                    self.activity_coeffs_baths[n] - self.activity_coeffs_ionomer_majority[n],
                    np.sign(self.valences[n]) * k,
                    self.L * n + (1 - 2 * n) * self.length_bath,
                )
            )


            return activity_coeffs

        return activity_coeffs

    def compute_mesh_spacing(self, x):
        if self.use_debye_for_spacing:
            spacing_fine = self.L_d
            spacing_fine_width = spacing_fine * 10
        else:
            spacing_fine = self.spacing_fine
            spacing_fine_width = self.spacing_fine_width

        density_fine = 1 / spacing_fine
        density_course = 1 / self.spacing_coarse

        if self.length_neutral_center == 0:
            density = np.maximum.reduce(
                [
                    density_course * np.ones(x.shape),
                    base.gaussian(x, density_fine, spacing_fine_width / 3, 0),
                    base.gaussian(x, density_fine, spacing_fine_width, self.length_bath),
                    base.gaussian(x, density_fine, spacing_fine_width, self.coordinate_junction),
                    base.gaussian(x, density_fine, spacing_fine_width, self.L - self.length_bath),
                    base.gaussian(x, density_fine, spacing_fine_width / 3, self.L),
                ]
            )

            spacing = 1 / density
        else:
            spacing = min(
                self.spacing_coarse,
                self.spacing_coarse - base.cauchy(x, self.spacing_coarse - spacing_fine, spacing_fine_width / 3, 0),
                self.spacing_coarse
                - base.cauchy(
                    x,
                    self.spacing_coarse - spacing_fine,
                    spacing_fine_width,
                    self.length_bath,
                ),
                self.spacing_coarse
                - base.cauchy(
                    x,
                    self.spacing_coarse - spacing_fine,
                    spacing_fine_width,
                    self.coordinate_junction - self.length_neutral_center / 2,
                ),
                self.spacing_coarse
                - base.cauchy(
                    x,
                    self.spacing_coarse - spacing_fine,
                    spacing_fine_width,
                    self.coordinate_junction + self.length_neutral_center / 2,
                ),
                self.spacing_coarse
                - base.cauchy(
                    x,
                    self.spacing_coarse - spacing_fine,
                    spacing_fine_width,
                    self.L - self.length_bath,
                ),
                self.spacing_coarse
                - base.cauchy(
                    x,
                    self.spacing_coarse - spacing_fine,
                    spacing_fine_width / 3,
                    self.L,
                ),
            )

        return spacing

    def _set_equilibrate_boundary_conditions(self):
        self._npp_equilibrate.set_boundary_condition(condition_type="voltage", value=0, where=self._mesh.facesRight)
        self._npp_equilibrate.set_boundary_condition(condition_type="voltage", value=0, where=self._mesh.facesLeft)
        self._npp_equilibrate.set_boundary_condition(
            condition_type="concentration",
            value=[self.concentrations_baths[0], self.concentrations_baths[1]],
            where=self._mesh.facesLeft,
        )
        self._npp_equilibrate.set_boundary_condition(
            condition_type="concentration",
            value=[self.concentrations_baths[0], self.concentrations_baths[1]],
            where=self._mesh.facesRight,
        )

    def _set_boundary_conditions(self):
        self._npp.set_boundary_condition(condition_type="voltage", value=0.0, where=self._mesh.facesLeft)
        self._npp.set_boundary_condition(
            condition_type="voltage",
            value=self.voltage_boundary,
            where=self._mesh.facesRight,
        )
        self._npp.set_boundary_condition(
            condition_type="concentration",
            value=[self.concentrations_baths[0], self.concentrations_baths[1]],
            where=self._mesh.facesLeft,
        )
        self._npp.set_boundary_condition(
            condition_type="concentration",
            value=[self.concentrations_baths[0], self.concentrations_baths[1]],
            where=self._mesh.facesRight,
        )
