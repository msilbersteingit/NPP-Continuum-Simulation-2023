#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 2023

@author: max
"""

import diode_bases.baths_concentration as baths_concentration
import diode_bases.base as base
import numpy as np

class Simulation(baths_concentration.Simulation):
    reaction_surface_count = 2
    rate_constant = 1e-3

    k = 1e-7
    K = 1

    spacing_intermediate = 1.0 * k
    spacing_intermediate_width = 25 * spacing_intermediate


    spacing_coarse = 1e-5
    spacing_fine = k / 10
    spacing_fine_width = spacing_fine * 20
    use_debye_for_spacing = False

    @property
    def reaction_surface_parameters(self):
        reaction_surface_parameters = [
            {
                "stoichiometry": [-1, 0],
                "rate_exponents": [1, 0],
                "concentration_normalization": self.concentrations_baths[0],
                "rate_constant": self.rate_constant,
                "charge_transfer_direction": 1,
                "name": "Oxidation",
            },
            {
                "stoichiometry": [1, 0],
                "rate_exponents": [0, 0],
                "concentration_normalization": 1.0,
                "rate_constant": self.rate_constant,
                "charge_transfer_direction": -1,
                "name": "Reduction",
            },
        ]

        return reaction_surface_parameters

    # def compute_mesh_spacing(self, x):
    #     if self.use_debye_for_spacing:
    #         spacing_fine = self.L_d
    #         spacing_fine_width = spacing_fine * 20
    #     else:
    #         spacing_fine = self.spacing_fine
    #         spacing_fine_width = self.spacing_fine_width

    #     density_fine = 1 / spacing_fine
    #     density_course = 1 / self.spacing_coarse
    #     density_intermediate = 1 / self.spacing_intermediate

    #     if self.length_neutral_center == 0:
    #         density = np.maximum.reduce(
    #             [
    #                 density_course * np.ones(x.shape),
    #                 base.gaussian(x, density_fine, spacing_fine_width, 0),
    #                 base.gaussian(x, density_intermediate, self.spacing_intermediate_width, 0),
    #                 base.gaussian(
    #                     x,
    #                     density_fine * self.K,
    #                     spacing_fine_width / self.K,
    #                     self.length_bath - 2 * self.k / (self.K),
    #                 ),
    #                 base.gaussian(
    #                     x,
    #                     density_intermediate * self.K,
    #                     self.spacing_intermediate_width / self.K,
    #                     self.length_bath,
    #                 ),
    #                 base.gaussian(
    #                     x,
    #                     density_fine * 2,
    #                     spacing_fine_width,
    #                     self.coordinate_junction,
    #                 ),
    #                 base.gaussian(
    #                     x,
    #                     density_intermediate,
    #                     self.spacing_intermediate_width,
    #                     self.coordinate_junction,
    #                 ),
    #                 base.gaussian(
    #                     x,
    #                     density_fine * self.K,
    #                     spacing_fine_width / self.K,
    #                     self.L - self.length_bath + 2 * self.k / (self.K),
    #                 ),
    #                 base.gaussian(
    #                     x,
    #                     density_intermediate * self.K,
    #                     self.spacing_intermediate_width / self.K,
    #                     self.L - self.length_bath,
    #                 ),
    #                 base.gaussian(x, density_fine, spacing_fine_width, self.L),
    #                 base.gaussian(x, density_intermediate, self.spacing_intermediate_width, self.L),
    #             ]
    #         )

    #         spacing = 1 / density
    #     else:
    #         density = np.maximum.reduce(
    #             [
    #                 density_course * np.ones(x.shape),
    #                 base.gaussian(x, density_fine, spacing_fine_width, 0),
    #                 base.gaussian(
    #                     x,
    #                     density_fine * self.K,
    #                     spacing_fine_width / self.K,
    #                     self.length_bath - self.k / (2 * self.K),
    #                 ),
    #                 base.gaussian(
    #                     x,
    #                     density_intermediate,
    #                     self.spacing_intermediate_width,
    #                     self.length_bath,
    #                 ),
    #                 base.gaussian(
    #                     x,
    #                     density_fine,
    #                     spacing_fine_width,
    #                     self.coordinate_junction - self.length_neutral_center / 2,
    #                 ),
    #                 base.gaussian(
    #                     x,
    #                     density_intermediate,
    #                     self.spacing_intermediate_width,
    #                     self.coordinate_junction - self.length_neutral_center / 2,
    #                 ),
    #                 base.gaussian(
    #                     x,
    #                     density_fine,
    #                     spacing_fine_width,
    #                     self.coordinate_junction + self.length_neutral_center / 2,
    #                 ),
    #                 base.gaussian(
    #                     x,
    #                     density_intermediate,
    #                     self.spacing_intermediate_width,
    #                     self.coordinate_junction + self.length_neutral_center / 2,
    #                 ),
    #                 base.gaussian(
    #                     x,
    #                     density_fine * self.K,
    #                     spacing_fine_width / self.K,
    #                     self.L - self.length_bath + self.k / (2 * self.K),
    #                 ),
    #                 base.gaussian(
    #                     x,
    #                     density_intermediate,
    #                     self.spacing_intermediate_width,
    #                     self.L - self.length_bath,
    #                 ),
    #                 base.gaussian(x, density_fine, spacing_fine_width, self.L),
    #             ]
    #         )

    #         spacing = 1 / density

    #     return spacing

    # @property
    # def concentration_initial(self):
    #     # Note, only coded for two species diodes at the moment.
    #     concentrations_ionomer_lower = self.concentrations_ionomer_lower

    #     concentrations_ionomer_upper = [None] * self.ion_count
    #     for n in range(self.ion_count):
    #         concentrations_ionomer_upper[n] = (
    #             self.concentrations_ionomer[n] + self.concentrations_ionomer_lower[(n + 1) % 2]
    #         )

    #     def concentration(mesh, n):
    #         x = mesh.x
    #         concentration = (
    #             self.concentrations_baths[n]
    #             + base.sigmoid(
    #                 x,
    #                 concentrations_ionomer_lower[n] - self.concentrations_baths[n],
    #                 np.sign(self.valences[n]) * self.k / self.K,
    #                 self.L * (1 - n) - (1 - 2 * n) * self.length_bath,
    #             )
    #             + base.sigmoid(
    #                 x,
    #                 concentrations_ionomer_upper[n] - concentrations_ionomer_lower[n],
    #                 np.sign(self.valences[n]) * self.k,
    #                 self.L / 2 + np.sign(self.valences[n]) * self.length_neutral_center / 2,
    #             )
    #             + base.sigmoid(
    #                 x,
    #                 self.concentrations_baths[n] - concentrations_ionomer_upper[n],
    #                 np.sign(self.valences[n]) * self.k / self.K,
    #                 self.L * n + (1 - 2 * n) * self.length_bath,
    #             )
    #         )
    #         return concentration

    #     return concentration


    def _set_equilibrate_boundary_conditions(self):
        self._npp_equilibrate.set_boundary_condition(condition_type="voltage", value=0, where=self._mesh.facesLeft)
        self._npp_equilibrate.set_boundary_condition(condition_type="voltage", value=0, where=self._mesh.facesRight)
        self._npp_equilibrate.set_boundary_condition(
            condition_type="flux", value=[0, 0], where=self._mesh.exteriorFaces
        )

    def _set_boundary_conditions(self):
        self._npp.set_boundary_condition(condition_type="voltage", value=0, where=self._mesh.facesLeft)
        self._npp.set_boundary_condition(
            condition_type="voltage",
            value=self.voltage_boundary,
            where=self._mesh.facesRight,
        )
        self._npp.set_boundary_condition(condition_type="flux", value=[0, 0], where=self._mesh.exteriorFaces)
