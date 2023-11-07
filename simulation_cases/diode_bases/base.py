#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:51:14 2023

@author: Max Tepermeister
"""
import os
import sys

# os.environ['PETSC_OPTIONS'] = '-help'
os.environ["FIPY_SOLVERS"] = "scipy"
# os.environ['FIPY_VERBOSE_SOLVER'] = '1'

import fipy
import fipy.tools.numerix as np
import numpy
import scipy

import nppsolver


def sigmoid(x, a, k, x_0):
    return a / (1 + np.exp((x - x_0) / -k))


def gaussian(x, a, sigma, x_0):
    return a * np.exp(-((x - x_0) ** 2) / (2 * sigma**2))


def cauchy(x, a, sigma, x_0):
    return a / (1 + ((x - x_0) / sigma) ** 2)


def square_wave(t, a, T, k):
    w = np.pi / T
    return a * np.sin(w * t) / np.sqrt(k**2 * np.cos(w * t) ** 2 + np.sin(w * t) ** 2)


# self.concentrations_baths = [0.1, 0.1]
# concentrations_ionomer_lower = [0, 0]
# length_bath = 4E-6


class Simulation:
    # Basic physical constants
    T = 293.15
    R = 8.31446261815324
    F = 96485.33212331
    epsilon_0 = 8.8541e-12

    # Material properties
    dielectric_ratio_ionomer = 78
    D_ionomer = [5e-12, 5e-12]

    # Geometric parameters
    ion_count = 2
    length_ionomer = 4e-6
    spacing_fine = 1e-8
    spacing_fine_width = 3e-7
    spacing_coarse = 8e-7
    length_neutral_center = 0

    # Initial Conditions
    concentrations_ionomer = [1.0, 1.0]
    valences = [-1, 1]
    t_end = 40
    t_end_equilibrate = 500

    # Chemical reactions
    reaction_bulk_count = 0
    reaction_bulk_parameters = []
    reaction_surface_count = 0
    reaction_surface_parameters = []

    # Activities
    use_activity_coeffs = False

    # Time parameters
    period_voltage = 10
    vpp = 0.2
    voc = 0

    # Solving parameters
    t_step_initial_equilibrate = (1e-8,)
    t_step_max_equilibrate = (1e2,)
    t_step_increase_equilibrate = (1.2,)
    t_end_equilibrate = (500,)

    t_step_initial = (1e-8,)
    t_step_max = (1e-1,)
    t_step_increase = (1.2,)
    t_end = (60,)

    # Other options
    plotting = False
    iterations_cumulative_plot = 20
    save_interval = 1e-3
    residual_max = 5e-10

    iterations_max = 60
    iterations_grow = 30

    use_debye_for_spacing = True
    solver_type = "rootfinding"

    iterations_cumulative_plot = 20

    additional_state = {}

    def __init__(self):
        # Internal objects
        self._npp = None
        self._npp_equilibrate = None
        self._mesh = None

    # Define all the properties that are computed

    @property
    def L(self):
        return self.length_ionomer * 2

    @property
    def L_d(self):
        return np.sqrt(self.permittivity * self.R * self.T / (2 * self.F**2 * self.concentrations_ionomer[0]))

    @property
    def permittivity_ionomer(self):
        return self.epsilon_0 * self.dielectric_ratio_ionomer

    @property
    def coordinate_junction(self):
        return self.L / 2

    @property
    def activity_coeffs(self):
        return [1]*self.ion_count

    # Functions to compute initial conditions

    @property
    def concentration_initial(self):
        raise NotImplementedError()

    @property
    def voltage_boundary(self):
        def voltage(t):
            return square_wave(t, a=self.vpp / 2, T=self.period_voltage, k=0.01)

        return voltage

    @property
    def permittivity(self):
        return self.permittivity_ionomer

    @property
    def diffusivities(self):
        return self.D_ionomer

    areas = 1

    def compute_mesh_spacing(self, x):
        pass

    def _construct_mesh(self, atol=1e-10, rtol=1e-13, first_step=1e-12, max_step=1e-2):
        L = self.L

        def integrator_done(t, y):
            return L - y[0]

        integrator_done.terminal = True

        solution = scipy.integrate.solve_ivp(
            lambda t, x: self.compute_mesh_spacing(x),
            [0, np.inf],
            [0.0],
            dense_output=True,
            events=integrator_done,
            max_step=max_step,
            rtol=rtol,
            atol=atol,
            first_step=first_step,
            method="BDF",
            vectorized=True,
        )

        mesh_x = solution.sol(np.arange(0, np.around(solution.t[-1]) + 0.5))[0]
        mesh_x[-1] = L
        
        #Make the outermost element extremely thin
        mesh_x[-2] = L-1e-10
        mesh_x[1] = 1e-10
        
        mesh_dx = np.diff(mesh_x)

        mesh = fipy.meshes.nonUniformGrid1D.NonUniformGrid1D(dx=mesh_dx)

        # Overwrite the cell areas to make a quasi-nonuniform mesh
        #mesh_areas = self.compute_mesh_areas(mesh)
        #mesh._faceAreas = mesh_areas
        #mesh._cellVolumes = mesh._calcCellVolumes()
        #mesh._cellAreas = mesh._calcCellAreas()

        self._mesh = mesh

    def get_params(self):
         params = {
             'potential_electrical': 0,
             'electromigration_coeff': 1 / (self.R * self.T),
             'permittivity': self.permittivity,
             'areas': self.areas,
             'constant_faraday': self.F,
             't_step': 0,
             'concentrations': self.concentration_initial,
             'activity_coeffs': self.activity_coeffs,
             'valences': self.valences,
             'diffusivities': self.diffusivities,
             'charge_density_fixed': None,
             'reaction_bulk_parameters': self.reaction_bulk_parameters,
             'reaction_surface_parameters': self.reaction_surface_parameters,
             'additional_state': self.additional_state,
         }

         return params

    def _set_equilibrate_boundary_conditions(self):
        raise NotImplementedError()

    def _set_boundary_conditions(self):
        raise NotImplementedError()

    def equilibrate(
        self,
        filepath_datafile=None,
        params=None,
        datafile_group=None,
        continue_run_if_possible=True,
    ):
        if params is None:
            params = self.get_params()

        if self._mesh is None:
            self._construct_mesh()

        params["t_step"] = self.t_step_initial_equilibrate

        if self._npp is None or not continue_run_if_possible:
            self._npp_equilibrate = nppsolver.NppSolver(
                mesh=self._mesh,
                ion_count=self.ion_count,
                reaction_bulk_count=self.reaction_bulk_count,
                reaction_surface_count=self.reaction_surface_count,
                use_activity_coeffs = self.use_activity_coeffs,
                state_initial=params,
                data_file_path=filepath_datafile,
                data_file_group=datafile_group,
                plotting=self.plotting,
            )

        self._set_equilibrate_boundary_conditions()

        self._npp_equilibrate.solve_until(
            t_end=self.t_end_equilibrate,
            t_step_max=self.t_step_max_equilibrate,
            residual_max=self.residual_max,
            t_step_increase=self.t_step_increase_equilibrate,
            iterations_max=self.iterations_max,
            iterations_grow=self.iterations_grow,
            save_interval=self.save_interval,
            plotting=self.plotting,
            iterations_cumulative_plot=self.iterations_cumulative_plot,
            solver_type=self.solver_type,
        )

        return self._npp_equilibrate

    def run(
        self,
        params=None,
        filepath_datafile=None,
        datafile_group=None,
        continue_run_if_possible=True,
    ):
        if params is None:
            params = self._npp_equilibrate.get_state()

        if self._mesh is None:
            self._construct_mesh()

        params["t_step"] = self.t_step_initial

        if self._npp is None or not continue_run_if_possible:
            self._npp = nppsolver.NppSolver(
                mesh=self._mesh,
                ion_count=self.ion_count,
                reaction_bulk_count=self.reaction_bulk_count,
                reaction_surface_count=self.reaction_surface_count,
                use_activity_coeffs = self.use_activity_coeffs,
                state_initial=params,
                data_file_path=filepath_datafile,
                data_file_group=datafile_group,
                plotting=self.plotting,
            )

            self._set_boundary_conditions()

        self._npp.solve_until(
            t_end=self.t_end,
            t_step_max=self.t_step_max,
            residual_max=self.residual_max,
            t_step_increase=self.t_step_increase,
            iterations_max=self.iterations_max,
            iterations_grow=self.iterations_grow,
            save_interval=self.save_interval,
            plotting=self.plotting,
            iterations_cumulative_plot=self.iterations_cumulative_plot,
            solver_type=self.solver_type,
        )

        return self._npp
