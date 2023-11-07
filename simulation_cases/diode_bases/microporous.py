#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 17:15:11 2022

@author: Max Tepermeister
"""
import os

# os.environ['PETSC_OPTIONS'] = '-help'
os.environ["FIPY_SOLVERS"] = "scipy"
# os.environ['FIPY_VERBOSE_SOLVER'] = '1'

import fipy
import fipy.tools.numerix as np
import numpy
import scipy

import sys

# import matplotlib.pyplot as plt

import nppsolver

# %matplotlib qt
# %%


def sigmoid(x, a, k, x_0):
    return a / (1 + np.exp((x - x_0) / -k))


def gaussian(x, a, sigma, x_0):
    return a * np.exp(-((x - x_0) ** 2) / (2 * sigma**2))


def square_wave(t, a, T, k):
    w = np.pi / T
    return a * np.sin(w * t) / np.sqrt(k**2 * np.cos(w * t) ** 2 + np.sin(w * t) ** 2)


# %%
# TODO: Add units to variables so that we can error check and non-dimentionalize easier


D = [5e-12, 5e-12]
T = 293.15
R = 8.31446261815324
F = 96485.33212331

valences = [-1, 1, 0]

epsilon_0 = 8.8541e-12
dielectric_ratio = 78

concentrations_ionomer = [1, 1, 0]
concentrations_salt_impurity = [0.01, 0.01]

t_end = fipy.tools.dimensions.physicalField.PhysicalField(40e-0)

# Coordinate parameters
length_ionomer = 4e-6
length_microporous = 4e-5
spacing_fine = 1e-8
spacing_fine_width = 3e-7
spacing_coarse = 8e-7


vpp = 0.2
voc = -0.20

k1 = 10
k2 = 1

t_step_initial = 1.0e-9
t_step_max = 1.0e-1
t_step_max_equilibrate = 1e2
residual_max = 1e-9

t_wave_duration = 10
t_end = 40


t_step_increase = 1.2


ion_count = 2
bulk_reaction_count = 0
bulk_reaction_coeffs = [
    {
        "stoichiometry": [-1.0, -1.0, 1.0],
        "rate_exponents": [1, 1, 0],
        "rate_constant": k1,
    },
    {"stoichiometry": [1, 1, -1], "rate_exponents": [-1, -1, 0], "rate_constant": k2},
]
plotting = False
iterations_cumulative_plot = 3

use_debye_for_spacing = True

data_file_group_equilibrate = None
data_file_group = None

npp_equilibrate = None
npp = None

# %%


def construct_inputs(atol=1e-10, rtol=1e-13, first_step=1e-12):
    L = length_microporous * 2 + length_ionomer * 2

    permittivity = epsilon_0 * dielectric_ratio
    coordinate_junction = L / 2

    L_d = np.sqrt(permittivity * R * T / (2 * F**2 * concentrations_ionomer[0]))

    print(f"Debye length = {L_d}")

    k = L_d

    if use_debye_for_spacing:
        spacing_fine = L_d
        spacing_fine_width = L_d * 100

    def get_spacing(x):
        spacing = min(
            spacing_coarse,
            spacing_coarse - gaussian(x, spacing_coarse - spacing_fine, spacing_fine_width / 3, 0),
            spacing_coarse - gaussian(x, spacing_coarse - spacing_fine, spacing_fine_width, length_microporous),
            spacing_coarse
            - gaussian(
                x,
                spacing_coarse - spacing_fine,
                spacing_fine_width,
                coordinate_junction,
            ),
            spacing_coarse
            - gaussian(
                x,
                spacing_coarse - spacing_fine,
                spacing_fine_width,
                L - length_microporous,
            ),
            spacing_coarse - gaussian(x, spacing_coarse - spacing_fine, spacing_fine_width / 3, L),
        )
        return np.array(spacing)

    def integrator_done(t, y):
        return L - y[0]

    integrator_done.terminal = True

    solution = scipy.integrate.solve_ivp(
        lambda t, x: get_spacing(x),
        [0, np.inf],
        [0.0],
        dense_output=True,
        events=integrator_done,
        max_step=1,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
    )

    mesh_x = solution.sol(np.arange(0, np.around(solution.t[-1]) + 0.5))[0]
    mesh_x[-1] = L

    mesh_dx = np.diff(mesh_x)

    mesh = fipy.meshes.nonUniformGrid1D.NonUniformGrid1D(dx=mesh_dx)

    # %%

    def get_concentration_initial(mesh, n):
        x = mesh.x
        if n in [0, 1]:
            concentration = concentrations_salt_impurity[n] + sigmoid(
                x,
                concentrations_ionomer[n],
                np.sign(valences[n]) * k,
                coordinate_junction,
            )
        else:
            concentration = concentrations_ionomer[n]

        return concentration

    def voltage_boundary(t):
        return square_wave(t, a=vpp / 2, T=10, k=0.01) + voc

    def get_permittivity(mesh):
        x = mesh.x
        permittivity_calculated = (
            permittivity
            + sigmoid(x, permittivity * 1.28e4, -50 * k, length_microporous)
            + sigmoid(x, permittivity * 1.28e4, 50 * k, L - length_microporous)
        )
        return permittivity_calculated

    def get_diffusivity_bulk(mesh, n):
        x = mesh.x.arithmeticFaceValue
        diffusivity = (
            D[n]
            # + sigmoid(x, epsilon_0*1e3, -5*k, length_microporous)
            # + sigmoid(x, epsilon_0*1e3, 5*k, L-length_microporous)
        )
        return diffusivity

    params = {
        "potential_electrical": 0,
        "electromigration_coeff": 1 / (R * T),
        "permittivity": get_permittivity,
        "constant_faraday": F,
        "t_step": t_step_initial,
        "concentrations": get_concentration_initial,
        "valences": valences,
        "diffusivities": get_diffusivity_bulk,
        "charge_density_fixed": None,
        "bulk_reaction_coefficients": bulk_reaction_coeffs,
        "voltage_boundary": voltage_boundary,
    }
    return mesh, params


def equilibrate(mesh, params, filepath):
    global npp_equilibrate

    npp_equilibrate = nppsolver.NppSolver(
        mesh=mesh,
        ion_count=ion_count,
        state_initial=params,
        data_file_path=filepath,
        data_file_group=data_file_group_equilibrate,
        plotting=plotting,
    )

    npp_equilibrate.set_boundary_condition(condition_type="charge", value=0, where=mesh.facesRight)
    npp_equilibrate.set_boundary_condition(condition_type="voltage", value=0, where=mesh.facesLeft)
    npp_equilibrate.set_boundary_condition(condition_type="flux", value=[0, 0], where=mesh.exteriorFaces)

    npp_equilibrate.solve_until(
        t_end=t_end,
        t_step_max=t_step_max_equilibrate,
        residual_max=residual_max,
        t_step_increase=t_step_increase,
        save_interval=1e-3,
        plotting=plotting,
    )

    return npp_equilibrate


def run(mesh, params, filepath):
    global npp

    print(params["t_step"])
    npp = nppsolver.NppSolver(
        mesh=mesh,
        ion_count=ion_count,
        state_initial=params,
        data_file_path=filepath,
        data_file_group=data_file_group,
        plotting=plotting,
    )
    npp.set_boundary_condition(condition_type="voltage", value=0, where=mesh.facesLeft)
    npp.set_boundary_condition(
        condition_type="voltage",
        value=params["voltage_boundary"],
        where=mesh.facesRight,
    )
    npp.set_boundary_condition(condition_type="flux", value=[0, 0], where=mesh.exteriorFaces)

    npp.solve_until(
        t_end=t_end,
        t_step_max=t_step_max,
        residual_max=residual_max,
        t_step_increase=t_step_increase,
        save_interval=1e-4,
        plotting=plotting,
        iterations_cumulative_plot=iterations_cumulative_plot,
    )

    return npp
