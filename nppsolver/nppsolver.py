# -*- coding: utf-8 -*-
"""
Author: Max Tepermeister
"""

import fipy
from functools import reduce
import fipy.tools.numerix as np

import fipy.solvers.scipy
import fipy.solvers.petsc

import scipy.linalg
import scipy.sparse
import scipy.optimize

import numpy
import tables

# import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.raise_window'] = False

import os

import datetime

import dotmap

import contextlib

# TODO:  Add type annotations to the class


class NppSolver:
    def __init__(
        self,
        mesh,
        ion_count,
        reaction_bulk_count=0,
        reaction_surface_count=0,
        use_activity_coeffs=False,
        state_initial=None,
        data_file_path=None,
        data_file_group=None,
        plotting=True,
    ):

        self.mesh = mesh

        self.ion_count = ion_count
        self.reaction_bulk_count = reaction_bulk_count
        self.reaction_surface_count = reaction_surface_count

        self.use_activity_coeffs = use_activity_coeffs

        self._instantiate_state_variables()
        self._formulate_equations()

        if state_initial is not None:
            self.set_state(state_initial)

        if plotting:
            self._setup_plot()


        if data_file_path is not None:
            self._init_datafile(data_file_path, data_file_group)
            self.write_data_enabled = True
            self.flush_interval = 10
        else:
            self.write_data_enabled = False

    def _init_datafile(self, filepath, group_name=None):
        self.datafile = dotmap.DotMap()

        self.datafile.filepath = filepath
        self.datafile.type_timestep = numpy.dtype(
            [
                ('voltage', np.float64, self.mesh.shape),
                ('concentrations', np.float64, (self.ion_count, *self.mesh.shape)),
                ('current', np.float64),
                ('time', np.float64),
                ('residual', np.float64),
                ('sweeps', np.uint16),
                ('timestamp', 'S23')
            ]
        )

        now = datetime.datetime.now()

        if group_name is not None:
            self.datafile.group_name = group_name
        else:
            self.datafile.group_name = str(now)

        file = tables.open_file(filepath, mode='a', title='Nernst Planck Poisson Simulation Data')

        self.datafile.file = file

        if '/simulations' not in file.root:
            file.create_group('/', 'simulations', "Simulations")

        #Delete existing simulations with the same name
        if self.datafile.group_name in file.root.simulations:
            file.remove_node("/simulations/", name = self.datafile.group_name)

        group = file.create_group(file.root.simulations, self.datafile.group_name, f"Simulation at {now}")
        self.datafile.group = group
        self.datafile.timestep_table = file.create_table(
            group, 'timesteps', self.datafile.type_timestep, title="Timestep data"
        )

        file.create_array(group, 'ion_count', self.ion_count, title="Number of charged mobile species")
        file.create_array(
            group,
            'valences',
            np.array([valence.value for valence in self.valences]),
            title="Charge of each mobile species",
        )
        file.create_array(group, 'electromigration_coeff', self.electromigration_coeff.value, title="1/RT")
        file.create_array(group, 'permittivity', self.permittivity.value, title="Dielectric Permittivity")
        file.create_array(group, 'constant_faraday', self.constant_faraday.value, title="Faraday's constant")
        file.create_array(
            group,
            'charge_density_fixed',
            self.charge_density_fixed.value,
            title="Fixed charge density",
        )
        file.create_array(
            group,
            'diffusivities',
            np.array([diffusivity.value for diffusivity in self.diffusivities]),
            title="Diffusivities for each mobile species",
        )
        file.create_array(
            group,
            'mesh_centers',
            self.mesh.cellCenters.value,
            title="Coordinates for each cell center",
        )
        file.create_array(
            group,
            'mesh_faces',
            self.mesh.faceCenters.value,
            title="Coordinates for each cell face",
        )

        if self.additional_state is not None:
            for state_name in self.additional_state:
                file.create_array(group, state_name, self.additional_state[state_name])

        file.close()

    def open_datafile(self):
        if self.write_data_enabled:
            # If the datafile isn't already open, open it and get references to the relevant tables
            if not self.datafile.file.isopen:
                file = tables.open_file(self.datafile.filepath, mode='a', title='Nernst Planck Poisson Simulation Data')
                self.datafile.file = file
                self.datafile.group = file.get_node('/simulations', self.datafile.group_name)
                self.datafile.timestep_table = file.get_node(self.datafile.group, 'timesteps')
                return file
            else:
                return self.datafile
        else:
            # If there's no data file to write to, return a null context
            return contextlib.nullcontext()

    def load_datafile_state(self, datafile_path, datafile_group, timestep_index):
        """Loads the solver state from an existing outputfile"""

        # Note that this doesn't load the mesh or other states required to construct the internal equations.

        datafile = tables.open_file(datafile_path, mode="r")

        simulation = datafile.get_node("/simulations", datafile_group)

        data = simulation.timesteps[timestep_index]
        data_old = simulation.timesteps[timestep_index - 1]

        for n in range(self.ion_count):
            self.concentrations[n].setValue(data['concentrations'][n])
            self.concentrations[n].old.setValue(data['concentrations'][n])
            self.concentrations[n].old2.setValue(data_old['concentrations'][n])

        self.potential_electrical.setValue(data['voltage'])
        self.potential_electrical.old.setValue(data['voltage'])

        self.t_step_old.setValue(value=data['time'] - data_old['time'])
        self.time.setValue(data['time'])

        datafile.close()

        pass

    def set_state(self, state):
        potential_electrical_value = (
            state['potential_electrical'](self.mesh)
            if callable(state['potential_electrical'])
            else state['potential_electrical']
        )
        electromigration_coeff = (
            state['electromigration_coeff'](self.mesh)
            if callable(state['electromigration_coeff'])
            else state['electromigration_coeff']
        )
        permittivity = state['permittivity'](self.mesh) if callable(state['permittivity']) else state['permittivity']
        constant_faraday = (
            state['constant_faraday']() if callable(state['constant_faraday']) else state['constant_faraday']
        )

        if self.reaction_bulk_count > 0:
            print(f"Initializing {self.reaction_bulk_count} Bulk Reactions")
            reaction_bulk_parameters = (
                state['reaction_bulk_parameters']()
                if callable(state['reaction_bulk_parameters'])
                else state['reaction_bulk_parameters']
            )

        if self.reaction_surface_count > 0:
            print(f"Initializing {self.reaction_surface_count} Surface Reactions")
            reaction_surface_parameters = (
                state['reaction_surface_parameters']()
                if callable(state['reaction_surface_parameters'])
                else state['reaction_surface_parameters']
            )

        self.potential_electrical.setValue(value=potential_electrical_value)
        self.electromigration_coeff.setValue(electromigration_coeff)
        self.permittivity.setValue(permittivity)
        self.constant_faraday.setValue(constant_faraday)


        for k in range(self.reaction_bulk_count):
            self.reaction_bulk_parameters[k]['stoichiometry'].setValue(reaction_bulk_parameters[k]['stoichiometry'])
            self.reaction_bulk_parameters[k]['rate_exponents'].setValue(reaction_bulk_parameters[k]['rate_exponents'])
            self.reaction_bulk_parameters[k]['rate_constant'].setValue(reaction_bulk_parameters[k]['rate_constant'])

        for k in range(self.reaction_surface_count):
            self.reaction_surface_parameters[k]['stoichiometry'].setValue(
                reaction_surface_parameters[k]['stoichiometry']
            )
            self.reaction_surface_parameters[k]['rate_exponents'].setValue(
                reaction_surface_parameters[k]['rate_exponents']
            )
            self.reaction_surface_parameters[k]['rate_constant'].setValue(
                reaction_surface_parameters[k]['rate_constant']
            )
            self.reaction_surface_parameters[k]['charge_transfer_direction'].setValue(
                reaction_surface_parameters[k]['charge_transfer_direction']
            )
            self.reaction_surface_parameters[k]['concentration_normalization'].setValue(
                reaction_surface_parameters[k]['concentration_normalization']
            )

        self.t_step.setValue(state['t_step'])

        for n in range(self.ion_count):
            concentration = (
                state['concentrations'](self.mesh, n)
                if callable(state['concentrations'])
                else state['concentrations'][n]
            )
            valence = state['valences'](self.mesh, n) if callable(state['valences']) else state['valences'][n]
            diffusivity = (
                state['diffusivities'](self.mesh, n) if callable(state['diffusivities']) else state['diffusivities'][n]
            )

            if self.use_activity_coeffs:
                activity_coeff = (
                    state['activity_coeffs'](self.mesh, n)
                    if callable(state['activity_coeffs'])
                    else state['activity_coeffs'][n]
                )
            else:
                activity_coeff = 1

            self.diffusivities[n].setValue(diffusivity)
            self.concentrations[n].setValue(value=concentration)
            self.valences[n].setValue(valence)
            self.activity_coeffs[n].setValue(activity_coeff)

        if state['charge_density_fixed'] is None:
            charge_density_fixed = -self.charge_density_mobile.value
        elif callable(state['charge_density_fixed']):
            charge_density_fixed = state['charge_density_fixed'](self.mesh)
        else:
            charge_density_fixed = state['charge_density_fixed']

        self.charge_density_fixed.setValue(charge_density_fixed)

        if 'additional_state' in state:
            self.additional_state = state['additional_state']

        self._precompute_jacobians()

    def get_state(self):
        state = {}
        state['potential_electrical'] = self.potential_electrical.value.copy()
        state['electromigration_coeff'] = self.electromigration_coeff.value.copy()
        state['permittivity'] = self.permittivity.value.copy()
        state['constant_faraday'] = self.constant_faraday.value.copy()
        state['t_step'] = self.t_step.value.copy()
        state['potential_electrical'] = self.potential_electrical.value.copy()

        state['diffusivities'] = [None] * self.ion_count
        state['concentrations'] = [None] * self.ion_count
        state['valences'] = [None] * self.ion_count
        state['activity_coeffs'] = [None] * self.ion_count

        reaction_bulk_parameters = [None] * self.reaction_bulk_count
        for k in range(self.reaction_bulk_count):
            reaction_bulk_parameters[k] = {}
            reaction_bulk_parameters[k]['stoichiometry'] = (self.reaction_bulk_parameters[k]['stoichiometry']).value
            reaction_bulk_parameters[k]['rate_exponents'] = (self.reaction_bulk_parameters[k]['rate_exponents']).value
            reaction_bulk_parameters[k]['rate_constant'] = (self.reaction_bulk_parameters[k]['rate_constant']).value

        reaction_surface_parameters = [None] * self.reaction_surface_count
        for k in range(self.reaction_surface_count):
            reaction_surface_parameters[k] = {}
            reaction_surface_parameters[k]['stoichiometry'] = (
                self.reaction_surface_parameters[k]['stoichiometry']
            ).value
            reaction_surface_parameters[k]['rate_exponents'] = (
                self.reaction_surface_parameters[k]['rate_exponents']
            ).value
            reaction_surface_parameters[k]['rate_constant'] = (
                self.reaction_surface_parameters[k]['rate_constant']
            ).value
            reaction_surface_parameters[k]['charge_transfer_direction'] = (
                self.reaction_surface_parameters[k]['charge_transfer_direction']
            ).value
            reaction_surface_parameters[k]['concentration_normalization'] = (
                self.reaction_surface_parameters[k]['concentration_normalization']
            ).value

        state['reaction_bulk_parameters'] = reaction_bulk_parameters
        state['reaction_surface_parameters'] = reaction_surface_parameters

        for n in range(self.ion_count):
            state['diffusivities'][n] = self.diffusivities[n].value.copy()
            state['valences'][n] = self.valences[n].value.copy()
            state['concentrations'][n] = self.concentrations[n].value.copy()
            state['activity_coeffs'][n] = self.activity_coeffs[n].value.copy()

        state['charge_density_fixed'] = self.charge_density_fixed.value.copy()

        state['additional_state'] = self.additional_state
        return state

    def _instantiate_state_variables(self):
        # Allocate objects for all of our internal variables. Everything is initialized to NaN

        nan = np.NAN
        nan_array_ion_count = np.array([np.NAN for n in range(self.ion_count)])
        none_list_ion_count = [nan] * self.ion_count

        self.potential_electrical = fipy.CellVariable(name="V", mesh=self.mesh, value=nan, hasOld=True)

        self.time = fipy.Variable(name='t', value=0.0)
        self.t_step = fipy.Variable(name='Δt', value=nan)
        self.t_step_old = fipy.Variable(name='Δt_old', value=nan)
        self.timestepping_BDF2 = fipy.Variable(name='BDF2', value=0.0)

        self.diffusivities = none_list_ion_count.copy()
        self.valences = none_list_ion_count.copy()

        self.concentrations = none_list_ion_count.copy()
        self.activity_coeffs = none_list_ion_count.copy()

        self.electromigration_coeff = fipy.FaceVariable(name='1/RT', mesh=self.mesh, value=nan)
        self.permittivity = fipy.CellVariable(name="ε", mesh=self.mesh, value=nan)

        for n in range(self.ion_count):
            ion_name = f"C_{n}"

            self.activity_coeffs[n] = fipy.CellVariable(mesh=self.mesh, name='γ', value=nan)
            self.diffusivities[n] = fipy.FaceVariable(name=f'D_{n}', mesh=self.mesh, value=nan)
            self.concentrations[n] = fipy.CellVariable(name=ion_name, mesh=self.mesh, value=nan, hasOld=True)
            self.concentrations[n].old2 = fipy.CellVariable(
                name=ion_name + "_old2", mesh=self.mesh, value=nan, hasOld=False
            )
            self.valences[n] = fipy.Variable(name=f'z_{n}', value=nan)

        self.reaction_bulk_parameters = [None] * self.reaction_bulk_count
        for k in range(self.reaction_bulk_count):
            self.reaction_bulk_parameters[k] = {
                'stoichiometry': fipy.Variable(value=nan_array_ion_count.copy()),
                'rate_exponents': fipy.Variable(value=nan_array_ion_count.copy()),
                'rate_constant': fipy.Variable(value=nan),
            }

        self.reaction_surface_parameters = [None] * self.reaction_surface_count
        for k in range(self.reaction_surface_count):
            self.reaction_surface_parameters[k] = {
                'stoichiometry': fipy.Variable(value=nan_array_ion_count.copy()),
                'rate_exponents': fipy.Variable(value=nan_array_ion_count.copy()),
                'rate_constant': fipy.Variable(value=nan),
                'charge_transfer_direction': fipy.Variable(value=nan),
                'concentration_normalization': fipy.Variable(value=nan),
            }


        self.constant_faraday = fipy.Variable(name="F", value=nan)
        self.charge_density_fixed = fipy.CellVariable(name="q_fixed", mesh=self.mesh, value=nan)

        self.is_first_step = True

        self.additional_state = {}

    def _formulate_equations(self):
        self.equations_ion = [None] * self.ion_count
        charge_densities_mobile = [None] * self.ion_count

        # Calculate the time step ratio for second order timestepping:
        omega = self.timestepping_BDF2 * self.t_step / self.t_step_old
        self.omega = omega

        # Compute various metrics that we want to plot, etc.

        self.charge_computed = self.potential_electrical.grad[0, -1] * self.permittivity[-2]

        self.current_displacement = (
            (self.potential_electrical.faceGrad - self.potential_electrical.old.faceGrad)
            / self.t_step
            * self.permittivity.faceValue
        )
        self.fluxes = [None] * self.ion_count
        self.currents_ionic = [None] * self.ion_count

        # This is an explicit and integrated form of the governing equations given in self.equations_ion.
        #   Source terms like the chemical reactions are ignored, so we just calculate the cell to cell fluxes.
        #   The discretization of the electrodiffusion term is different from self.equations_ion, so things won't match up exactly
        for n in range(self.ion_count):
            self.fluxes[n] = self.diffusivities[n] * (
                self.concentrations[n].faceGrad
                + self.potential_electrical.faceGrad
                * self.concentrations[n].faceValue
                * self.constant_faraday
                * self.electromigration_coeff
                * self.valences[n]
            )

            if self.use_activity_coeffs:
                self.fluxes[n] += (
                    self.diffusivities[n]
                    * self.concentrations[n].faceValue
                    * self.activity_coeffs[n].faceGrad
                    / self.activity_coeffs[n].faceValue
                )

            self.currents_ionic[n] = self.fluxes[n] * self.constant_faraday * self.valences[n]

        self.current_ionic = sum(self.currents_ionic)

        self.current_computed = self.current_ionic + self.current_displacement

        # Formulate the bulk reaction rate laws and compute the rates:
        reactions_bulk_rate = [None] * (self.reaction_bulk_count)

        for k in range(self.reaction_bulk_count):
            concentrations_with_exponents = [None] * self.ion_count

            # Compute C_n^gamma. Note that we use C_n as a standin for the activity
            for n in range(self.ion_count):
                concentrations_with_exponents[n] = (
                    self.concentrations[n] ** self.reaction_bulk_parameters[k]['rate_exponents'][n]
                )

            # Multiply all the terms together
            concentration_coefficient = reduce(lambda a, b: a * b, concentrations_with_exponents, 1)

            reactions_bulk_rate[k] = self.reaction_bulk_parameters[k]['rate_constant'] * concentration_coefficient

        # Formulate the surface reaction rate laws and compute the rates according to a butler volmer type law:
        #    Assumes a 1d mesh
        reactions_surface_rate = [None] * (self.reaction_surface_count)

        for k in range(self.reaction_surface_count):
            concentrations_with_exponents = [None] * self.ion_count

            # Compute C_n^gamma. Note that we use C_n as a standin for the activity
            for n in range(self.ion_count):
                concentrations_with_exponents[n] = (
                    self.concentrations[n].faceValue ** self.reaction_surface_parameters[k]['rate_exponents'][n]
                )

            # Multiply all the terms together
            concentration_coefficient = reduce(lambda a, b: a * b, concentrations_with_exponents, 1)

            # Compute the reaction rate by generalized butler volmer law
            reactions_surface_rate[k] = (
                self.reaction_surface_parameters[k]['rate_constant']
                * concentration_coefficient
                / self.reaction_surface_parameters[k]['concentration_normalization']
                * np.exp(
                    np.dot(self.potential_electrical.faceGrad, self.mesh.faceNormals)
                    * 1e-10  # characteristic length for reaction. Stern thickness
                    * self.reaction_surface_parameters[k][
                        'charge_transfer_direction'
                    ]  # -1 onto surface, 1 out of surface
                    * self.constant_faraday
                    * self.electromigration_coeff
                )
            )

        self.reactions_surface_rate = reactions_surface_rate


        # Formulate the PDE for each mobile species:
        for n in range(self.ion_count):
            # Formulate the BDF2 time differencing
            term_dc_dt = fipy.TransientTerm(var=self.concentrations[n],
                                            coeff=((1.0 + 2.0 * omega) / (1.0 + omega))) + (
                (self.concentrations[n].old2 - self.concentrations[n].old)
                * (omega**2 / (1.0 + omega))
                * self.mesh.cellVolumes
                / self.t_step
            )

            # ∇C
            term_diffusion = fipy.DiffusionTerm(var=self.concentrations[n], coeff=self.diffusivities[n])

            # C∇φ
            term_electromigration = fipy.DiffusionTerm(
                var=self.potential_electrical,
                coeff=self.constant_faraday
                * self.electromigration_coeff
                * self.diffusivities[n]
                * self.concentrations[n].faceValue
                * self.valences[n],
            )

            #C∇γ/γ
            if self.use_activity_coeffs:
                term_activity_diffusion = fipy.ConvectionTerm(
                    var=self.concentrations[n],
                    coeff=(self.diffusivities[n]
                           * self.activity_coeffs[n].faceGrad
                           / self.activity_coeffs[n].faceValue),
                )
            else:
                term_activity_diffusion = 0

            term_reaction_bulk = 0
            for k in range(self.reaction_bulk_count):
                term_reaction_bulk += self.reaction_bulk_parameters[k]['stoichiometry'][n] * reactions_bulk_rate[k]

            reactions_surface_flux = 0
            for k in range(self.reaction_surface_count):
                reactions_surface_flux += (
                    self.reaction_surface_parameters[k]['stoichiometry'][n]
                    * self.reactions_surface_rate[k]
                )

            # For now, all reactions are calculated at all exterior faces
            term_reaction_surface = (
                reactions_surface_flux * self.mesh.exteriorFaces * self.mesh.faceNormals
            ).divergence


            # Combine all of the terms together into one PDE per ion
            self.equations_ion[n] = (
                term_dc_dt
                == term_diffusion
                + term_activity_diffusion
                + term_electromigration
                + term_reaction_bulk
                + term_reaction_surface
            )

            charge_densities_mobile[n] = self.constant_faraday * self.valences[n] * self.concentrations[n]

        # Formulate Poisson's equations

        self.charge_density_mobile = sum(charge_densities_mobile)
        self.concentration_mobile_total = sum(self.concentrations)

        self.charge_density = self.charge_density_fixed + self.charge_density_mobile
        self.equation_poisson = (
            fipy.DiffusionTerm(var=self.potential_electrical, coeff=self.permittivity) == -self.charge_density
        )

        # Couple all of the equations together into one matrix

        self.equations_ion_coupled = reduce(lambda a, b: a & b, self.equations_ion)
        self.equations_coupled = self.equation_poisson & self.equations_ion_coupled

        # Cache the matrix for our Jacobian calculation
        self.equations_coupled.cacheMatrix()

    def _precompute_jacobians(self):
        #This is all 1d specific

        self.jacobians = dotmap.DotMap()

        var1 = fipy.CellVariable(mesh=self.mesh, value=1.0)
        var2 = fipy.CellVariable(mesh=self.mesh, value=1.0)

        jacobians_dc_dphi = []

        jacobians_dc_dphi_2 = []

        jacobians_dc_dc = []
        jacobians_dc_dc_2 = []

        for n in range(self.ion_count):
            term1 = fipy.DiffusionTerm(
                var=var2,
                coeff=(1)
                * self.constant_faraday
                * self.electromigration_coeff
                * self.diffusivities[n]
                * var1.faceValue
                * self.valences[n],
            )


            var1.constrain(value=1, where=self.mesh.exteriorFaces)

            term2 = (
                self.constant_faraday
                * self.electromigration_coeff
                * self.diffusivities[n]
                * self.valences[n]
                * var2.faceValue
            ).divergence

            def jacobian_function1(x):
                var2.setValue(x)
                return term1.justResidualVector()

            def jacobian_function2(x):
                var2.setValue(x)
                return term2()

            jacobian1 = scipy.optimize.approx_fprime(np.ones(var2.shape), jacobian_function1, 1e-3)

            jacobian2 = scipy.optimize.approx_fprime(np.ones(var2.shape), jacobian_function2, 1e-3)

            A = (self.constant_faraday * self.electromigration_coeff * self.diffusivities[n] * self.valences[n]).value

            jacobian3 = np.diag((A[1:] + A[0:-1]) / 2)

            var2.setValue(1)
            jacobian4 = scipy.optimize.approx_fprime(np.ones(var2.shape), jacobian_function2, 1e-3)

            jacobians_dc_dphi.append(jacobian1)
            jacobians_dc_dphi_2.append(jacobian2)
            jacobians_dc_dc.append(jacobian3)
            jacobians_dc_dc_2.append(jacobian4)

        self.jacobians.jacobians_dc_dphi_diffusive = jacobians_dc_dphi
        self.jacobians.jacobians_dc_dphi_coupling = jacobians_dc_dphi_2
        self.jacobians.jacobians_dc_dc_diffusive = jacobians_dc_dc
        self.jacobians.jacobians_dc_dc_coupling = jacobians_dc_dc_2

    def set_boundary_condition(self, condition_type, value, where):
        '''Set the external boundary conditions for both the electrical and species equations'''

        match condition_type:
            case 'voltage':
                value_evaluated = value(self.time) if callable(value) else value
                self.potential_electrical.constrain(value_evaluated, where=where)

            case 'charge':
                value_evaluated = value(self.time) if callable(value) else value
                self.potential_electrical.faceGrad.constrain(value_evaluated / self.permittivity.faceValue, where=where)

            case 'concentration':
                for n in range(self.ion_count):
                    value_evaluated = value(self.time, n) if callable(value) else value[n]
                    if value_evaluated is not None:
                        self.concentrations[n].constrain(value_evaluated, where=where)

            case 'flux':
                # Constrain the diffusivities to be 0 on the walls so there's no flux there
                for n in range(self.ion_count):
                    value_evaluated = value(self.time, n) if callable(value) else value[n]

                    print("Setting zero flux boundary condition")

                    if value_evaluated != 0.0 and value_evaluated is not None:
                        raise (NotImplementedError("non-zero or time dependent flux conditions are not implemented"))
                        # self.concentrations[n].faceGrad.constrain(value_evaluated/self.diffusivities[n][0], where=where)
                    elif value_evaluated is not None:
                        self.diffusivities[n].constrain(0.0, where=where)

    def _setup_plot(self):
        self.plot = {}

        lines = {}
        self.plot['lines'] = lines

        fig, axes = plt.subplots(2, 3, layout="constrained")

        fig2 = plt.figure()
        axes2 = fig2.subplots()
        self.plot['figure_convergence'] = fig2
        self.plot['axes_convergence'] = axes2
        axes2.set_title("Residuals")

        self.plot['figure'] = fig
        self.plot['axes'] = axes


        # Change which line is uncommented to switch between index and space coordinate for plots
        #x = self.mesh.x.value
        x = np.arange(0, len(self.mesh.x.value))

        fig.set_size_inches([10, 7])

        fig.suptitle(f"Time: {self.time.value:0.04e}")

        axes[0, 0].set_title("Concentration")
        lines['C'] = [axes[0, 0].plot(x, concentration.value)[0] for concentration in self.concentrations]

        lines['C_fixed'] = axes[0, 0].plot(x, self.charge_density_fixed / self.constant_faraday)[0]


        axes[1, 0].set_title("Charge Density")
        lines['Q'] = axes[1, 0].plot(x, self.charge_density.value)[0]

        axes[0, 1].set_title("Voltage")
        lines['V'] = axes[0, 1].plot(x, self.potential_electrical.value)[0]

        axes[1, 1].set_title("E Field")
        lines['E'] = axes[1, 1].plot(x, -self.potential_electrical.faceGrad.value[0][0:-1])[0]

        axes[0, 2].set_title("Electrode Current")
        lines['A_elec'] = axes[0, 2].plot([], [])[0]

        axes[1, 2].set_title("Electrode Charge")
        lines['Q_elec'] = axes[1, 2].plot([], [])[0]

        self.plot_frame = 0

    def generate_plot(self, filepath=None):
        fig = self.plot['figure']
        axes = self.plot['axes']
        lines = self.plot['lines']

        fig.suptitle(f"Time: {self.time.value:0.04e}")

        for n in range(self.ion_count):
            lines['C'][n].set_ydata(self.concentrations[n].value)

        lines['C_fixed'].set_ydata(self.charge_density_fixed / self.constant_faraday)

        lines['Q'].set_ydata(self.charge_density.value)

        lines['V'].set_ydata(self.potential_electrical.value)

        lines['E'].set_ydata(-self.potential_electrical.faceGrad.value[0][0:-1])

        line_A_elec_old = lines['A_elec'].get_data()
        lines['A_elec'].set_data(
            np.append(line_A_elec_old[0], self.time.value),
            np.append(line_A_elec_old[1], np.median(self.current_computed.value)),
        )

        line_Q_elec_old = lines['Q_elec'].get_data()
        lines['Q_elec'].set_data(
            np.append(line_Q_elec_old[0], self.time.value),
            np.append(line_Q_elec_old[1], self.charge_computed.value),
        )

        self.plot['axes_convergence'].clear()
        self.plot['axes_convergence'].plot(numpy.log10(numpy.abs(self.residuals)), '.')

        for axis in axes.flat:
            if axis.autoscale:
                axis.relim()
                axis.autoscale_view(True, True, True)

        plt.pause(0.001)
        # fig.pause(0.001)

        if filepath != None:
            plt.savefig(filepath)

    def update_old(self):
        for concentration in self.concentrations:
            concentration.old2.setValue(concentration.old.value)
            concentration.updateOld()

        self.potential_electrical.updateOld()

        self.t_step_old.setValue(self.t_step.value)

    def reset_to_old(self):
        for concentration in self.concentrations:
            concentration.setValue(concentration.old.value)
        self.potential_electrical.setValue(self.potential_electrical.old.value)

    def compute_jacobian_and_residual(self, state, dt):
        # TODO use sparse matrices instead. Unclear how that interacts with scipy optimization

        size = self.mesh.shape[0]

        state = state.copy()
        state.shape = (self.ion_count + 1, size)

        self.potential_electrical.setValue(state[0, :])

        # Don't allow negative concentrations as a solution
        for n in range(self.ion_count):
            concentration = state[n + 1, :]
            concentration[concentration < 1e-20] = 1e-20
            self.concentrations[n].setValue(state[n + 1, :])

        dt = self.t_step()

        residual = self.equations_coupled.justResidualVector(dt=dt)
        jacobian = self.equations_coupled.matrix.numpyArray

        jacobian_normalized = jacobian

        cell_volume_extended = numpy.expand_dims(self.mesh.cellVolumes, 1)
        potential_electrical_grad_extended = numpy.expand_dims(self.potential_electrical.grad[0], 1)
        potential_electrical_grad2_extended = numpy.expand_dims((self.potential_electrical.faceGrad).divergence, 1)

        for n in range(self.ion_count):
            jacobian_normalized[0:size, (n + 1) * size : (n + 2) * size] = numpy.diag(
                (self.constant_faraday * self.valences[n] * np.ones((size)) * self.mesh.cellVolumes)
            )


            jacobians_dc_dc_coupling = (
                self.jacobians.jacobians_dc_dc_coupling[n] * potential_electrical_grad_extended * cell_volume_extended
            )

            jacobians_dc_dc_diffusive = (
                self.jacobians.jacobians_dc_dc_diffusive[n] * cell_volume_extended * potential_electrical_grad2_extended
            )

            jacobian_normalized[(n + 1) * size : (n + 2) * size, (n + 1) * size : (n + 2) * size] += (
                -jacobians_dc_dc_diffusive - jacobians_dc_dc_coupling
            )

        residual_normalized = residual


        return residual_normalized, jacobian_normalized

    def solve_until(
        self,
        t_end,
        t_step_max=1e-3,
        t_step_increase=1.01,
        residual_max=1e-2,
        xtol=1e-10,
        iterations_max=60,
        iterations_grow=30,
        plotting=True,
        iterations_cumulative_plot=5,
        save_interval=0,
        output_directory=None,
        solver_type='rootfinding',
    ):

        with self.open_datafile():
            iterations_since_plot = 0
            time_saved_last = 0

            # Do one backward euler timestep to start things off if we have never stepped before
            if self.is_first_step:
                print("Doing initial backwards euler step")
                # Update twice to initialize both old and old2 previous timestep DOF values
                self.update_old()
                self.update_old()

            while self.time.value < t_end:
                if self.t_step.value > t_step_max:
                    self.t_step.setValue(t_step_max)

                residual = np.inf
                sweeps = 0

                fun = scipy.optimize._optimize.MemoizeJac(self.compute_jacobian_and_residual)


                if solver_type == 'rootfinding':

                    while residual > residual_max:
                        x = numpy.ravel(numpy.stack([self.potential_electrical] + self.concentrations))

                        solution = scipy.optimize.root(
                            self.compute_jacobian_and_residual,
                            x,
                            (self.t_step()),
                            method='hybr',
                            jac=True,
                        )

                        residuals = fun(solution.x, self.t_step())
                        self.residuals = residuals

                        residual = numpy.sqrt(numpy.sum(residuals**2))
                        sweeps = solution.nfev

                        if residual > residual_max:

                            status_string = (
                                f"t: {self.time.value:0.4e}    Δt: {self.t_step.value:0.3e}    ω:"
                                f" {self.omega.value:.3f}    Iter: {sweeps:0>4d}    Res: {residual:.5e}   FAILED"
                            )
                            print(status_string)

                            # Reset our solution to the old value
                            self.reset_to_old()

                            # Decrease our timestep
                            self.t_step.setValue(self.t_step.value * 0.5 * (residual_max / residual) ** 0.5)
                            # print(f"Retrying with Timestep: {self.t_step.value:0.3e} ")

                            if self.t_step < 1e-10:
                                self.t_step.setValue(1e-10)

                elif solver_type == 'fixed_point':
                    while residual > residual_max or sweeps < 2:
                        # Solve this iteration
                        residual = self.equations_coupled.sweep(dt=self.t_step.value, cacheResidual=True, underRelaxation=0.95)

                        residuals = self.equations_coupled.residualVector
                        self.residuals = residuals

                        sweeps += 1

                        # Check if we've spent too long and it's not converging
                        if sweeps > iterations_max:
                            status_string = (
                                f"Time: {self.time.value:0.4e}    Timestep: {self.t_step.value:0.3e}    Omega:"
                                f" {self.omega.value:.3f}    Sweeps: {sweeps:0>4d}    Res: {residual:.5e}    FAILED"
                            )
                            print(status_string)

                            # Reset our solution to the old value
                            self.reset_to_old()

                            # Decrease our timestep
                            self.t_step.setValue(self.t_step.value * 0.5)

                            # Try again
                            sweeps = 0

                else:
                    raise (ValueError("Solver type not one of 'rootfinding' or 'fixed_point'"))

                # If we got here our timestep succeded:

                # Print status
                status_string = (
                    f"t: {self.time.value:0.4e}    Δt: {self.t_step.value:0.3e}    ω: {self.omega.value:.3f}    Iter:"
                    f" {sweeps:0>4d}    Res: {residual:.5e}"
                )
                print(status_string)

                iterations_since_plot += 1

                # Save the results in output file
                if self.write_data_enabled and (self.time.value - time_saved_last) > save_interval:
                    print("Saving!")
                    timestep = self.datafile.timestep_table.row
                    timestep['voltage'] = self.potential_electrical.value
                    timestep['concentrations'] = [concentration.value for concentration in self.concentrations]
                    timestep['time'] = self.time.value
                    timestep['residual'] = residual
                    timestep['sweeps'] = sweeps
                    timestep['current'] = np.median(self.current_computed.value)
                    timestep['timestamp'] = datetime.datetime.now().isoformat(timespec = 'milliseconds')

                    timestep.append()

                    time_saved_last = self.time.value.copy()

                    self.datafile.file.flush()

                # Plot the results
                if plotting and (iterations_since_plot >= iterations_cumulative_plot):
                    if output_directory != None:
                        filepath = os.path.join(output_directory, f'{self.plot_frame:04d}.jpg')
                    else:
                        filepath = None

                    self.generate_plot(filepath)
                    iterations_since_plot = 0
                    self.plot_frame += 1

                # Update the old variables
                self.update_old()

                # Calculate the new time
                self.time.setValue(self.time.value + self.t_step.value)

                # Increase the timestep if we need to
                if sweeps < iterations_grow and self.t_step.value < t_step_max:
                    self.t_step.setValue(self.t_step.value * t_step_increase)

                # If it's the first timestep, change timestepping schemes for future steps
                if self.is_first_step:
                    # We succeeded in our first step
                    print('Initial backwards Euler iteration done!')

                    # Update the old values and switch to BDF2 timestepping
                    self.update_old()
                    self.timestepping_BDF2.setValue(1.0)

                    self.is_first_step = False

            # Completed our solution interval
