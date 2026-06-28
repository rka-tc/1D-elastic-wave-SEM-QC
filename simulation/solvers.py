#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Sub-module for implementing different wave equation solvers.}

{
    Copyright (C) [2023]  [Malte Schade]

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
}
"""

# -------- IMPORTS --------
# Built-in modules
from typing import Dict, List, Any
from time import sleep
from itertools import product
import json

# Other modules
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from qiskit_aer import AerSimulator

# Own modules
from utility.transformMain import HamiltonianBuilderSEM
from utility.processing import MediumProcessor, StateProcessor
from utility.backends import CloudBackend, LocalBackend
from utility.circuits import CircuitGen1DA
from utility.tomography import parallel_transport

# -------- CLASSES --------
class Solver1D:
    """
    A class for solving 1D elastic wave forward problems using quantum computing methods.

    Args:
        logger (object): A logging instance to record the process and errors.
        **kwargs: Arbitrary keyword arguments for configuration.

    Attributes:
        logger (object): Logger for logging information.
        kwargs (dict): Dictionary of keyword arguments.
        data (dict): Dictionary to store various configurations and results.
    """
    def __init__(self, base_data: object, logger: object, **kwargs) -> None:
        self.base_data = base_data
        self.logger = logger
        self.kwargs = kwargs
        self.idx = kwargs['idx']
        self.data = {}

        # Check parameters
        self.check_kwargs()
        self.logger.info('Parameters checked for validity.')

        # Save kwargs
        self.data['settings'] = kwargs

        # Set time steps
        self.times = self.get_times(kwargs['nt'], kwargs['dt']) 
        self.data['times'] = self.times
        self.logger.info(f'Solving for {self.kwargs["nt"]} time steps.')
        self.logger.debug(f'Times: {self.data["times"]}')

        # Set transform
        self.logger.info('Calculating Transformation and Hamiltonian.')
        self.tf = self.get_transform(kwargs['nx'], kwargs['order'], kwargs['L'], kwargs['mu'], kwargs['rho']
                                        )
        self.data['transform'] = self.tf.get_dict()
        self.logger.info('Calculation completed.')

        # Update nx and initial conditions for boundary conditions
        self.kwargs['nx'] = self.tf.nx
        self.kwargs['u'] = np.array(self.kwargs['u'])[1:-1].tolist()
        self.kwargs['v'] = np.array(self.kwargs['v'])[1:-1].tolist()

        # Set medium
        self.md = self.get_medium_processor(kwargs['mu'], kwargs['rho'])
        self.medium = self.md.get_medium()
        self.data['medium'] = self.md.get_dict()
        self.logger.info('Medium initialized.')

    def check_kwargs(self):
        """
        Validates the keyword arguments provided to the solver.

        Raises:
            AssertionError: If any of the conditions for the arguments are not met.
        """
        # Nel = (self.kwargs['nx'] - 1) // (self.kwargs['order'] - 1)
        # assert self.kwargs['nx'] == len(self.kwargs['u']),   'length of u must equal nx'
        # assert self.kwargs['nx'] == len(self.kwargs['v']),   'length of v must equal nx'
        # assert self.kwargs['nx'] == len(self.kwargs['mu'])-1,                'length of mu must equal Nel'
        # assert self.kwargs['nx'] == len(self.kwargs['rho']),               'length of rho must equal Nel'
        assert np.all(np.array(self.kwargs['mu']) > 0), 'mu must be positive'
        assert np.all(np.array(self.kwargs['rho']) > 0), 'rho must be positive'
        assert self.kwargs['nt'] > 0, 'nt must be greater than zero'
        assert self.kwargs['dt'] > 0, 'dt must be greater than zero'
        assert self.kwargs['L'] > 0, 'dx must be greater than zero'
        assert self.kwargs['order'] in [1,2,3,4,5,6,7,8], "Order must be in [1,2,3,4]"

    def get_times(self, nt, dt) -> np.ndarray:
        """
        Sets up the time steps for the solver based on the 'nt' and 'dt' arguments.
        
        Args:
            nt (int): The number of time steps.
            dt (float): The time step size.
            
        Returns:
            np.ndarray: An array of time steps.
        """

        return np.arange(nt)*dt

    def get_transform(self,nx: int, order: int, L: float, mu: np.ndarray, rho: np.ndarray
                      ) -> HamiltonianBuilderSEM:
        """
        Initializes and returns a finite difference transformation object for 1D analysis.

        This method creates an instance of the FDTransform1DA class, which is used for
        transforming the state space based on the provided medium parameters.

        Args:
            mu (np.ndarray): An array of µ values representing the medium shear modulus.
            rho (np.ndarray): An array of ρ values representing the medium density.
            dx (float): The spatial step size.
            nx (int): The number of spatial steps.
            order (int): The order of the finite difference scheme.

        Returns:
            FDTransform1DA: An instance of FDTransform1DA initialized with the given parameters.
        """
        return HamiltonianBuilderSEM(nx, order, L, mu, rho)

    def get_medium_processor(self, mu: np.ndarray, rho: np.ndarray) -> MediumProcessor:
        """
        Initializes and configures a medium processor for the simulation.

        This method creates an instance of MediumProcessor, setting it up with the 
        specified µ (mu) and ρ (rho) values representing the medium's properties. 

        Args:
            mu (np.ndarray): An array of µ values, representing the medium shear modulus.
            rho (np.ndarray): An array of ρ values, representing the medium density.

        Returns:
            MediumProcessor: An initialized and configured medium processor object.
        """

        # Initialize medium processor
        md =  MediumProcessor(len(mu), len(rho))

        # Set medium parameters
        md.set_mu(mu)
        md.set_rho(rho)

        return md

class Solver1DODE(Solver1D):
    """
    A subclass of Solver1D for solving with a classical
        Ordinary Differential Equations (ODEs) solver.

    Inherits from Solver1D and adds specific methods for handling ODEs.

    Args:
        logger (object): A logging instance to record the process and errors.
        **kwargs: Arbitrary keyword arguments for configuration.
    """

    def __init__(self, base_data: object, logger: object, **kwargs) -> None:
        super().__init__(base_data, logger, **kwargs)
        self.st = StateProcessor(self.kwargs['nx'], self.kwargs['nt'], shift=0)
        self.st.set_u(self.kwargs['u'], 0)
        self.st.set_v(self.kwargs['v'], 0)
        self.st.forward_state(0, self.tf.sqrt_m)
        self.logger.info('Initial state forward-transformed.')

    def run(self) -> Dict[str, Any]:
        """
        Runs the ODE solver and processes the results.

        Returns:
            Dict[str, Any]: A dictionary containing the field data and other results.
        """
        self.logger.info('Solving ODE.')
        # print("transformation matrix shape:", self.tf.t.shape) # --- IGNORE ---
        self.st.states = solve_ivp(lambda t, y: self.tf.q @ y, (0, self.times[-1]),
                                     self.st.get_state(0), t_eval=self.times,
                                     method='Radau').y.T
        self.logger.info('ODE solved.')

        _ = [self.st.inverse_state(i, self.tf.inv_sqrt_m)
         for i in range(len(self.times))]
        self.logger.info('States inverse-transformed.')

        self.data['field'] = self.st.get_dict()
        return self.data

class Solver1DEXP(Solver1D):
    """
    A subclass of Solver1D for solving with a classical
        Matrix exponential time evolution solver.

    Inherits from Solver1D.

    Args:
        logger (object): A logging instance to record the process and errors.
        **kwargs: Arbitrary keyword arguments for configuration.
    """

    def __init__(self, base_data: object, logger: object, **kwargs) -> None:
        super().__init__(base_data, logger, **kwargs)
        self.st = StateProcessor(self.kwargs['nx'], self.kwargs['nt'], shift=0)
        # print("kwargs[nx]:", self.kwargs['nx'] )
        self.st.set_u(self.kwargs['u'], 0)
        self.st.set_v(self.kwargs['v'], 0)
        self.st.forward_state(0, self.tf.t @ self.tf.sqrt_m)
        # self.st.forward_state(0, self.tf.sqrt_m)       
        self.logger.info('Initial state forward-transformed.')

    def run(self) -> Dict[str, Any]:
        """
        Runs the matrix exponential solver and processes the results.

        Returns:
            Dict[str, Any]: A dictionary containing the field data and other results.
        """
        self.logger.info('Solving matrix exponential.')
        np.save('phi0.npy', self.st.get_state(0))
        self.st.states = np.array([
            np.real(scipy.linalg.expm(time * -1j * self.tf.h) @ self.st.get_state(0))
            for time in self.times])
        self.logger.info('Matrix exponential solved.')

        _ = [self.st.inverse_state(i, self.tf.inv_sqrt_m @ self.tf.inv_t)
        # _ = [self.st.inverse_state(i, self.tf.inv_sqrt_m)
         for i in range(len(self.times))]
        self.logger.info('States inverse-transformed.')

        self.data['field'] = self.st.get_dict()
        return self.data

class Solver1DLocal(Solver1D):
    """
    A subclass of Solver1D for local quantum computing simulations.

    Inherits from Solver1D and adds specific methods for handling local simulations.

    Args:
        logger (object): A logging instance to record the process and errors.
        **kwargs: Arbitrary keyword arguments for configuration.
    """

    def __init__(self, base_data: object, logger: object, **kwargs) -> None:
        super().__init__(base_data, logger, **kwargs)
        self.st = StateProcessor(self.kwargs['nx'], self.kwargs['nt'], shift=-2)
        self.st.set_u(self.kwargs['u'], 0)
        self.st.set_v(self.kwargs['v'], 0)
        self.st.forward_state(0, self.tf.t @ self.tf.sqrt_m)
        self.circuit_groups = []
        self.logger.info('Initial state transformed.')

    def run(self) -> Dict[str, Any]:
        """
        Runs the local solver, including quantum circuit generation, execution, and statevector extraction.

        Returns:
            Dict[str, Any]: A dictionary containing the field data and other results.
        """

        self.logger.info('Initializing backend.')
        backend = LocalBackend(self.logger,
                               backend=None,
                               fake=self.kwargs['backend']['fake'],
                               method=self.kwargs['backend']['method'],
                               seed=self.kwargs['backend']['seed'],
                               shots=self.kwargs['backend']['shots'],
                               optimization=self.kwargs['backend']['optimization'],
                               resilience=self.kwargs['backend']['resilience'],
                               local_transpilation = self.kwargs['backend']['local_transpilation'],
                               max_parallel_experiments=0)
        self.logger.info('Backend initialized.')

        self.logger.info('Generating statevector circuits.')
        circuit_gen = CircuitGen1DA(self.logger, backend.fake_backend)
        self.circuit_groups = circuit_gen.statevector_circuits(
            self.st.get_state(0),
            self.tf.h,
            self.times[1:],
            self.kwargs['backend']['synthesis'],
            self.kwargs['backend']['batch_size'],
            self.kwargs['backend']['optimization'],
            self.kwargs['backend']['seed'],
            self.kwargs['backend']['local_transpilation'])

        self.logger.info('Running circuits with AerSimulator.')
        
        # Create AerSimulator for statevector extraction
        aer_simulator = AerSimulator(
            method='statevector',
            seed_simulator=self.kwargs['backend']['seed']
        )
        
        # Run all circuits and collect statevectors
        statevectors = []
        for group_idx, circuits in enumerate(self.circuit_groups):
            self.logger.info(f'Running circuit group {group_idx + 1}/{len(self.circuit_groups)}.')
            
            # Transpile circuits to basis gates for Aer compatibility
            from qiskit.compiler import transpile
            transpiled_circuits = transpile(
                circuits,
                backend=aer_simulator,
                optimization_level=0
            )
            
            # Run circuits
            job = aer_simulator.run(transpiled_circuits)
            result = job.result()
            
            # Extract statevectors
            for circ_idx in range(len(transpiled_circuits)):
                try:
                    # Get statevector from result
                    sv = result.data(circ_idx)['statevector']
                    statevectors.append(np.array(sv))
                    self.logger.debug(f'Extracted statevector {len(statevectors)} with shape {np.array(sv).shape}')
                except (KeyError, IndexError, TypeError) as e:
                    self.logger.warning(f'Could not extract statevector for circuit {circ_idx}: {e}')
        
        self.logger.info(f'Extracted {len(statevectors)} statevectors.')
        
        # Store statevectors in state processor (starting from index 1, as index 0 is initial state)
        states_raw = np.array(statevectors)
        
        # Apply parallel transport phase correction
        self.st.states = np.real(parallel_transport(states_raw, self.st.get_state(0)))
        self.logger.info('State polarization corrected.')
        
        # Inverse transform states
        _ = [self.st.inverse_state(i, self.tf.inv_sqrt_m @ self.tf.inv_t)
         for i in range(1, len(self.times))]
        self.logger.info('States inverse-transformed.')

        self.data['field'] = self.st.get_dict()
        return self.data


class Solver1DCloud(Solver1D):
    """
    A subclass of Solver1D for cloud-based quantum computing simulations.

    Inherits from Solver1D and adds specific methods for handling cloud-based simulations.

    Args:
        logger (object): A logging instance to record the process and errors.
        **kwargs: Arbitrary keyword arguments for configuration.
    """

    def __init__(self, base_data: object, logger: object, **kwargs) -> None:
        super().__init__(base_data, logger, **kwargs)
        self.st = StateProcessor(self.kwargs['nx'], self.kwargs['nt'], shift=1)
        self.st.set_u(self.kwargs['u'], 0)
        self.st.set_v(self.kwargs['v'], 0)
        self.st.forward_state(0, self.tf.t @ self.tf.sqrt_m)
        self.logger.info('Initial state transformed.')

    def run(self) -> Dict[str, Any]:
        """
        Runs the cloud-based solver, including quantum circuit generation,
        execution, and tomography.

        Returns:
            Dict[str, Any]: A dictionary containing the field data and other results.
        """

        self.logger.info('Initializing backend.')
        backend = CloudBackend(self.logger,
                               backend=self.kwargs['backend']['backend'],
                               fake=self.kwargs['backend']['fake'],
                               seed=self.kwargs['backend']['seed'],
                               shots=self.kwargs['backend']['shots'],
                               optimization=self.kwargs['backend']['optimization'],
                               resilience=self.kwargs['backend']['resilience'],
                               local_transpilation = self.kwargs['backend']['local_transpilation'])

        sampler, _ = backend.get_sampler()
        self.logger.info('Backend initialized.')

        self.logger.info('Generating circuits.')
        circuit_gen = CircuitGen1DA(self.logger,
                                    backend.fake_backend
                                    if backend.fake_backend
                                    else backend.backend)
        circuit_groups = circuit_gen.tomography_circuits(
            self.st.get_state(0),
            self.tf.h,
            self.times[1:],
            self.kwargs['backend']['synthesis'],
            self.kwargs['backend']['batch_size'],
            self.kwargs['backend']['optimization'],
            self.kwargs['backend']['seed'],
            self.kwargs['backend']['local_transpilation'])

        self.logger.info(f'Submitting {len(circuit_groups)} jobs to backend.')
        jobs, job_ids = [], []
        for i, circuits in enumerate(circuit_groups):
            job_transmitted = False
            while not job_transmitted:
                try:
                    job = sampler.run(circuits)
                    job_transmitted = True
                except ConnectionError as e:
                    self.logger.warning(f'Job transmission {i} failed with error {e}.\
                                        Retrying in 5 seconds.')
                    sleep(5)
            self.logger.info(f'Job {i} submitted with job_id {job.job_id()}.')
            jobs.append(job)
            job_ids.append(job.job_id())
        self.logger.info('Jobs submitted.')
        _save_jobids(job_ids, self.base_data/f'jobids_{self.idx}.json')
        _wait_for_completion(jobs, self.logger)
        result_groups = [job.result() for job in jobs]
        self.logger.info('Jobs completed.')

        self.logger.info('Running tomography.')
        tomo = TomographyReal(self.logger, self.kwargs['backend']['fitter'])
        observables = list(product("ZX", repeat=int(np.log2(self.tf.h.shape[0]))))
        self.logger.debug(f'Observables: {observables}')
        states_raw = tomo.run_tomography(result_groups, observables, self.times[1:])
        self.logger.info('Tomography completed.')

        self.st.states = np.real(parallel_transport(states_raw, self.st.get_state(0)))
        self.logger.info('State polarization corrected.')
        _ = [self.st.inverse_state(i, self.tf.inv_sqrt_m @ self.tf.inv_t)
         for i in range(1, len(self.times))]
        self.logger.info('States inverse-transformed.')

        self.data['field'] = self.st.get_dict()
        return self.data

    def load(self) -> Dict[str, Any]:
        """
        Loads quantum computation jobs from IBM Quantum using provided job IDs,
        processes the results, and updates the state processor.

        This method retrieves jobs from the IBM Quantum service, waits for their completion, and 
        then runs quantum state tomography on the results. It applies corrections to the states and 
        performs inverse transformations to get the final state data.

        Args:
            job_ids (List[str]): A list of job IDs for retrieval from IBM Quantum.

        Returns:
            Dict[str, Any]: A dictionary containing the processed field data.
        """
        self.logger.info('Loading jobs from IBM Quantum.')
        job_ids = json.load(open(self.base_data/f'jobids_{self.idx}.json',
                                 'r', encoding='utf8'))
        service = BackendService().service
        jobs = [service.job(job_id) for job_id in job_ids]
        _wait_for_completion(jobs, self.logger)
        result_groups = [job.result() for job in jobs]
        self.logger.info('Jobs completed.')

        self.logger.info('Running tomography.')
        tomo = TomographyReal(self.logger, self.kwargs['backend']['fitter'])
        observables = list(product("ZX", repeat=int(np.log2(self.tf.h.shape[0]))))
        self.logger.debug(f'Observables: {observables}')
        states_raw = tomo.run_tomography(result_groups, observables, self.times[1:])
        self.logger.info('Tomography completed.')

        self.st.states = np.real(parallel_transport(states_raw, self.st.get_state(0)))
        self.logger.info('State polarization corrected.')
        _ = [self.st.inverse_state(i, self.tf.inv_sqrt_m @ self.tf.inv_t)
         for i in range(1, len(self.times))]
        self.logger.info('States inverse-transformed.')

        self.data['field'] = self.st.get_dict()
        return self.data

# -------- FUNCTIONS --------
def _wait_for_completion(jobs: List[object], logger: object, sleep_time: float = 10) -> None:
    """
    Waits for a list of jobs to complete.

    Args:
        jobs (List[object]): A list of jobs to wait for.
        logger (object): A logger to record the status of the jobs.
    """

    all_completed = False
    while not all_completed:
        sleep(sleep_time)
        status = [job.status().name for job in jobs]
        logger.debug(f"Jobs status: {status}")
        if 'ERROR' in status:
            logger.debug(f"Fatal error occurred: {[job.status() for job in jobs]}")
            raise RuntimeError('Runtime error in simulating the quantum circuit.\
                This might be a problem of your qiskit installation.')
        completed = [job.status().name == 'DONE' for job in jobs]
        logger.info(f"Jobs completed: {sum(completed)} | {len(jobs)}")
        all_completed = all(completed)

def _save_jobids(job_ids: List[str], path: str, indent: int = 4,
                 encoding: str = 'utf8') -> None:
    """
    Saves a list of job IDs to a JSON file.

    Args:
        job_ids (List[str]): A list of job IDs to save.
        path (str): The path to save the job IDs to.
    """

    with open(path, 'w', encoding=encoding) as f:
        json.dump(job_ids, f, indent=indent)
