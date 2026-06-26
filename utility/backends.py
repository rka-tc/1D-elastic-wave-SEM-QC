from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_aer.noise import NoiseModel


# -------- FAKE PROVIDERS (add more as needed) --------
# Uncomment and extend if you want noise models from fake backends
# from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeGuadalupeV2
# FAKE_PROVIDERS = {
#     'guadalupe': FakeGuadalupeV2(),
#     'sherbrooke': FakeSherbrooke(),
# }
FAKE_PROVIDERS: Dict[str, object] = {}  # empty by default for noise-free local sim


# -------- OPTIONS REPLACEMENT --------
# Replaces qiskit_ibm_runtime.Options with a plain dataclass
@dataclass
class SimulatorOptions:
    seed_simulator: Optional[int] = None
    coupling_map: Optional[object] = None
    noise_model: Optional[object] = None

@dataclass
class TranspileOptions:
    skip_transpilation: bool = False
    seed_transpiler: Optional[int] = None

@dataclass
class ExecutionOptions:
    shots: int = 1024

@dataclass
class LocalOptions:
    optimization_level: int = 1
    resilience_level: int = 0
    simulator: SimulatorOptions = field(default_factory=SimulatorOptions)
    transpilation: TranspileOptions = field(default_factory=TranspileOptions)
    execution: ExecutionOptions = field(default_factory=ExecutionOptions)


# -------- BASE BACKEND --------
class BaseBackend:
    """
    Base class for quantum backends. Holds shared option initialization logic.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.options = self.init_options(
            kwargs.get('fake'),
            kwargs.get('seed'),
            kwargs.get('shots', 1024),
            kwargs.get('optimization', 1),
            kwargs.get('resilience', 0),
            kwargs.get('local_transpilation', False)
        )

    def init_options(self, fake: str, seed: int, shots: int,
                     optimization: int, resilience: int,
                     local_transpilation: bool) -> LocalOptions:
        """
        Builds a LocalOptions object mirroring the old qiskit_ibm_runtime.Options interface.
        """
        self.fake_backend = FAKE_PROVIDERS.get(fake, None) if fake else None

        sim_opts = SimulatorOptions(seed_simulator=seed)
        if self.fake_backend:
            sim_opts.coupling_map = self.fake_backend.coupling_map
            sim_opts.noise_model = NoiseModel.from_backend(self.fake_backend)

        return LocalOptions(
            optimization_level=optimization,
            resilience_level=resilience,
            simulator=sim_opts,
            transpilation=TranspileOptions(
                skip_transpilation=local_transpilation,
                seed_transpiler=seed
            ),
            execution=ExecutionOptions(shots=shots)
        )


# -------- LOCAL BACKEND --------
class LocalBackend(BaseBackend):
    """
    Local Aer backend. Drop-in replacement for the old LocalBackend that
    depended on qiskit_ibm_runtime.Options.

    All configuration is passed via kwargs — same interface as before.
    """

    def __init__(self, logger: object, **kwargs) -> None:
        self.logger = logger
        self.logger.info('Setting backend options.')
        super().__init__(**kwargs)

        # Build the AerSimulator used for transpilation targeting
        aer_kwargs = {'method': self.kwargs.get('method', 'matrix_product_state')}
        if self.options.simulator.seed_simulator is not None:
            aer_kwargs['seed_simulator'] = self.options.simulator.seed_simulator
        if self.options.simulator.noise_model is not None:
            aer_kwargs['noise_model'] = self.options.simulator.noise_model
        if self.options.simulator.coupling_map is not None:
            aer_kwargs['coupling_map'] = self.options.simulator.coupling_map

        self.fake_backend = AerSimulator(**aer_kwargs)
        self.logger.info('Local Aer backend initialized.')

    def get_sampler(self) -> Tuple[AerSampler, None]:
        """
        Returns an AerSampler configured for local simulation.
        No session needed — returns (sampler, None) to match cloud interface.
        """
        self.logger.info('Initializing AerSampler backend.')

        backend_options = {
            'method': self.kwargs.get('method', 'matrix_product_state'),
            'max_parallel_threads': 0,
            'max_parallel_experiments': self.kwargs.get('max_parallel_experiments', 64),
            'max_parallel_shots': 1,
            'statevector_parallel_threshold': 4,
        }

        # Attach noise/coupling only if a fake backend was requested
        if self.options.simulator.noise_model is not None:
            backend_options['noise_model'] = self.options.simulator.noise_model
        if self.options.simulator.coupling_map is not None:
            backend_options['coupling_map'] = self.options.simulator.coupling_map
        if self.options.simulator.seed_simulator is not None:
            backend_options['seed_simulator'] = self.options.simulator.seed_simulator

        transpile_options = {
            'seed_transpiler': self.kwargs.get('seed')
        }

        run_options = {
            'shots': self.options.execution.shots
        }

        self.logger.debug(
            f'Aer backend_options={backend_options}, '
            f'transpile_options={transpile_options}, '
            f'run_options={run_options}'
        )

        sampler = AerSampler(
            backend_options=backend_options,
            transpile_options=transpile_options,
            run_options=run_options
        )

        return sampler, None


# -------- CLOUD BACKEND (stub) --------
# Kept as a stub so imports don't break if CloudBackend is referenced elsewhere.
# To restore cloud functionality, re-add qiskit_ibm_runtime imports.
class CloudBackend(BaseBackend):
    """
    Stub for cloud backend. Raises clearly if accidentally used.
    Re-enable by restoring qiskit_ibm_runtime imports.
    """

    def __init__(self, logger: object, **kwargs) -> None:
        raise NotImplementedError(
            "CloudBackend requires qiskit_ibm_runtime. "
            "Use LocalBackend for local Aer simulation."
        )

    def get_sampler(self):
        raise NotImplementedError("CloudBackend is disabled.")