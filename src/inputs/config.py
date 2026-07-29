"""Parametros que descrevem uma execucao.

Reune o carregamento dos YAMLs do Hydra, os pedidos que a UI/CLI monta e
entrega ao workflow, e a configuracao do ensemble estocastico.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from hydra import compose, initialize_config_dir


# --------------------------------------------------------------------------
# Carregamento dos YAMLs (conf/)
# --------------------------------------------------------------------------


class EnvironmentRepository:
    """Compoe os YAMLs de conf/ aplicando os overrides da execucao."""

    def compose_config(
        self,
        config_name: str = "main",
        environment: Optional[str] = None,
        additional_overrides: Optional[List[str]] = None,
    ):
        overrides = list(additional_overrides or [])
        if environment is not None:
            overrides.append(f'environment="{environment}"')

        config_dir = Path(__file__).resolve().parents[2] / "conf"
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            return compose(config_name=config_name, overrides=overrides)


# --------------------------------------------------------------------------
# Pedidos de execucao
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigRequest:
    config_name: str = "main"
    environment: str = "2019"
    simulation_name: str = "sim4validation"
    run_name: Optional[str] = None
    shp_zip: Optional[str] = None
    min_long: Optional[float] = None
    max_long: Optional[float] = None
    min_lat: Optional[float] = None
    max_lat: Optional[float] = None

    def to_overrides(self) -> List[str]:
        overrides: List[str] = [f"simulation={self.simulation_name}"]
        if self.shp_zip:
            overrides.append(f'paths.plataformas_shp="{self.shp_zip}"')
        if self.min_long is not None:
            overrides.append(f"copernicusmarine.min_long={self.min_long}")
        if self.max_long is not None:
            overrides.append(f"copernicusmarine.max_long={self.max_long}")
        if self.min_lat is not None:
            overrides.append(f"copernicusmarine.min_lat={self.min_lat}")
        if self.max_lat is not None:
            overrides.append(f"copernicusmarine.max_lat={self.max_lat}")
        if self.run_name:
            overrides.append(f'simulation.name="{self.run_name}"')
        return overrides


@dataclass(frozen=True)
class ObservedSpillRequest:
    spill_path: Path
    offset_hours: float = 0.0
    start_index: int = 1
    padding_animation_frame: float = 0.1


@dataclass
class ObservedSpillContext:
    manchas: Any
    plot_bounds: Tuple[float, float, float, float]


@dataclass(frozen=True)
class ValidationRunRequest:
    config_name: str = "main"
    skip_animation: bool = False
    skip_simulation: bool = False
    skip_plots: bool = False
    evaluation: bool = False
    optimize_wdf_cdf: bool = False
    fast_particles_per_wdf: int = 1
    wdf_min: float = 0.0
    wdf_max: float = 0.05
    wdf_step: float = 0.0025
    cdf_min: float = 0.5
    cdf_max: float = 1.0
    cdf_step: float = 0.1
    padding_animation_frame: float = 0.1
    wind_drift_factor: Optional[float] = None
    current_drift_factor: Optional[float] = None
    processes_dispersion: Optional[str] = None
    processes_evaporation: Optional[str] = None
    oil_types: Optional[str] = None
    oil_types_file: Optional[str] = None
    environment: str = "2019"
    shp_zip: Optional[str] = None
    min_long: Optional[float] = None
    max_long: Optional[float] = None
    min_lat: Optional[float] = None
    max_lat: Optional[float] = None
    start_index: int = 0
    environmental_offset_hours: Optional[float] = None
    environmental_offset_values: Optional[Sequence[float]] = None
    run_name: Optional[str] = None
    forcing_source: str = "COPERNICUS"
    current_dataset_path: Optional[str] = None
    wind_dataset_path: Optional[str] = None
    current_dataset_paths: Optional[Sequence[str]] = None
    wind_dataset_paths: Optional[Sequence[str]] = None
    disable_environment_offset: bool = False


@dataclass(frozen=True)
class DeterministicRunRequest:
    """Simulacao a partir de um ponto, data e duracao informados pelo usuario.

    Nao usa manchas observadas: serve para prever o destino de um vazamento
    hipotetico. A area de forcing e derivada do ponto de vazamento.
    """

    lon: float
    lat: float
    start_time: datetime
    duration_days: float = 1.0
    leak_duration_hours: float = 0.0
    seed_radius_m: float = 1000.0
    num_seed_elements: int = 2000
    time_step_minutes: float = 5.0
    output_time_step_minutes: float = 60.0
    wind_drift_factor: Optional[float] = None
    current_drift_factor: Optional[float] = None
    oil_type: Optional[str] = None
    processes_dispersion: Optional[str] = None
    processes_evaporation: Optional[str] = None
    config_name: str = "main"
    environment: str = "2019"
    run_name: Optional[str] = None
    forcing_source: str = "COPERNICUS"
    current_dataset_path: Optional[str] = None
    wind_dataset_path: Optional[str] = None
    current_dataset_paths: Optional[Sequence[str]] = None
    wind_dataset_paths: Optional[Sequence[str]] = None
    environmental_offset_hours: Optional[float] = None
    use_sal_temp: bool = True
    skip_animation: bool = True


@dataclass(frozen=True)
class DeterministicRunResult:
    run_name: str
    out_dir: Path
    sim_path: Path
    lon: float
    lat: float
    start_time: datetime
    end_time: datetime
    artifact_paths: Tuple[Path, ...] = ()


@dataclass(frozen=True)
class StochasticValidationRunRequest:
    stochastic_config: Any
    config_name: str = "main"
    environment: str = "2019"
    shp_zip: Optional[str] = None
    min_long: Optional[float] = None
    max_long: Optional[float] = None
    min_lat: Optional[float] = None
    max_lat: Optional[float] = None
    start_index: int = 1
    padding_animation_frame: float = 0.1
    run_name: Optional[str] = None
    forcing_source: str = "COPERNICUS"
    current_dataset_path: Optional[str] = None
    wind_dataset_path: Optional[str] = None
    current_dataset_paths: Optional[Sequence[str]] = None
    wind_dataset_paths: Optional[Sequence[str]] = None
    environmental_offset_hours: Optional[float] = None
    disable_environment_offset: bool = False
    oil_types: Optional[str] = None
    oil_types_file: Optional[str] = None
    processes_dispersion: Optional[str] = None
    processes_evaporation: Optional[str] = None


@dataclass(frozen=True)
class ValidationRunResult:
    run_name: str
    out_dir: Path
    sim_path: Path
    wind_drift_factor: Optional[float] = None
    wave_effects_enabled: bool = False
    current_drift_factor: Optional[float] = None
    environmental_offset_hours: Optional[float] = None
    oil_type: Optional[str] = None
    comparison_gif: Optional[Path] = None
    frames_dir: Optional[Path] = None
    artifact_paths: Tuple[Path, ...] = ()


# --------------------------------------------------------------------------
# Configuracao do ensemble estocastico
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StochasticParameterConfig:
    enabled: bool
    mean: float
    std: float
    min_value: float
    max_value: float
    distribution: str = "normal"
    default_value: Optional[float] = None


@dataclass(frozen=True)
class TemporalLagConfig:
    enabled: bool
    mean: float
    std: float
    min_value: float
    max_value: float
    input_unit: str = "seconds"
    rounding_granularity: str = "seconds"
    distribution: str = "normal"
    default_seconds: float = 0.0


@dataclass(frozen=True)
class StochasticGridConfig:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    spatial_resolution: float
    margin: float = 0.0
    crs: str = "EPSG:4326"


@dataclass(frozen=True)
class StochasticRunConfig:
    run_name: str
    n_simulations: int
    seed: Optional[int]
    cdf: StochasticParameterConfig
    wdf: StochasticParameterConfig
    temporal_lag: TemporalLagConfig
    grid: StochasticGridConfig
    output_root: Path
    execution_mode: str = "deterministic_ensemble"
    number_of_workers: int = 4


@dataclass(frozen=True)
class SampledParameterSet:
    simulation_id: int
    seed: int
    cdf: float
    wdf: float
    temporal_lag_original_value: float
    temporal_lag_input_unit: str
    temporal_lag_rounding_granularity: str
    temporal_lag_seconds: float
    status: str = "pending"
    error_message: str = ""
    output_path: str = ""


@dataclass(frozen=True)
class StochasticRunResult:
    run_name: str
    total_simulations: int
    successful_simulations: int
    failed_simulations: int
    output_path: Path
    sampled_parameters_path: Path
    summary_path: Path
    hit_count_map_path: Optional[Path] = None
    probability_map_path: Optional[Path] = None
    hit_count_final_timestep_map_path: Optional[Path] = None
    probability_final_timestep_map_path: Optional[Path] = None
    hourly_probability_map_paths: Tuple[Path, ...] = ()
