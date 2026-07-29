"""Orquestracao das execucoes: validacao, otimizacao, deriva por ponto e ensemble."""

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from opendrift.models.openoil import OpenOil

from src.inputs import forcing
from src.inputs.config import (
    ConfigRequest,
    DeterministicRunRequest,
    DeterministicRunResult,
    EnvironmentRepository,
    ObservedSpillContext,
    ObservedSpillRequest,
    StochasticValidationRunRequest,
    ValidationRunRequest,
    ValidationRunResult,
)
from src.simulation.drift import SimulationService
from src.stochastic.runner import StochasticSimulationService
from src.inputs.forcing import CopernicusGateway
from src.simulation.optimize import OptimizationService
from src.outputs.artifacts import WorkspaceRepository
from src.outputs.plots import generate_comparison_gif, render_comparison_frames
from src.inputs.spills import SpillRepository

# Meia-largura da caixa de forcing em volta do ponto numa simulacao
# deterministica: espaco suficiente para a mancha derivar por alguns dias.
DETERMINISTIC_AREA_HALF_SPAN_DEG = 2.0


class _ProgressPrinter:
    def __init__(
        self,
        total,
        every=15,
        label="Progress",
        on_tick: Optional[Callable[[int, int], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ):
        self.total = int(total)
        self.every = int(every)
        self.label = label
        self.count = 0
        self.on_tick = on_tick
        self.should_cancel = should_cancel

    def tick(self, n=1):
        if self.should_cancel is not None and self.should_cancel():
            raise RuntimeError("Execution cancelled by user.")
        if self.total <= 0:
            return
        self.count += int(n)
        if self.on_tick is not None:
            self.on_tick(self.count, self.total)
        if self.count % self.every == 0 or self.count >= self.total:
            print(f"{self.label}: {self.count}/{self.total}")


@dataclass
class _ValidationContext:
    config: Any
    offset_hours: float
    environmental_offset_hours: float
    forcing_source: str
    current_dataset_path: Path
    wind_dataset_path: Optional[Path]
    current_dataset_paths: tuple[Path, ...]
    wind_dataset_paths: tuple[Path, ...]
    real_manchas: Any
    plot_bounds: tuple[float, float, float, float]
    observed_trajectory: Any
    environmental_offset_values: Optional[tuple[float, ...]]
    start_index: int
    skip_animation: bool
    skip_plots: bool
    wave_effects_enabled: bool
    wind_drift_factor: Optional[float]
    current_drift_factor: Optional[float]
    processes_dispersion: Optional[bool]
    processes_evaporation: Optional[bool]
    selected_oil_type: Optional[str]


class SimulationController:
    """Conduz uma execucao de ponta a ponta.

    Monta o contexto (config, manchas, forcing), opcionalmente calibra os
    fatores de deriva, roda a simulacao e gera as figuras. O modo estocastico
    reaproveita o mesmo contexto e delega ao ensemble.
    """

    def __init__(
        self,
        spill_repository: Optional[SpillRepository] = None,
        environment_repository: Optional[EnvironmentRepository] = None,
        workspace_repository: Optional[WorkspaceRepository] = None,
        copernicus_gateway: Optional[CopernicusGateway] = None,
        optimization_service: Optional[OptimizationService] = None,
        simulation_service: Optional[SimulationService] = None,
        stochastic_service: Optional[StochasticSimulationService] = None,
    ) -> None:
        self.spill_repository = spill_repository or SpillRepository()
        self.environment_repository = environment_repository or EnvironmentRepository()
        self.workspace_repository = workspace_repository or WorkspaceRepository()
        self.copernicus_gateway = copernicus_gateway or CopernicusGateway()
        self.optimization_service = optimization_service or OptimizationService(
            spill_repository=self.spill_repository
        )
        self.simulation_service = simulation_service or SimulationService(
            spill_repository=self.spill_repository
        )
        self.stochastic_service = stochastic_service or StochasticSimulationService(
            simulation_service=self.simulation_service
        )

    # ----------------------------------------------------------------- utils

    @staticmethod
    def _parse_bool_string(value):
        if value is None:
            return None
        parsed = value.strip().lower()
        mapping = {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        }
        if parsed not in mapping:
            raise ValueError(f"Invalid boolean value: {value}")
        return mapping[parsed]

    @staticmethod
    def _load_oil_types(oil_types, oil_types_file):
        def normalize_name(name):
            return " ".join(name.strip().split()).lower()

        selected = []
        if oil_types_file:
            file_path = Path(oil_types_file)
            if not file_path.exists():
                raise ValueError(f"oil types file not found: {file_path}")
            selected.extend(
                [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            )
        if oil_types:
            selected.extend([item.strip() for item in oil_types.split(",") if item.strip()])

        if not selected:
            return []

        available = OpenOil(loglevel=50).oiltypes
        by_normalized = {normalize_name(name): name for name in available}
        normalized = []
        invalid = []
        for name in selected:
            mapped = by_normalized.get(normalize_name(name))
            if mapped:
                normalized.append(mapped)
                continue
            invalid.append(name)

        if invalid:
            raise ValueError(
                "Unknown oil types: "
                + ", ".join(invalid[:10])
                + (" ..." if len(invalid) > 10 else "")
            )

        return list(dict.fromkeys(normalized))

    @staticmethod
    def _build_sim_filename(
        base_name,
        wind_drift_factor=None,
        wave_effects_enabled=None,
        current_drift_factor=None,
        processes_dispersion=None,
        processes_evaporation=None,
    ):
        sim_filename = base_name
        if wind_drift_factor is not None:
            safe_wdf = f"{wind_drift_factor:.4f}".replace(".", "p")
            sim_filename = f"{sim_filename}_wdf{safe_wdf}"
        if wave_effects_enabled is not None:
            sim_filename = f"{sim_filename}_{'waves' if wave_effects_enabled else 'nowaves'}"
        if current_drift_factor is not None:
            safe_cdf = f"{current_drift_factor:.2f}".replace(".", "p")
            sim_filename = f"{sim_filename}_cdf{safe_cdf}"
        if processes_dispersion is not None:
            sim_filename = f"{sim_filename}_{'disp' if processes_dispersion else 'nodisp'}"
        if processes_evaporation is not None:
            sim_filename = f"{sim_filename}_{'evap' if processes_evaporation else 'noevap'}"
        return sim_filename

    @staticmethod
    def _check_cancelled(should_cancel: Optional[Callable[[], bool]]) -> None:
        if should_cancel is not None and should_cancel():
            raise RuntimeError("Execution cancelled by user.")

    # --------------------------------------------------------------- entradas

    def load_config(self, request: ConfigRequest):
        return self.environment_repository.compose_config(
            config_name=request.config_name,
            environment=request.environment,
            additional_overrides=request.to_overrides(),
        )

    def download_environment_data(
        self,
        environment: str,
        config_name: str = "main",
        force: bool = False,
        log_callback: Optional[Callable[[str], None]] = None,
        copernicus_username: Optional[str] = None,
        copernicus_password: Optional[str] = None,
        min_long: Optional[float] = None,
        max_long: Optional[float] = None,
        min_lat: Optional[float] = None,
        max_lat: Optional[float] = None,
    ) -> dict:
        config = self.load_config(
            ConfigRequest(
                config_name=config_name,
                environment=environment,
                simulation_name="sim4validation",
                min_long=min_long,
                max_long=max_long,
                min_lat=min_lat,
                max_lat=max_lat,
            )
        )
        return self.copernicus_gateway.download_environment_data(
            config=config,
            force=force,
            log_callback=log_callback,
            username=copernicus_username,
            password=copernicus_password,
        )

    def load_observed_spills(self, request: ObservedSpillRequest) -> ObservedSpillContext:
        manchas = gpd.read_file(Path(request.spill_path)).to_crs(epsg=4326)
        self.spill_repository.ensure_datetime_column(manchas, offset_hours=request.offset_hours)
        manchas.sort_values("datetime", inplace=True)

        unique_times = sorted(manchas["datetime"].unique())
        if request.start_index < 0 or request.start_index >= len(unique_times):
            raise ValueError(f"start-index out of range (0..{len(unique_times)-1})")
        if request.start_index:
            start_time = unique_times[request.start_index]
            manchas = manchas[manchas["datetime"] >= start_time].copy()

        minlon, minlat, maxlon, maxlat = manchas.total_bounds
        pad_lon = (maxlon - minlon) * request.padding_animation_frame
        pad_lat = (maxlat - minlat) * request.padding_animation_frame
        plot_bounds = (
            minlon - pad_lon,
            maxlon + pad_lon,
            minlat - pad_lat,
            maxlat + pad_lat,
        )
        return ObservedSpillContext(manchas=manchas, plot_bounds=plot_bounds)

    def _build_validation_context(self, request: ValidationRunRequest) -> _ValidationContext:
        skip_animation = request.skip_animation
        skip_plots = request.skip_plots
        if request.evaluation:
            skip_animation = True
            skip_plots = True

        config = self.load_config(
            ConfigRequest(
                config_name=request.config_name,
                environment=request.environment,
                simulation_name="sim4validation",
                run_name=request.run_name,
                shp_zip=request.shp_zip,
                min_long=request.min_long,
                max_long=request.max_long,
                min_lat=request.min_lat,
                max_lat=request.max_lat,
            )
        )
        forcing_source = forcing.normalize_forcing_source(request.forcing_source)
        offset_hours = (
            0.0
            if request.disable_environment_offset
            else float(getattr(config.copernicusmarine.specificities, "datetime_offset_hours", 0) or 0.0)
        )
        environmental_offset_hours = (
            forcing.DEFAULT_ENVIRONMENTAL_OFFSET_HOURS
            if request.environmental_offset_hours is None
            else float(request.environmental_offset_hours)
        )
        if abs(environmental_offset_hours) > forcing.MAX_ENVIRONMENTAL_OFFSET_HOURS:
            raise ValueError(
                f"environmental-offset-hours must be between "
                f"-{forcing.MAX_ENVIRONMENTAL_OFFSET_HOURS:g} and "
                f"{forcing.MAX_ENVIRONMENTAL_OFFSET_HOURS:g}"
            )
        environmental_offset_values = forcing.normalize_environmental_offset_values(
            request.environmental_offset_values
        )
        current_paths = forcing.normalize_path_list(
            request.current_dataset_path,
            list(request.current_dataset_paths) if request.current_dataset_paths else None,
        )
        if not current_paths:
            current_paths = [Path(config.copernicusmarine.specificities.water_dataset_path)]

        wind_paths = forcing.normalize_path_list(
            request.wind_dataset_path,
            list(request.wind_dataset_paths) if request.wind_dataset_paths else None,
        )
        if not wind_paths:
            config_wind_path = getattr(config.copernicusmarine.specificities, "wind_dataset_path", None)
            if config_wind_path:
                wind_paths = [Path(config_wind_path)]
        sal_temp_dataset_path = Path(config.copernicusmarine.specificities.sal_temp_dataset_path)
        observed_context = self.load_observed_spills(
            ObservedSpillRequest(
                spill_path=Path(config.paths.plataformas_shp),
                offset_hours=offset_hours,
                start_index=request.start_index,
                padding_animation_frame=request.padding_animation_frame,
            )
        )
        if environmental_offset_values is None:
            forcing.validate_environment_coverage(
                observed_context.manchas,
                current_dataset_paths=tuple(current_paths),
                wind_dataset_paths=tuple(wind_paths),
                sal_temp_dataset_path=sal_temp_dataset_path,
                environmental_offset_hours=environmental_offset_hours,
            )
        else:
            environmental_offset_values = forcing.valid_environmental_offsets(
                observed_context.manchas,
                current_dataset_paths=tuple(current_paths),
                wind_dataset_paths=tuple(wind_paths),
                sal_temp_dataset_path=sal_temp_dataset_path,
                environmental_offset_values=environmental_offset_values,
            )
            environmental_offset_hours = environmental_offset_values[0]

        selected_oil_types = self._load_oil_types(request.oil_types, request.oil_types_file)
        selected_oil_type = selected_oil_types[0] if selected_oil_types else None

        return _ValidationContext(
            config=config,
            offset_hours=offset_hours,
            environmental_offset_hours=environmental_offset_hours,
            forcing_source=forcing_source,
            current_dataset_path=current_paths[0],
            wind_dataset_path=wind_paths[0] if wind_paths else None,
            current_dataset_paths=tuple(current_paths),
            wind_dataset_paths=tuple(wind_paths),
            real_manchas=observed_context.manchas,
            plot_bounds=observed_context.plot_bounds,
            observed_trajectory=self.spill_repository.build_observed_trajectory(
                observed_context.manchas
            ),
            environmental_offset_values=environmental_offset_values,
            start_index=int(request.start_index),
            skip_animation=skip_animation,
            skip_plots=skip_plots,
            wave_effects_enabled=False,
            wind_drift_factor=request.wind_drift_factor,
            current_drift_factor=request.current_drift_factor,
            processes_dispersion=self._parse_bool_string(request.processes_dispersion),
            processes_evaporation=self._parse_bool_string(request.processes_evaporation),
            selected_oil_type=selected_oil_type,
        )

    # ------------------------------------------------------------------ fases

    def _run_fast_optimization_phase(
        self,
        request: ValidationRunRequest,
        context: _ValidationContext,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> bool:
        if not request.optimize_wdf_cdf:
            return True
        self._check_cancelled(should_cancel)

        if context.wind_drift_factor is not None:
            print("Ignoring --wind-drift-factor because --optimize-wdf-cdf is set.")
        if context.current_drift_factor is not None:
            print("Ignoring --current-drift-factor because --optimize-wdf-cdf is set.")
        if request.wdf_step <= 0:
            raise ValueError("wdf-step must be > 0")
        if request.wdf_max < request.wdf_min:
            raise ValueError("wdf-max must be >= wdf-min")
        if request.cdf_step <= 0:
            raise ValueError("cdf-step must be > 0")
        if request.cdf_max < request.cdf_min:
            raise ValueError("cdf-max must be >= cdf-min")

        wdf_values = np.arange(
            request.wdf_min,
            request.wdf_max + (request.wdf_step / 2),
            request.wdf_step,
        )
        cdf_values = np.arange(
            request.cdf_min,
            request.cdf_max + (request.cdf_step / 2),
            request.cdf_step,
        )
        out_dir = self.workspace_repository.simulation_output_dir(context.config)

        environmental_offset_values = (
            context.environmental_offset_values
            if context.environmental_offset_values is not None
            else (float(context.environmental_offset_hours),)
        )
        total_runs = len(environmental_offset_values) * len(cdf_values)
        print(
            f"Will test {len(environmental_offset_values)} environmental offsets x "
            f"{len(cdf_values)} cdf = {total_runs} simulations "
            f"(fast; all WDFs per run, waves disabled)."
        )
        optimization_time_step_minutes = max(
            5.0,
            float(getattr(context.config.simulation, "time_step_minutes", 1.0) or 1.0),
        )
        print(
            f"Optimization OpenDrift timestep: {optimization_time_step_minutes:g} min "
            f"(final simulation keeps {context.config.simulation.time_step_minutes:g} min)."
        )
        progress = _ProgressPrinter(
            total_runs,
            every=15,
            label="Progress",
            on_tick=progress_callback,
            should_cancel=should_cancel,
        )
        results_frames = []
        for environmental_offset_hours in environmental_offset_values:
            self._check_cancelled(should_cancel)
            print(f"Testing environmental offset {environmental_offset_hours:g} h...")
            _, offset_results_df = self.optimization_service.fast_grid_search_wdf_cdf(
                manchas=context.real_manchas,
                config=context.config,
                observed_trajectory=context.observed_trajectory,
                wdf_values=wdf_values,
                current_drift_values=cdf_values,
                particles_per_wdf=request.fast_particles_per_wdf,
                oil_type=context.selected_oil_type,
                progress=progress,
                should_cancel=should_cancel,
                forcing_source=context.forcing_source,
                current_dataset_path=str(context.current_dataset_path),
                wind_dataset_path=(str(context.wind_dataset_path) if context.wind_dataset_path else None),
                current_dataset_paths=[str(path) for path in context.current_dataset_paths],
                wind_dataset_paths=[str(path) for path in context.wind_dataset_paths],
                environmental_offset_hours=environmental_offset_hours,
            )
            if not offset_results_df.empty:
                offset_results_df = offset_results_df.copy()
                offset_results_df["environmental_offset_hours"] = float(environmental_offset_hours)
                results_frames.append(offset_results_df)

        results_df = (
            pd.concat(results_frames, ignore_index=True)
            if results_frames
            else pd.DataFrame(
                columns=[
                    "wind_drift_factor",
                    "skillscore",
                    "current_drift_factor",
                    "environmental_offset_hours",
                ]
            )
        )
        best_row = None
        if not results_df.empty and results_df["skillscore"].notna().any():
            best_row = results_df.loc[results_df["skillscore"].idxmax()]

        results_name = "wdf_cdf_optimization_fast"
        self.workspace_repository.write_csv(out_dir / f"{results_name}.csv", results_df)
        if best_row is None or pd.isna(best_row["skillscore"]):
            print("Combined optimization failed: no valid skillscore computed.")
            return False

        context.wind_drift_factor = float(best_row["wind_drift_factor"])
        context.current_drift_factor = float(best_row["current_drift_factor"])
        context.environmental_offset_hours = float(best_row["environmental_offset_hours"])
        if "oil_type" in best_row:
            context.selected_oil_type = str(best_row["oil_type"])

        summary = {
            "wind_drift_factor": context.wind_drift_factor,
            "wave_effects_enabled": False,
            "current_drift_factor": context.current_drift_factor,
            "environmental_offset_hours": context.environmental_offset_hours,
            "skillscore": float(best_row["skillscore"]),
            "wdf_min": float(request.wdf_min),
            "wdf_max": float(request.wdf_max),
            "wdf_step": float(request.wdf_step),
            "cdf_min": float(request.cdf_min),
            "cdf_max": float(request.cdf_max),
            "cdf_step": float(request.cdf_step),
            "environmental_offset_values": [
                float(value) for value in environmental_offset_values
            ],
            "particles_per_wdf": int(request.fast_particles_per_wdf),
        }
        if context.selected_oil_type:
            summary["oil_type"] = context.selected_oil_type
        self.workspace_repository.write_json(out_dir / f"{results_name}.json", summary)
        print(
            f"Best wdf/cdf: {context.wind_drift_factor:.4f} "
            f"cdf={context.current_drift_factor:.2f} "
            f"env_offset={context.environmental_offset_hours:+g}h "
            f"oil={context.selected_oil_type or 'default'} "
            f"(skillscore {best_row['skillscore']:.4f})"
        )
        return True

    def _run_simulation_phase(
        self,
        request: ValidationRunRequest,
        context: _ValidationContext,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> tuple[Path, Path]:
        print("Start simulation...")
        self._check_cancelled(should_cancel)

        sim_filename = self._build_sim_filename(
            "sim_2019_P53_TEST_NOWAVES_30WDF_75CDF",
            wind_drift_factor=context.wind_drift_factor,
            wave_effects_enabled=context.wave_effects_enabled,
            current_drift_factor=context.current_drift_factor,
            processes_dispersion=context.processes_dispersion,
            processes_evaporation=context.processes_evaporation,
        )

        out_dir = self.workspace_repository.simulation_output_dir(context.config)
        run_params = {
            "environment": request.environment,
            "simulation_name": context.config.simulation.name,
            "forcing_source": context.forcing_source,
            "start_index": int(request.start_index),
            "observed_start_timestep": int(request.start_index),
            "environmental_offset_hours": float(context.environmental_offset_hours),
            "wave_effects_enabled": False,
            "current_dataset_path": str(context.current_dataset_path),
            "current_dataset_paths": [str(path) for path in context.current_dataset_paths],
            "observed_offset_hours": float(context.offset_hours),
        }
        if context.wind_drift_factor is not None:
            run_params["wind_drift_factor"] = float(context.wind_drift_factor)
        if context.current_drift_factor is not None:
            run_params["current_drift_factor"] = float(context.current_drift_factor)
        if context.wind_dataset_path is not None:
            run_params["wind_dataset_path"] = str(context.wind_dataset_path)
        if context.wind_dataset_paths:
            run_params["wind_dataset_paths"] = [str(path) for path in context.wind_dataset_paths]
        if context.selected_oil_type:
            run_params["oil_type"] = context.selected_oil_type
        if context.processes_dispersion is not None:
            run_params["processes_dispersion"] = bool(context.processes_dispersion)
        if context.processes_evaporation is not None:
            run_params["processes_evaporation"] = bool(context.processes_evaporation)
        self.workspace_repository.write_json(out_dir / f"{sim_filename}.json", run_params)

        if not request.skip_simulation:
            self._check_cancelled(should_cancel)
            self.simulation_service.simulate_drift(
                manchas=context.real_manchas,
                out_filename=sim_filename,
                config=context.config,
                skip_animation=context.skip_animation,
                padding_animation_frame=request.padding_animation_frame,
                wind_drift_factor=context.wind_drift_factor,
                current_drift_factor=context.current_drift_factor,
                oil_type=context.selected_oil_type,
                processes_dispersion=context.processes_dispersion,
                processes_evaporation=context.processes_evaporation,
                forcing_source=context.forcing_source,
                current_dataset_path=str(context.current_dataset_path),
                wind_dataset_path=(str(context.wind_dataset_path) if context.wind_dataset_path else None),
                current_dataset_paths=[str(path) for path in context.current_dataset_paths],
                wind_dataset_paths=[str(path) for path in context.wind_dataset_paths],
                observed_offset_hours=float(context.offset_hours),
                environmental_offset_hours=float(context.environmental_offset_hours),
            )
            print(f"The results have been generated in {out_dir}")
        else:
            print(f"The results are probably already present in {out_dir}")

        return out_dir / f"{sim_filename}.nc", out_dir

    def _run_visualization_phase(
        self,
        context: _ValidationContext,
        sim_path: Path,
        out_dir: Path,
        should_cancel: Optional[Callable[[], bool]] = None,
        show_plots: bool = True,
    ) -> tuple[Optional[Path], Optional[Path]]:
        self._check_cancelled(should_cancel)
        compare_gif: Optional[Path] = None
        frames_dir: Optional[Path] = None

        if not context.skip_animation:
            compare_gif = out_dir / f"{sim_path.stem}_compare.gif"
            try:
                generate_comparison_gif(
                    sim_nc=sim_path,
                    shp_zip=context.config.paths.plataformas_shp,
                    out=compare_gif,
                    extent=",".join(f"{value:.6f}" for value in context.plot_bounds),
                    datetime_offset_hours=context.offset_hours,
                    real_steps=int(context.real_manchas["datetime"].nunique()),
                    start_index=int(context.start_index),
                )
            except Exception as exc:
                print(f"Failed to generate comparison GIF: {exc}")

        if context.skip_plots:
            return compare_gif, frames_dir

        frames_dir = render_comparison_frames(
            manchas=context.real_manchas,
            observed_trajectory=context.observed_trajectory,
            sim_path=sim_path,
            out_dir=out_dir,
            show_plots=show_plots,
            should_cancel=should_cancel,
        )
        return compare_gif, frames_dir

    # ------------------------------------------------------------ execucoes

    def run_validation(
        self,
        request: ValidationRunRequest,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
        show_plots: bool = True,
    ) -> Optional[ValidationRunResult]:
        self._check_cancelled(should_cancel)
        context = self._build_validation_context(request)
        if not self._run_fast_optimization_phase(
            request,
            context,
            progress_callback=progress_callback,
            should_cancel=should_cancel,
        ):
            return None
        sim_path, out_dir = self._run_simulation_phase(request, context, should_cancel=should_cancel)
        compare_gif, frames_dir = self._run_visualization_phase(
            context,
            sim_path,
            out_dir,
            should_cancel=should_cancel,
            show_plots=show_plots,
        )
        artifact_paths = tuple(
            sorted((path for path in out_dir.iterdir() if path.is_file()), key=lambda p: p.name)
        )
        return ValidationRunResult(
            run_name=context.config.simulation.name,
            out_dir=out_dir,
            sim_path=sim_path,
            wind_drift_factor=context.wind_drift_factor,
            wave_effects_enabled=False,
            current_drift_factor=context.current_drift_factor,
            environmental_offset_hours=context.environmental_offset_hours,
            oil_type=context.selected_oil_type,
            comparison_gif=compare_gif,
            frames_dir=frames_dir,
            artifact_paths=artifact_paths,
        )

    def run_deterministic(
        self,
        request: DeterministicRunRequest,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> DeterministicRunResult:
        """Roda uma simulacao a partir de um ponto, data e duracao informados.

        Nao passa pelo contexto de validacao: sem manchas observadas nao ha o
        que validar contra. A area de forcing e a caixa em volta do ponto.
        """
        self._check_cancelled(should_cancel)

        run_name = request.run_name or (
            f"deterministic_{request.start_time.strftime('%Y%m%dT%H%M')}"
        )
        # Caixa em volta do ponto: define a area de recorte do Copernicus e da
        # margem suficiente para a mancha derivar sem sair do dominio.
        half_span = DETERMINISTIC_AREA_HALF_SPAN_DEG
        config = self.load_config(
            ConfigRequest(
                config_name=request.config_name,
                environment=request.environment,
                simulation_name="sim4validation",
                run_name=run_name,
                min_long=request.lon - half_span,
                max_long=request.lon + half_span,
                min_lat=request.lat - half_span,
                max_lat=request.lat + half_span,
            )
        )
        forcing_source = forcing.normalize_forcing_source(request.forcing_source)
        environmental_offset_hours = (
            forcing.DEFAULT_ENVIRONMENTAL_OFFSET_HOURS
            if request.environmental_offset_hours is None
            else float(request.environmental_offset_hours)
        )

        oil_types = self._load_oil_types(request.oil_type, None)
        out_dir = self.workspace_repository.simulation_output_dir(config)
        end_time = request.start_time + timedelta(days=float(request.duration_days))
        out_filename = (
            f"{run_name}_{request.start_time.strftime('%Y%m%dT%H%M')}"
            f"_{end_time.strftime('%Y%m%dT%H%M')}"
        )

        self.workspace_repository.write_json(
            out_dir / f"{out_filename}.json",
            {
                "mode": "deterministic",
                "lon": float(request.lon),
                "lat": float(request.lat),
                "start_time": request.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_days": float(request.duration_days),
                "leak_duration_hours": float(request.leak_duration_hours),
                "seed_radius_m": float(request.seed_radius_m),
                "num_seed_elements": int(request.num_seed_elements),
                "time_step_minutes": float(request.time_step_minutes),
                "output_time_step_minutes": float(request.output_time_step_minutes),
                "environment": request.environment,
                "forcing_source": forcing_source,
                "environmental_offset_hours": environmental_offset_hours,
                "wind_drift_factor": request.wind_drift_factor,
                "current_drift_factor": request.current_drift_factor,
                "oil_type": oil_types[0] if oil_types else None,
                "current_dataset_paths": list(request.current_dataset_paths or []),
                "wind_dataset_paths": list(request.wind_dataset_paths or []),
            },
        )

        self._check_cancelled(should_cancel)
        sim_path = self.simulation_service.simulate_point_drift(
            config=config,
            lon=request.lon,
            lat=request.lat,
            seed_time_start=request.start_time,
            duration_days=request.duration_days,
            out_filename=out_filename,
            leak_duration_hours=request.leak_duration_hours,
            seed_radius_m=request.seed_radius_m,
            num_seed_elements=request.num_seed_elements,
            oil_type=oil_types[0] if oil_types else None,
            wind_drift_factor=request.wind_drift_factor,
            current_drift_factor=request.current_drift_factor,
            processes_dispersion=self._parse_bool_string(request.processes_dispersion),
            processes_evaporation=self._parse_bool_string(request.processes_evaporation),
            time_step_minutes=request.time_step_minutes,
            output_time_step_minutes=request.output_time_step_minutes,
            forcing_source=forcing_source,
            current_dataset_path=request.current_dataset_path,
            wind_dataset_path=request.wind_dataset_path,
            current_dataset_paths=request.current_dataset_paths,
            wind_dataset_paths=request.wind_dataset_paths,
            environmental_offset_hours=environmental_offset_hours,
            use_sal_temp=request.use_sal_temp,
            skip_animation=request.skip_animation,
        )

        artifact_paths = tuple(
            sorted((path for path in out_dir.iterdir() if path.is_file()), key=lambda p: p.name)
        )
        return DeterministicRunResult(
            run_name=run_name,
            out_dir=out_dir,
            sim_path=sim_path,
            lon=float(request.lon),
            lat=float(request.lat),
            start_time=request.start_time,
            end_time=end_time,
            artifact_paths=artifact_paths,
        )

    def run_stochastic_validation(
        self,
        request: StochasticValidationRunRequest,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        validation_request = ValidationRunRequest(
            config_name=request.config_name,
            environment=request.environment,
            shp_zip=request.shp_zip,
            min_long=request.min_long,
            max_long=request.max_long,
            min_lat=request.min_lat,
            max_lat=request.max_lat,
            start_index=request.start_index,
            padding_animation_frame=request.padding_animation_frame,
            run_name=request.run_name or request.stochastic_config.run_name,
            forcing_source=request.forcing_source,
            current_dataset_path=request.current_dataset_path,
            wind_dataset_path=request.wind_dataset_path,
            current_dataset_paths=request.current_dataset_paths,
            wind_dataset_paths=request.wind_dataset_paths,
            environmental_offset_hours=request.environmental_offset_hours,
            disable_environment_offset=request.disable_environment_offset,
            oil_types=request.oil_types,
            oil_types_file=request.oil_types_file,
            processes_dispersion=request.processes_dispersion,
            processes_evaporation=request.processes_evaporation,
            skip_animation=True,
            skip_plots=True,
        )
        self._check_cancelled(should_cancel)
        context = self._build_validation_context(validation_request)
        return self.stochastic_service.run(
            manchas=context.real_manchas,
            base_config=context.config,
            stochastic_config=request.stochastic_config,
            padding_animation_frame=request.padding_animation_frame,
            forcing_source=context.forcing_source,
            current_dataset_paths=[str(path) for path in context.current_dataset_paths],
            wind_dataset_paths=[str(path) for path in context.wind_dataset_paths],
            observed_offset_hours=float(context.offset_hours),
            environmental_offset_hours=float(context.environmental_offset_hours),
            oil_type=context.selected_oil_type,
            processes_dispersion=context.processes_dispersion,
            processes_evaporation=context.processes_evaporation,
            progress_callback=progress_callback,
            should_cancel=should_cancel,
            log_callback=log_callback,
        )
