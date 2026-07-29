"""Ensemble estocastico: roda N membros deterministicos e agrega o resultado.

Cada membro e uma simulacao completa com CDF, WDF e lag temporal proprios;
o conjunto vira mapas de probabilidade de presenca de oleo.
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from src.inputs.config import SampledParameterSet, StochasticRunConfig, StochasticRunResult
from src.simulation.drift import SimulationService
from src.outputs.artifacts import StochasticOutputService
from src.stochastic.raster import DriftRasterConverter, EnsembleAggregationResult, EnsembleAggregator
from src.stochastic.sampling import ParameterSampler


def _build_member_config(base_config, simulation_data_root: Path, simulation_name: str):
    """Copia a config base redirecionando a saida para o diretorio do membro.

    A copia e profunda porque os membros rodam em processos distintos e cada
    um reescreve simulation.name e o caminho de saida.
    """
    member_config = copy.deepcopy(base_config)
    member_config.paths.simulation_data = str(simulation_data_root)
    member_config.simulation.name = simulation_name
    return member_config


def _normalize_simulation_output_timeline(
    output_path: Path,
    expected_output_times: Sequence[object],
) -> None:
    """Reindexa o .nc do membro na timeline comum do ensemble.

    Membros podem terminar com timesteps ligeiramente diferentes (o lag
    temporal desloca o forcing). Reindexar garante que a agregacao por hora
    compare o mesmo instante entre todos: horas faltantes viram NaN em vez de
    deslocar as demais.
    """
    output_path = Path(output_path)
    expected_index = pd.DatetimeIndex(pd.to_datetime(list(expected_output_times)))
    if expected_index.empty:
        return

    tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    normalized = None
    try:
        with xr.open_dataset(output_path) as dataset:
            normalized = dataset.reindex(time=expected_index.values).load()
            normalized.attrs = dataset.attrs.copy()
            normalized.attrs["time_coverage_start"] = str(expected_index[0].to_pydatetime())
            normalized.attrs["time_coverage_end"] = str(expected_index[-1].to_pydatetime())
        normalized.to_netcdf(tmp_path)
        tmp_path.replace(output_path)
    finally:
        if normalized is not None:
            normalized.close()
        tmp_path.unlink(missing_ok=True)


def _run_stochastic_member_worker(payload: dict) -> SampledParameterSet:
    """Roda um membro do ensemble e devolve o sample com status preenchido.

    Precisa ser funcao de modulo (nao metodo) para ser serializavel pelo
    ProcessPoolExecutor com contexto spawn. Falhas viram status="failed" em vez
    de excecao: um membro quebrado nao deve derrubar o ensemble inteiro.
    """
    sample: SampledParameterSet = payload["sample"]
    run_name = f"run_{sample.simulation_id:04d}"
    member_config = _build_member_config(
        base_config=payload["base_config"],
        simulation_data_root=Path(payload["individual_runs_dir"]),
        simulation_name=run_name,
    )
    output_dir = Path(payload["individual_runs_dir"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    out_filename = "simulation"
    output_path = output_dir / f"{out_filename}.nc"

    try:
        SimulationService().simulate_drift(
            manchas=payload["manchas"].copy(),
            out_filename=out_filename,
            config=member_config,
            skip_animation=True,
            padding_animation_frame=payload["padding_animation_frame"],
            wind_drift_factor=sample.wdf,
            current_drift_factor=sample.cdf,
            oil_type=payload["oil_type"],
            processes_dispersion=payload["processes_dispersion"],
            processes_evaporation=payload["processes_evaporation"],
            forcing_source=payload["forcing_source"],
            current_dataset_paths=payload["current_dataset_paths"],
            wind_dataset_paths=payload["wind_dataset_paths"],
            observed_offset_hours=payload["observed_offset_hours"],
            environmental_offset_hours=payload["environmental_offset_hours"],
            temporal_lag_seconds=sample.temporal_lag_seconds,
        )

        if not output_path.exists():
            raise FileNotFoundError(f"Simulation output was not created: {output_path}")
        _normalize_simulation_output_timeline(
            output_path=output_path,
            expected_output_times=payload["expected_output_times"],
        )
        return replace(sample, status="success", output_path=str(output_path))
    except Exception as exc:
        return replace(sample, status="failed", error_message=str(exc), output_path=str(output_path))


class StochasticSimulationService:
    """Executa o ensemble Monte Carlo como N simulacoes deterministicas.

    A incerteza nao entra dentro do modelo: cada membro e uma rodada normal do
    OpenDrift com um sorteio proprio de CDF, WDF e lag temporal. A distribuicao
    dos resultados e que vira o mapa de probabilidade.
    """

    def __init__(
        self,
        simulation_service: Optional[SimulationService] = None,
        parameter_sampler: Optional[ParameterSampler] = None,
        converter: Optional[DriftRasterConverter] = None,
        aggregator: Optional[EnsembleAggregator] = None,
        output_service: Optional[StochasticOutputService] = None,
    ) -> None:
        self.simulation_service = simulation_service or SimulationService()
        self.parameter_sampler = parameter_sampler or ParameterSampler()
        self.converter = converter or DriftRasterConverter()
        self.aggregator = aggregator or EnsembleAggregator(self.converter)
        self.output_service = output_service or StochasticOutputService()

    def run(
        self,
        manchas,
        base_config,
        stochastic_config: StochasticRunConfig,
        padding_animation_frame: float,
        forcing_source: str = "COPERNICUS",
        current_dataset_paths: Optional[Sequence[str]] = None,
        wind_dataset_paths: Optional[Sequence[str]] = None,
        observed_offset_hours: Optional[float] = None,
        environmental_offset_hours: Optional[float] = None,
        oil_type: Optional[str] = None,
        processes_dispersion: Optional[bool] = None,
        processes_evaporation: Optional[bool] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> StochasticRunResult:
        start = time.monotonic()
        paths = self.output_service.prepare_run_directory(stochastic_config)
        logs_path = paths["logs_path"]

        def log(message: str) -> None:
            print(message)
            self.output_service.append_log(logs_path, message)
            if log_callback is not None:
                log_callback(message)

        def check_cancelled() -> None:
            if should_cancel is not None and should_cancel():
                raise RuntimeError("Execution cancelled by user.")

        self.parameter_sampler.validate_run_config(stochastic_config)
        grid = self.converter.fixed_grid_from_config(stochastic_config.grid)
        samples = self.parameter_sampler.sample(stochastic_config)

        self.output_service.write_config(stochastic_config, paths["config_path"])
        self.output_service.write_samples(samples, paths["sampled_parameters_path"])
        expected_output_times = self._expected_output_times(
            manchas=manchas,
            base_config=base_config,
            observed_offset_hours=observed_offset_hours,
        )
        log(
            f"Starting stochastic run '{stochastic_config.run_name}' with "
            f"{stochastic_config.n_simulations} deterministic simulations."
        )
        if expected_output_times:
            log(
                f"Fixed output timeline: {len(expected_output_times)} timestep(s), "
                f"{expected_output_times[0]} to {expected_output_times[-1]}."
            )
        log(
            f"Fixed grid: lon=[{grid.lon_min:.6f},{grid.lon_max:.6f}] "
            f"lat=[{grid.lat_min:.6f},{grid.lat_max:.6f}] "
            f"resolution={grid.spatial_resolution:g} crs={grid.crs}"
        )

        sample_results: dict[int, SampledParameterSet] = {
            sample.simulation_id: sample for sample in samples
        }
        worker_count = min(max(1, int(stochastic_config.number_of_workers)), len(samples))
        log(f"Running deterministic members with {worker_count} worker(s).")

        worker_payload = {
            "manchas": manchas,
            "base_config": base_config,
            "individual_runs_dir": paths["individual_runs_dir"],
            "padding_animation_frame": padding_animation_frame,
            "forcing_source": forcing_source,
            "current_dataset_paths": current_dataset_paths,
            "wind_dataset_paths": wind_dataset_paths,
            "observed_offset_hours": observed_offset_hours,
            "environmental_offset_hours": environmental_offset_hours,
            "oil_type": oil_type,
            "processes_dispersion": processes_dispersion,
            "processes_evaporation": processes_evaporation,
            "expected_output_times": expected_output_times,
        }

        completed_simulations = 0
        if worker_count == 1:
            for sample in samples:
                check_cancelled()
                log(
                    f"[{sample.simulation_id + 1}/{len(samples)}] "
                    f"cdf={sample.cdf:.6g} wdf={sample.wdf:.6g} "
                    f"tau={sample.temporal_lag_seconds:.0f}s seed={sample.seed}"
                )
                result = _run_stochastic_member_worker({**worker_payload, "sample": sample})
                sample_results[result.simulation_id] = result
                if result.status == "failed":
                    log(f"Simulation {result.simulation_id} failed: {result.error_message}")
                completed_simulations += 1
                self.output_service.write_samples(
                    self._ordered_samples(sample_results),
                    paths["sampled_parameters_path"],
                )
                if progress_callback is not None:
                    progress_callback(completed_simulations, len(samples))
        else:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=mp.get_context("spawn"),
            ) as executor:
                futures = {
                    executor.submit(
                        _run_stochastic_member_worker,
                        {**worker_payload, "sample": sample},
                    ): sample
                    for sample in samples
                }
                for future in as_completed(futures):
                    check_cancelled()
                    sample = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        output_path = (
                            paths["individual_runs_dir"]
                            / f"run_{sample.simulation_id:04d}"
                            / "simulation.nc"
                        )
                        result = replace(
                            sample,
                            status="failed",
                            error_message=str(exc),
                            output_path=str(output_path),
                        )
                    sample_results[result.simulation_id] = result
                    if result.status == "failed":
                        log(f"Simulation {result.simulation_id} failed: {result.error_message}")
                    completed_simulations += 1
                    self.output_service.write_samples(
                        self._ordered_samples(sample_results),
                        paths["sampled_parameters_path"],
                    )
                    if progress_callback is not None:
                        progress_callback(completed_simulations, len(samples))

        check_cancelled()
        hit_count: Optional[np.ndarray] = None
        hit_count_final_timestep: Optional[np.ndarray] = None
        hourly_hit_counts: list[np.ndarray] = []
        hourly_time_indices: list[int] = []
        hourly_time_labels: list[str] = []
        successful_paths: list[Path] = []
        valid_simulations = 0
        hit_count_map_path = None
        probability_map_path = None
        hit_count_final_timestep_map_path = None
        probability_final_timestep_map_path = None
        hourly_probability_map_paths: list[Path] = []

        # Cada membro contribui com uma mascara binaria por celula (atingiu ou
        # nao), nunca com contagem de particulas: a probabilidade e a fracao de
        # membros que alcancaram a celula, e nao a densidade de oleo nela.
        # Sao produzidas tres visoes: varredura de todos os instantes, apenas o
        # instante final, e uma por hora.
        log("Aggregating valid simulation outputs.")
        for sample in self._ordered_samples(sample_results):
            if sample.status != "success":
                continue
            output_path = Path(sample.output_path)
            try:
                binary = self.converter.convert_simulation_to_binary_array(output_path, grid)
                hourly_binaries = self.converter.convert_simulation_to_binary_arrays_for_times(
                    output_path,
                    grid,
                    expected_times=expected_output_times,
                    start_time_index=1,
                )
                if hourly_binaries:
                    binary_final_timestep = hourly_binaries[-1][2]
                else:
                    binary_final_timestep = self.converter.convert_simulation_to_binary_array(
                        output_path,
                        grid,
                        time_index=-1,
                    )
                if hit_count is None:
                    hit_count = np.zeros(binary.shape, dtype=np.uint32)
                if hit_count_final_timestep is None:
                    hit_count_final_timestep = np.zeros(binary_final_timestep.shape, dtype=np.uint32)
                hit_count += binary.astype(np.uint32)
                hit_count_final_timestep += binary_final_timestep.astype(np.uint32)

                if hourly_binaries:
                    if not hourly_hit_counts:
                        hourly_time_indices = [time_index for time_index, _, _ in hourly_binaries]
                        hourly_time_labels = [time_label for _, time_label, _ in hourly_binaries]
                        hourly_hit_counts = [
                            np.zeros(binary_by_time.shape, dtype=np.uint32)
                            for _, _, binary_by_time in hourly_binaries
                        ]
                    elif len(hourly_binaries) != len(hourly_hit_counts):
                        log(
                            f"Simulation {sample.simulation_id} has "
                            f"{len(hourly_binaries)} hourly rasters after initial; "
                            f"expected {len(hourly_hit_counts)}. Missing hours are zero-filled."
                        )
                        hourly_binaries = self._pad_or_trim_hourly_binaries(
                            hourly_binaries=hourly_binaries,
                            expected_count=len(hourly_hit_counts),
                            grid_shape=hit_count.shape,
                        )

                    for hour_index, (_, _, binary_by_time) in enumerate(hourly_binaries):
                        if binary_by_time.shape != hourly_hit_counts[hour_index].shape:
                            raise ValueError(
                                "Hourly raster shape differs from previous valid members. "
                                f"Expected {hourly_hit_counts[hour_index].shape}, "
                                f"got {binary_by_time.shape}."
                            )
                        hourly_hit_counts[hour_index] += binary_by_time.astype(np.uint32)

                successful_paths.append(output_path)
                valid_simulations += 1
            except Exception as exc:
                log(f"Simulation {sample.simulation_id} raster aggregation failed: {exc}")
                sample_results[sample.simulation_id] = replace(
                    sample,
                    status="failed",
                    error_message=f"Raster aggregation failed: {exc}",
                )

        self.output_service.write_samples(
            self._ordered_samples(sample_results),
            paths["sampled_parameters_path"],
        )

        if valid_simulations:
            probability_map = hit_count.astype(np.float32) / float(valid_simulations)
            aggregation = self.aggregator.save_maps(
                EnsembleAggregationResult(
                    valid_simulations=valid_simulations,
                    hit_count=hit_count,
                    probability_map=probability_map,
                ),
                paths["aggregated_dir"],
                grid,
            )
            hit_count_map_path = aggregation.hit_count_map_path
            probability_map_path = aggregation.probability_map_path
            probability_final_timestep_map = (
                hit_count_final_timestep.astype(np.float32) / float(valid_simulations)
            )
            final_timestep_aggregation = self.aggregator.save_maps(
                EnsembleAggregationResult(
                    valid_simulations=valid_simulations,
                    hit_count=hit_count_final_timestep,
                    probability_map=probability_final_timestep_map,
                ),
                paths["aggregated_dir"],
                grid,
                hit_count_filename="hit_count_final_timestep_map.tif",
                probability_filename="probability_final_timestep_map.tif",
            )
            hit_count_final_timestep_map_path = final_timestep_aggregation.hit_count_map_path
            probability_final_timestep_map_path = final_timestep_aggregation.probability_map_path
            hourly_probability_map_paths = self._save_hourly_probability_rasters(
                hourly_hit_counts=hourly_hit_counts,
                hourly_time_indices=hourly_time_indices,
                hourly_time_labels=hourly_time_labels,
                valid_simulations=valid_simulations,
                output_dir=paths["aggregated_dir"] / "hourly_probability_rasters",
                grid=grid,
            )
            log(
                f"Aggregated {valid_simulations} valid simulations "
                f"into raster, final-timestep, and {len(hourly_probability_map_paths)} hourly maps."
            )
        else:
            log("No valid simulations available for aggregation.")

        failed_simulations = len(samples) - valid_simulations
        summary = {
            "run_name": stochastic_config.run_name,
            "total_simulations": len(samples),
            "successful_simulations": valid_simulations,
            "failed_simulations": failed_simulations,
            "output_path": str(paths["run_dir"]),
            "sampled_parameters_path": str(paths["sampled_parameters_path"]),
            "summary_path": str(paths["summary_path"]),
            "hit_count_map_path": str(hit_count_map_path) if hit_count_map_path else None,
            "probability_map_path": str(probability_map_path) if probability_map_path else None,
            "hit_count_final_timestep_map_path": (
                str(hit_count_final_timestep_map_path)
                if hit_count_final_timestep_map_path
                else None
            ),
            "probability_final_timestep_map_path": (
                str(probability_final_timestep_map_path)
                if probability_final_timestep_map_path
                else None
            ),
            "hourly_probability_map_paths": [
                str(path) for path in hourly_probability_map_paths
            ],
            "individual_output_paths": [str(path) for path in successful_paths],
            "elapsed_seconds": time.monotonic() - start,
            "stochastic_config": asdict(stochastic_config),
        }
        self.output_service.write_summary(summary, paths["summary_path"])
        log(
            f"Finished stochastic run: {valid_simulations} success, "
            f"{failed_simulations} failed, output={paths['run_dir']}"
        )

        return StochasticRunResult(
            run_name=stochastic_config.run_name,
            total_simulations=len(samples),
            successful_simulations=valid_simulations,
            failed_simulations=failed_simulations,
            output_path=paths["run_dir"],
            sampled_parameters_path=paths["sampled_parameters_path"],
            summary_path=paths["summary_path"],
            hit_count_map_path=hit_count_map_path,
            probability_map_path=probability_map_path,
            hit_count_final_timestep_map_path=hit_count_final_timestep_map_path,
            probability_final_timestep_map_path=probability_final_timestep_map_path,
            hourly_probability_map_paths=tuple(hourly_probability_map_paths),
        )

    @staticmethod
    def _ordered_samples(samples_by_id: dict[int, SampledParameterSet]) -> list[SampledParameterSet]:
        return [
            samples_by_id[simulation_id]
            for simulation_id in sorted(samples_by_id)
        ]

    def _expected_output_times(
        self,
        manchas,
        base_config,
        observed_offset_hours: Optional[float],
    ) -> tuple[str, ...]:
        manchas_for_time = manchas.copy()
        offset_hours = observed_offset_hours
        if offset_hours is None:
            offset_hours = float(
                getattr(
                    getattr(base_config.copernicusmarine, "specificities", object()),
                    "datetime_offset_hours",
                    0,
                )
                or 0.0
            )
        self.simulation_service.spill_repository.ensure_datetime_column(
            manchas_for_time,
            offset_hours=float(offset_hours or 0.0),
        )
        datetimes = pd.to_datetime(manchas_for_time["datetime"]).sort_values()
        if datetimes.empty:
            return ()

        start_time = pd.Timestamp(datetimes.iloc[0])
        end_time = pd.Timestamp(datetimes.iloc[-1])
        step_minutes = float(getattr(base_config.simulation, "output_time_step_minutes", 60) or 60)
        step = pd.Timedelta(minutes=step_minutes)
        if step <= pd.Timedelta(0):
            raise ValueError("simulation output_time_step_minutes must be > 0")

        times: list[str] = []
        current_time = start_time
        while current_time <= end_time:
            times.append(current_time.isoformat())
            current_time += step
        return tuple(times)

    @staticmethod
    def _pad_or_trim_hourly_binaries(
        hourly_binaries: list[tuple[int, str, np.ndarray]],
        expected_count: int,
        grid_shape: tuple[int, int],
    ) -> list[tuple[int, str, np.ndarray]]:
        if len(hourly_binaries) >= expected_count:
            return hourly_binaries[:expected_count]
        padded = list(hourly_binaries)
        for time_index in range(len(hourly_binaries) + 1, expected_count + 1):
            padded.append(
                (
                    time_index,
                    f"missing_{time_index:03d}",
                    np.zeros(grid_shape, dtype=np.uint8),
                )
            )
        return padded

    def _save_hourly_probability_rasters(
        self,
        hourly_hit_counts: list[np.ndarray],
        hourly_time_indices: list[int],
        hourly_time_labels: list[str],
        valid_simulations: int,
        output_dir: Path,
        grid,
    ) -> list[Path]:
        output_paths: list[Path] = []
        for position, hit_count_by_time in enumerate(hourly_hit_counts):
            time_index = hourly_time_indices[position]
            time_label = self._safe_filename_label(hourly_time_labels[position])
            probability = hit_count_by_time.astype(np.float32) / float(valid_simulations)
            output_path = output_dir / f"probability_timestep_{time_index:03d}_{time_label}.tif"
            self.converter.save_geotiff(probability, output_path, grid, dtype=np.float32)
            output_paths.append(output_path)
        return output_paths

    @staticmethod
    def _safe_filename_label(value: str) -> str:
        label = "".join(char if char.isalnum() else "_" for char in str(value))
        return "_".join(part for part in label.split("_") if part) or "time"
