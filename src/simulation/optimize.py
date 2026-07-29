"""Busca dos melhores fatores de deriva (WDF/CDF) e a metrica que os julga."""

from typing import Optional

import numpy as np
import pandas as pd
from opendrift.models.openoil import OpenOil

from src.inputs import forcing
from src.simulation.drift import DEFAULT_OIL_TYPE
from src.inputs.forcing import ForcingDatasetAdapter
from src.inputs.spills import SpillRepository

# Abaixo deste passo a simulacao fica lenta demais para uma busca em grade.
MIN_OPTIMIZATION_TIME_STEP_MINUTES = 5.0


class MetricsService:
    """Metricas de distancia e aderencia entre trajetoria observada e simulada."""

    @staticmethod
    def haversine_m(lon1, lat1, lon2, lat2):
        radius_m = 6371000.0
        lon1 = np.radians(lon1)
        lat1 = np.radians(lat1)
        lon2 = np.radians(lon2)
        lat2 = np.radians(lat2)
        delta_lon = lon2 - lon1
        delta_lat = lat2 - lat1
        a_value = (
            np.sin(delta_lat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
        )
        c_value = 2 * np.arctan2(np.sqrt(a_value), np.sqrt(1 - a_value))
        return radius_m * c_value

    def liu_weissberg_skillscore(self, observed_df: pd.DataFrame, modeled_df: pd.DataFrame) -> float:
        merged = observed_df.merge(
            modeled_df,
            on="time",
            suffixes=("_obs", "_mod"),
        ).sort_values("time")
        merged = merged.dropna(subset=["lon_obs", "lat_obs", "lon_mod", "lat_mod"])
        if len(merged) < 2:
            return np.nan
        if len(merged) < 3:
            # With only 2 matched timestamps, trajectory-based skill is unstable.
            # Use endpoint distance and convert to a bounded "higher-is-better" score.
            endpoint_distance_m = self.haversine_m(
                merged["lon_obs"].iloc[-1],
                merged["lat_obs"].iloc[-1],
                merged["lon_mod"].iloc[-1],
                merged["lat_mod"].iloc[-1],
            )
            endpoint_distance_km = float(endpoint_distance_m) / 1000.0
            return 1.0 / (1.0 + endpoint_distance_km)

        obs_lon = merged["lon_obs"].to_numpy()
        obs_lat = merged["lat_obs"].to_numpy()
        mod_lon = merged["lon_mod"].to_numpy()
        mod_lat = merged["lat_mod"].to_numpy()

        separation = self.haversine_m(obs_lon, obs_lat, mod_lon, mod_lat)
        path_length = self.haversine_m(obs_lon[1:], obs_lat[1:], obs_lon[:-1], obs_lat[:-1])
        denominator = np.nansum(path_length)
        if denominator <= 0:
            return np.nan
        return 1.0 - (np.nansum(separation[1:]) / denominator)


class OptimizationService:
    """Busca em grade dos fatores de deriva que melhor reproduzem o observado."""

    def __init__(
        self,
        spill_repository: Optional[SpillRepository] = None,
        metrics_service: Optional[MetricsService] = None,
        forcing_dataset_adapter: Optional[ForcingDatasetAdapter] = None,
    ) -> None:
        self.spill_repository = spill_repository or SpillRepository()
        self.metrics_service = metrics_service or MetricsService()
        self.forcing_dataset_adapter = forcing_dataset_adapter or ForcingDatasetAdapter()

    def fast_grid_search_wind_drift_factor(
        self,
        manchas,
        config,
        observed_trajectory,
        wdf_values,
        particles_per_wdf=1,
        current_drift_factor=None,
        oil_type=None,
        progress=None,
        should_cancel=None,
        forcing_source="COPERNICUS",
        current_dataset_path=None,
        wind_dataset_path=None,
        current_dataset_paths=None,
        wind_dataset_paths=None,
        environmental_offset_hours=None,
    ):
        if should_cancel is not None and should_cancel():
            raise RuntimeError("Execution cancelled by user.")
        if "datetime" not in manchas.columns:
            raise ValueError("Expected 'datetime' column in manchas")
        if particles_per_wdf < 1:
            raise ValueError("particles_per_wdf must be >= 1")

        wdf_values = np.array(wdf_values, dtype=float)
        wdf_values = wdf_values[np.isfinite(wdf_values)]
        if wdf_values.size == 0:
            raise ValueError("wdf_values must contain at least one finite value")

        # Use the first observed timestep as seed to keep behavior consistent
        # with the main simulation service and support short (2-step) cases.
        shape_inicial = manchas.iloc[0]
        seed_time_start = shape_inicial["datetime"]
        shape_final = manchas.loc[manchas["datetime"].idxmax()]
        end_time = shape_final["datetime"]

        obs = observed_trajectory[
            (observed_trajectory["time"] >= seed_time_start)
            & (observed_trajectory["time"] <= end_time)
        ].copy()
        if obs.empty or len(obs) < 2:
            raise ValueError("Observed trajectory has insufficient timestamps within the simulation window")

        initial_group = manchas[manchas["datetime"] == seed_time_start]
        start_lat, start_lon = self.spill_repository.centroid_lat_lon_from_group(initial_group)
        if not np.isfinite(start_lat) or not np.isfinite(start_lon):
            raise ValueError("Invalid centroid for the initial timestep geometry")

        current_paths, wind_paths, sal_temp_path = forcing.resolve_forcing_paths(
            config=config,
            forcing_source=forcing_source,
            adapter=self.forcing_dataset_adapter,
            current_dataset_path=current_dataset_path,
            wind_dataset_path=wind_dataset_path,
            current_dataset_paths=current_dataset_paths,
            wind_dataset_paths=wind_dataset_paths,
        )

        model = OpenOil(loglevel=50)
        model.add_reader(
            forcing.build_readers(
                current_paths=current_paths,
                wind_paths=wind_paths,
                sal_temp_path=sal_temp_path,
                environmental_offset_hours=environmental_offset_hours,
            )
        )
        model.set_config("drift:advection_scheme", "runge-kutta4")
        model.set_config("drift:stokes_drift", False)
        if current_drift_factor is not None:
            model.set_config("seed:current_drift_factor", float(current_drift_factor))

        # O truque que torna esta busca barata: em vez de uma simulacao por
        # WDF, semeia todas as particulas de uma vez com wind_drift_factor
        # vetorizado. Uma rodada cobre a grade inteira de WDF; so o CDF, que
        # e config global do modelo, ainda exige uma rodada por valor.
        wdf_array = np.repeat(wdf_values, particles_per_wdf)
        lon_array = np.full_like(wdf_array, start_lon, dtype=float)
        lat_array = np.full_like(wdf_array, start_lat, dtype=float)

        model.seed_elements(
            lon=lon_array,
            lat=lat_array,
            time=seed_time_start,
            wind_drift_factor=wdf_array,
            oil_type=oil_type or getattr(config.simulation, "oil_type", DEFAULT_OIL_TYPE),
        )

        model.prepare_run()
        if should_cancel is not None and should_cancel():
            raise RuntimeError("Execution cancelled by user.")
        optimization_time_step_minutes = max(
            MIN_OPTIMIZATION_TIME_STEP_MINUTES,
            float(getattr(config.simulation, "time_step_minutes", 1.0) or 1.0),
        )
        model.run(
            end_time=end_time,
            time_step=optimization_time_step_minutes * 60,
            time_step_output=config.simulation.output_time_step_minutes * 60,
        )
        if progress is not None:
            progress.tick()

        ds_result = model.result
        sim_times = pd.to_datetime(ds_result["time"].values)
        time_indices = [int((abs(sim_times - dt)).argmin()) for dt in obs["time"]]
        wdf_per_traj = ds_result["wind_drift_factor"].isel(time=0).values

        results = []
        for wdf in wdf_values:
            mask = np.isclose(wdf_per_traj, wdf, rtol=0, atol=1e-6)
            if not np.any(mask):
                results.append({"wind_drift_factor": float(wdf), "skillscore": float("nan")})
                continue

            rows = []
            for dt, idx in zip(obs["time"], time_indices):
                lons = ds_result["lon"].isel(time=idx).values[mask]
                lats = ds_result["lat"].isel(time=idx).values[mask]
                lons = np.ma.filled(lons, np.nan).astype(float).ravel()
                lats = np.ma.filled(lats, np.nan).astype(float).ravel()
                valid = np.isfinite(lons) & np.isfinite(lats)
                if not valid.any():
                    rows.append({"time": pd.to_datetime(dt), "lon": np.nan, "lat": np.nan})
                    continue
                rows.append(
                    {
                        "time": pd.to_datetime(dt),
                        "lon": float(np.nanmean(lons[valid])),
                        "lat": float(np.nanmean(lats[valid])),
                    }
                )

            sim_traj = pd.DataFrame(rows)
            score = self.metrics_service.liu_weissberg_skillscore(obs, sim_traj)
            results.append({"wind_drift_factor": float(wdf), "skillscore": float(score)})

        results_df = pd.DataFrame(results)
        best_row = None
        if not results_df.empty and results_df["skillscore"].notna().any():
            best_row = results_df.loc[results_df["skillscore"].idxmax()]
        return best_row, results_df

    def fast_grid_search_wdf_cdf(
        self,
        manchas,
        config,
        observed_trajectory,
        wdf_values,
        current_drift_values,
        particles_per_wdf=1,
        oil_type=None,
        progress=None,
        should_cancel=None,
        forcing_source="COPERNICUS",
        current_dataset_path=None,
        wind_dataset_path=None,
        current_dataset_paths=None,
        wind_dataset_paths=None,
        environmental_offset_hours=None,
    ):
        if isinstance(current_drift_values, (int, float, np.floating, np.integer)):
            current_drift_values = [float(current_drift_values)]
        current_drift_values = [float(value) for value in current_drift_values]
        if not current_drift_values:
            raise ValueError("current_drift_values must contain at least one value")

        results = []
        for current_drift_factor in current_drift_values:
            if should_cancel is not None and should_cancel():
                raise RuntimeError("Execution cancelled by user.")
            _, dataframe = self.fast_grid_search_wind_drift_factor(
                manchas,
                config,
                observed_trajectory,
                wdf_values,
                particles_per_wdf=particles_per_wdf,
                current_drift_factor=current_drift_factor,
                oil_type=oil_type,
                progress=progress,
                should_cancel=should_cancel,
                forcing_source=forcing_source,
                current_dataset_path=current_dataset_path,
                wind_dataset_path=wind_dataset_path,
                current_dataset_paths=current_dataset_paths,
                wind_dataset_paths=wind_dataset_paths,
                environmental_offset_hours=environmental_offset_hours,
            )
            if dataframe.empty:
                continue

            dataframe = dataframe.copy()
            dataframe["current_drift_factor"] = float(current_drift_factor)
            results.append(dataframe)

        results_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
        best_row = None
        if not results_df.empty and results_df["skillscore"].notna().any():
            best_row = results_df.loc[results_df["skillscore"].idxmax()]
        return best_row, results_df
