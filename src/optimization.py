from pathlib import Path

import numpy as np
import pandas as pd
from opendrift.models.openoil import OpenOil
from opendrift.readers import reader_netCDF_CF_generic


def build_observed_trajectory(manchas):
    if "datetime" not in manchas.columns:
        raise ValueError("Expected 'datetime' column in manchas")
    grouped = manchas.groupby("datetime", sort=True)
    rows = []
    for dt, group in grouped:
        geom = group.geometry.union_all() if hasattr(group.geometry, "union_all") else group.geometry.unary_union
        centroid = geom.centroid
        rows.append({"time": pd.to_datetime(dt), "lon": centroid.x, "lat": centroid.y})
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def _haversine_m(lon1, lat1, lon2, lat2):
    radius_m = 6371000.0
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius_m * c


def liu_weissberg_skillscore(observed_df, modeled_df):
    merged = observed_df.merge(modeled_df, on="time", suffixes=("_obs", "_mod")).sort_values("time")
    merged = merged.dropna(subset=["lon_obs", "lat_obs", "lon_mod", "lat_mod"])
    if len(merged) < 2:
        return np.nan

    obs_lon = merged["lon_obs"].to_numpy()
    obs_lat = merged["lat_obs"].to_numpy()
    mod_lon = merged["lon_mod"].to_numpy()
    mod_lat = merged["lat_mod"].to_numpy()

    separation = _haversine_m(obs_lon, obs_lat, mod_lon, mod_lat)
    path_len = _haversine_m(obs_lon[1:], obs_lat[1:], obs_lon[:-1], obs_lat[:-1])
    denom = np.nansum(path_len)
    if denom <= 0:
        return np.nan
    return 1.0 - (np.nansum(separation[1:]) / denom)


def fast_grid_search_wind_drift_factor(
    manchas,
    config,
    observed_trajectory,
    wdf_values,
    particles_per_wdf=1,
    current_drift_factor=None,
    horizontal_diffusivity=None,
    oil_type=None,
    progress=None,
):
    if "datetime" not in manchas.columns:
        raise ValueError("Expected 'datetime' column in manchas")
    if particles_per_wdf < 1:
        raise ValueError("particles_per_wdf must be >= 1")

    wdf_values = np.array(wdf_values, dtype=float)
    wdf_values = wdf_values[np.isfinite(wdf_values)]
    if wdf_values.size == 0:
        raise ValueError("wdf_values must contain at least one finite value")

    if len(manchas) > 1:
        shape_inicial = manchas.iloc[1]
    else:
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

    start_point = shape_inicial.geometry.centroid

    o = OpenOil(loglevel=50)
    readers = [
        reader_netCDF_CF_generic.Reader(Path(config.copernicusmarine.specificities.water_dataset_path)),
        reader_netCDF_CF_generic.Reader(Path(config.copernicusmarine.specificities.wind_dataset_path)),
        reader_netCDF_CF_generic.Reader(Path(config.copernicusmarine.specificities.wave_dataset_path)),
        reader_netCDF_CF_generic.Reader(Path(config.copernicusmarine.specificities.sal_temp_dataset_path)),
    ]
    o.add_reader(readers)
    o.set_config("drift:advection_scheme", "runge-kutta4")
    # Fixed by design for this simplified optimizer.
    o.set_config("drift:stokes_drift", False)
    if current_drift_factor is not None:
        o.set_config("seed:current_drift_factor", float(current_drift_factor))
    if horizontal_diffusivity is not None:
        o.set_config("drift:horizontal_diffusivity", float(horizontal_diffusivity))
    o.set_config("wave_entrainment:entrainment_rate", "Li et al. (2017)")
    o.set_config("wave_entrainment:droplet_size_distribution", "Johansen et al. (2015)")

    wdf_array = np.repeat(wdf_values, particles_per_wdf)
    lon_array = np.full_like(wdf_array, float(start_point.x), dtype=float)
    lat_array = np.full_like(wdf_array, float(start_point.y), dtype=float)

    o.seed_elements(
        lon=lon_array,
        lat=lat_array,
        time=seed_time_start,
        wind_drift_factor=wdf_array,
        oil_type=oil_type or getattr(config.simulation, "oil_type", "SOCKEYE SWEET"),
    )

    o.prepare_run()
    o.run(
        end_time=end_time,
        time_step=config.simulation.time_step_minutes * 60,
        time_step_output=config.simulation.output_time_step_minutes * 60,
    )
    if progress is not None:
        progress.tick()

    ds_result = o.result
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
        score = liu_weissberg_skillscore(obs, sim_traj)
        results.append({"wind_drift_factor": float(wdf), "skillscore": float(score)})

    results_df = pd.DataFrame(results)
    best_row = None
    if not results_df.empty and results_df["skillscore"].notna().any():
        best_row = results_df.loc[results_df["skillscore"].idxmax()]
    return best_row, results_df


def fast_grid_search_wdf_stokes_current_drift(
    manchas,
    config,
    observed_trajectory,
    wdf_values,
    current_drift_values,
    horizontal_diffusivity_values=None,
    particles_per_wdf=1,
    oil_type=None,
    progress=None,
):
    # stokes is intentionally fixed to False in this simplified module.
    if isinstance(current_drift_values, (int, float, np.floating, np.integer)):
        current_drift_values = [float(current_drift_values)]
    current_drift_values = [float(v) for v in current_drift_values]
    if not current_drift_values:
        raise ValueError("current_drift_values must contain at least one value")
    if horizontal_diffusivity_values is None:
        horizontal_diffusivity_values = [None]
    elif isinstance(horizontal_diffusivity_values, (int, float, np.floating, np.integer)):
        horizontal_diffusivity_values = [float(horizontal_diffusivity_values)]
    else:
        horizontal_diffusivity_values = [float(v) for v in horizontal_diffusivity_values]
    if not horizontal_diffusivity_values:
        raise ValueError("horizontal_diffusivity_values must contain at least one value")

    results = []
    for current_drift_factor in current_drift_values:
        for horizontal_diffusivity in horizontal_diffusivity_values:
            _, df = fast_grid_search_wind_drift_factor(
                manchas,
                config,
                observed_trajectory,
                wdf_values,
                particles_per_wdf=particles_per_wdf,
                current_drift_factor=current_drift_factor,
                horizontal_diffusivity=horizontal_diffusivity,
                oil_type=oil_type,
                progress=progress,
            )
            if df.empty:
                continue
            df = df.copy()
            df["stokes_drift"] = False
            df["current_drift_factor"] = float(current_drift_factor)
            if horizontal_diffusivity is not None:
                df["horizontal_diffusivity"] = float(horizontal_diffusivity)
            results.append(df)

    results_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    best_row = None
    if not results_df.empty and results_df["skillscore"].notna().any():
        best_row = results_df.loc[results_df["skillscore"].idxmax()]
    return best_row, results_df
