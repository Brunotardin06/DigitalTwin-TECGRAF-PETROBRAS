"""Figuras de saida: gif de comparacao e frames por timestep."""

from pathlib import Path
import io

import geopandas as gpd
import imageio.v2 as imageio
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

# Backend nao-interativo: as figuras sao criadas em threads de trabalho
# (execucao pelo Flet) e o Tkinter quebra no teardown nesse cenario.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from shapely.geometry import MultiPoint, Point  # noqa: E402


def _parse_extent(extent_str):
    parts = [p.strip() for p in extent_str.split(",")]
    if len(parts) != 4:
        raise ValueError("extent must be 'min_lon,max_lon,min_lat,max_lat'")
    return [float(p) for p in parts]


def _nearest_time(times, target):
    times = pd.to_datetime(times)
    target = pd.to_datetime(target)
    idx = int(np.abs(times - target).argmin())
    return times[idx]


def _pick_times_from_list(times, n_times):
    times = sorted(pd.to_datetime(times))
    if len(times) == 0:
        raise ValueError("No timestamps available to plot.")
    if len(times) < n_times:
        n_times = len(times)
    idx = np.linspace(0, len(times) - 1, n_times).round().astype(int)
    return [times[i] for i in idx]


def _prepare_real(shp_zip, datetime_offset_hours=0.0, start_index=0):
    real = gpd.read_file(Path(shp_zip)).to_crs(epsg=4326)
    columns = set(real.columns)
    if "DATA_HORA1" in columns and "TEMPO_ENTR" in columns:
        real["date"] = pd.to_datetime(real["DATA_HORA1"], format="%d/%m/%Y")
        real["time"] = pd.to_datetime(real["TEMPO_ENTR"], format="%H:%M")
        real["datetime"] = real.apply(
            lambda row: pd.Timestamp.combine(row["date"].date(), row["time"].time()), axis=1
        )
    elif "Data/Hora" in columns:
        real["datetime"] = pd.to_datetime(real["Data/Hora"], dayfirst=True, errors="raise")
    else:
        raise ValueError("Missing datetime fields. Expected DATA_HORA1/TEMPO_ENTR or Data/Hora.")
    if datetime_offset_hours:
        real["datetime"] = real["datetime"] + pd.Timedelta(hours=float(datetime_offset_hours))
    real = real.sort_values("datetime")
    unique_times = sorted(real["datetime"].unique())
    start_index = int(start_index or 0)
    if start_index < 0 or start_index >= len(unique_times):
        raise ValueError(f"start-index out of range (0..{len(unique_times)-1})")
    if start_index:
        real = real[real["datetime"] >= unique_times[start_index]].copy()
    return real


def _sim_points_at_time(ds, dt):
    sim_times = pd.to_datetime(ds["time"].values)
    idx = int((abs(sim_times - dt)).argmin())
    lons = ds["lon"].isel(time=idx).values
    lats = ds["lat"].isel(time=idx).values
    lons = np.ma.filled(lons, np.nan).astype(float).ravel()
    lats = np.ma.filled(lats, np.nan).astype(float).ravel()
    mask = np.isfinite(lons) & np.isfinite(lats)
    return lons[mask], lats[mask]


def _real_polygon_at_time(real, dt):
    obs_times = pd.to_datetime(real["datetime"].unique())
    nearest = _nearest_time(obs_times, pd.to_datetime(dt))
    subset = real[real["datetime"] == nearest]
    geom = subset.geometry.union_all() if hasattr(subset.geometry, "union_all") else subset.geometry.unary_union
    return geom, nearest


def generate_comparison_gif(
    sim_nc,
    shp_zip="documentos/20_de_junho.zip",
    out="comparison.gif",
    extent="-42.910682,-42.760085,-25.853895,-25.603492",
    fps=0.5,
    frame_step=1,
    use_sim_times=False,
    sim_max_points=3000,
    sim_style="hull",
    real_steps=0,
    real_dynamic=False,
    real_alpha=0.35,
    sim_alpha=0.8,
    datetime_offset_hours=0.0,
    start_index=0,
):
    real = _prepare_real(
        shp_zip,
        datetime_offset_hours=datetime_offset_hours,
        start_index=start_index,
    )
    obs_times = pd.to_datetime(real["datetime"].unique())

    ds = xr.open_dataset(sim_nc, engine="netcdf4")
    sim_times = pd.to_datetime(ds["time"].values)

    obs_in_range = obs_times[(obs_times >= sim_times.min()) & (obs_times <= sim_times.max())]
    base_real_times = obs_in_range if len(obs_in_range) else obs_times
    if real_steps is None or int(real_steps) <= 0:
        real_steps = len(base_real_times)
    static_real_times = _pick_times_from_list(base_real_times, real_steps)
    static_real_geoms = []
    for t in static_real_times:
        geom, nearest = _real_polygon_at_time(real, t)
        static_real_geoms.append((geom, nearest))

    if use_sim_times:
        min_t = obs_times.min()
        max_t = obs_times.max()
        in_range = sim_times[(sim_times >= min_t) & (sim_times <= max_t)]
        times = in_range if len(in_range) else sim_times
    else:
        times = obs_times

    if frame_step > 1:
        times = times[::frame_step]
    if len(times) == 0:
        raise ValueError("No timestamps available to plot.")

    min_lon, max_lon, min_lat, max_lat = _parse_extent(extent)

    frames = []
    for dt in times:
        fig, ax = plt.subplots(figsize=(8, 5))

        # Real polygons (static background) or per-frame
        if real_dynamic:
            real_geom, real_time = _real_polygon_at_time(real, dt)
            gpd.GeoSeries([real_geom], crs="EPSG:4326").plot(
                ax=ax, color="red", alpha=real_alpha, edgecolor="none", label="Real"
            )
        else:
            for i, (geom, _) in enumerate(static_real_geoms):
                gpd.GeoSeries([geom], crs="EPSG:4326").plot(
                    ax=ax,
                    color="red",
                    alpha=real_alpha,
                    edgecolor="none",
                    label="Real" if i == 0 else None,
                )

        # Simulated spill (black)
        lons, lats = _sim_points_at_time(ds, dt)
        if sim_max_points and len(lons) > sim_max_points:
            step = max(1, int(len(lons) / sim_max_points))
            lons = lons[::step]
            lats = lats[::step]

        if len(lons) > 2 and sim_style.lower() == "hull":
            hull = MultiPoint(list(zip(lons, lats))).convex_hull
            gpd.GeoSeries([hull], crs="EPSG:4326").plot(
                ax=ax, color="black", alpha=sim_alpha, edgecolor="none", label="Simulated"
            )
        else:
            ax.scatter(lons, lats, s=6, color="black", alpha=sim_alpha, label="Simulated")

        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        ax.set_title(f"Sim {pd.to_datetime(dt)}")
        ax.grid(True)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", "box")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))

    if fps <= 0:
        raise ValueError("fps must be > 0")
    out = Path(out)
    imageio.mimsave(out, frames, duration=1 / fps)
    print(f"Saved GIF to {out}")


def render_comparison_frames(
    manchas,
    observed_trajectory,
    sim_path: Path,
    out_dir: Path,
    show_plots: bool = False,
    should_cancel=None,
) -> Path:
    """Gera um PNG por timestep observado, sobrepondo particulas e mancha real.

    Retorna o diretorio onde os frames foram escritos.
    """
    frames_dir = Path(out_dir) / f"{Path(sim_path).stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    ds_result = xr.open_dataset(sim_path, engine="netcdf4")
    try:
        for dt_snapshot in list(observed_trajectory["time"]):
            if should_cancel is not None and should_cancel():
                raise RuntimeError("Execution cancelled by user.")
            sim_times = pd.to_datetime(ds_result["time"].values)
            idx_time = (abs(sim_times - dt_snapshot)).argmin()
            lons = ds_result["lon"].isel(time=idx_time).values
            lats = ds_result["lat"].isel(time=idx_time).values
            gdf_sim = gpd.GeoDataFrame(
                {"lon": lons, "lat": lats},
                geometry=[Point(xy) for xy in zip(lons, lats)],
                crs="EPSG:4326",
            )
            manchas_at_time = manchas[manchas["datetime"] == dt_snapshot]

            _, ax = plt.subplots(figsize=(8, 6))
            gdf_sim.plot(ax=ax, color="blue", markersize=5, label="Simulated particles")
            manchas_at_time.plot(ax=ax, color="red", alpha=0.5, label="Observed spill")
            plt.title(f"Oil spill comparison at {dt_snapshot}")
            plt.legend()
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.grid(True)
            timestamp_label = pd.to_datetime(dt_snapshot).strftime("%Y%m%d_%H%M%S")
            plt.savefig(frames_dir / f"step_{timestamp_label}.png", dpi=150, bbox_inches="tight")
            if show_plots:
                plt.show()
            plt.close()
    finally:
        ds_result.close()

    return frames_dir

