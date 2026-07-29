import click
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt


def _pick_var_name(ds, candidates):
    for name in candidates:
        if name in ds.variables:
            return name
    return None


def _pick_coord_name(ds, candidates):
    for name in candidates:
        if name in ds.coords:
            return name
    for name in candidates:
        if name in ds.dims:
            return name
    return None


def _slice_surface(da):
    for dim in ("depth", "depthu", "depthv", "z"):
        if dim in da.dims:
            return da.isel({dim: 0})
    return da


def _parse_extent(extent_str):
    parts = [p.strip() for p in extent_str.split(",")]
    if len(parts) != 4:
        raise ValueError("extent must be 'min_lon,max_lon,min_lat,max_lat'")
    return [float(p) for p in parts]


def _pick_times_from_list(times, n_times):
    times = sorted(times)
    if len(times) == 0:
        raise ValueError("No timestamps available to plot.")
    if len(times) < n_times:
        print(f"Warning: only {len(times)} timestamps available, using all of them.")
        n_times = len(times)
    idx = np.linspace(0, len(times) - 1, n_times).round().astype(int)
    return [times[i] for i in idx]


def _load_times(times_str, times_file):
    selected = []
    if times_file:
        file_path = Path(times_file)
        if not file_path.exists():
            raise ValueError(f"times file not found: {file_path}")
        selected.extend([line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()])
    if times_str:
        selected.extend([item.strip() for item in times_str.split(",") if item.strip()])
    if not selected:
        return None
    return [pd.to_datetime(t) for t in selected]


@click.command()
@click.option("--shp-zip", default="documentos/20_de_junho.zip")
@click.option("--water-nc", default="data/1-raw/marine_copernicus/water_032019.nc")
@click.option("--sim-nc", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--n-times", type=int, default=4)
@click.option("--extent", default="-42.910682,-42.760085,-25.853895,-25.603492")
@click.option("--extent-pad", type=float, default=0.0)
@click.option("--vmin", type=float, default=-1.0)
@click.option("--vmax", type=float, default=1.0)
@click.option("--quiver-scale", type=float, default=20.0)
@click.option("--sim-max-points", type=int, default=1500)
@click.option("--only-spills", is_flag=True, default=False)
@click.option("--sim-alpha", type=float, default=0.25)
@click.option("--times", type=str, default=None)
@click.option("--times-file", type=str, default=None)
@click.option("--overlay", is_flag=True, default=False)
@click.option("--datetime-offset-hours", type=float, default=0.0)
@click.option("--out-dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default=None)
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), default=None)
def main(
    shp_zip,
    water_nc,
    sim_nc,
    n_times,
    extent,
    extent_pad,
    vmin,
    vmax,
    quiver_scale,
    sim_max_points,
    only_spills,
    sim_alpha,
    times,
    times_file,
    overlay,
    datetime_offset_hours,
    out_dir,
    out,
):
    # Load observed spills
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
    observed_times = sorted(real["datetime"].unique())

    sim_ds = None
    sim_times = None
    if sim_nc:
        sim_ds = xr.open_dataset(Path(sim_nc), engine="netcdf4")
        sim_times = pd.to_datetime(sim_ds["time"].values)
        sim_start = sim_times.min()
        sim_end = sim_times.max()
        observed_in_range = [t for t in observed_times if sim_start <= t <= sim_end]
        base_times = observed_in_range if observed_in_range else observed_times
    else:
        base_times = observed_times

    selected_times = _load_times(times, times_file)
    if selected_times:
        times = selected_times
    elif out_dir:
        times = list(base_times)
    else:
        times = _pick_times_from_list(base_times, n_times)

    min_lon, max_lon, min_lat, max_lat = _parse_extent(extent)
    if extent_pad < 0:
        raise ValueError("extent-pad must be >= 0")
    if extent_pad > 0:
        width = max_lon - min_lon
        height = max_lat - min_lat
        if width > 0 and height > 0:
            pad_lon = width * extent_pad
            pad_lat = height * extent_pad
            min_lon -= pad_lon
            max_lon += pad_lon
            min_lat -= pad_lat
            max_lat += pad_lat
    water_times = None
    ds = None
    if not only_spills:
        # Load water dataset
        ds = xr.open_dataset(Path(water_nc))
        u_name = _pick_var_name(ds, ["uo", "x_sea_water_velocity", "eastward_sea_water_velocity"])
        v_name = _pick_var_name(ds, ["vo", "y_sea_water_velocity", "northward_sea_water_velocity"])
        if not u_name or not v_name:
            raise ValueError("Could not find current variables (uo/vo) in water dataset.")

        lat_name = _pick_coord_name(ds, ["latitude", "lat", "y"])
        lon_name = _pick_coord_name(ds, ["longitude", "lon", "x"])
        time_name = _pick_coord_name(ds, ["time"])
        if not lat_name or not lon_name or not time_name:
            raise ValueError("Could not find lat/lon/time coordinates in water dataset.")
        water_times = pd.to_datetime(ds[time_name].values)

    mesh = None
    if overlay:
        fig, ax = plt.subplots(figsize=(8, 6))
        axes = [ax]
        color_map = plt.cm.get_cmap("tab10", len(times))
        colors = [color_map(i) for i in range(len(times))]
    else:
        ncols = int(np.ceil(np.sqrt(n_times)))
        nrows = int(np.ceil(n_times / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8), sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        for dt in times:
            fig, ax = plt.subplots(figsize=(8, 6))
            mesh = None
            if not only_spills:
                idx = int((abs(water_times - dt)).argmin())
                ds_t = ds.isel({time_name: idx})
                u = _slice_surface(ds_t[u_name]).squeeze()
                v = _slice_surface(ds_t[v_name]).squeeze()

                u = u.transpose(lat_name, lon_name)
                v = v.transpose(lat_name, lon_name)
                lats = u[lat_name].values
                lons = u[lon_name].values
                u_vals = u.values
                v_vals = v.values

                mesh = ax.pcolormesh(lons, lats, u_vals, vmin=vmin, vmax=vmax, cmap="coolwarm", shading="auto")

                stride = max(1, int(max(u_vals.shape) / 25))
                ax.quiver(
                    lons[::stride],
                    lats[::stride],
                    u_vals[::stride, ::stride],
                    v_vals[::stride, ::stride],
                    scale=quiver_scale,
                    width=0.002,
                    color="k",
                )

            subset = real[real["datetime"] == dt]
            subset.plot(ax=ax, color="red", alpha=0.5, edgecolor="none")
            if sim_ds is not None:
                sim_idx = int((abs(sim_times - dt)).argmin())
                sim_lons = sim_ds["lon"].isel(time=sim_idx).values
                sim_lats = sim_ds["lat"].isel(time=sim_idx).values
                sim_lons = np.ma.filled(sim_lons, np.nan).astype(float).ravel()
                sim_lats = np.ma.filled(sim_lats, np.nan).astype(float).ravel()
                mask = np.isfinite(sim_lons) & np.isfinite(sim_lats)
                sim_lons = sim_lons[mask]
                sim_lats = sim_lats[mask]
                if sim_max_points and len(sim_lons) > sim_max_points:
                    step = max(1, int(len(sim_lons) / sim_max_points))
                    sim_lons = sim_lons[::step]
                    sim_lats = sim_lats[::step]
                ax.scatter(sim_lons, sim_lats, s=6, color="blue", alpha=sim_alpha, label="Simulated")

            ax.set_xlim(min_lon, max_lon)
            ax.set_ylim(min_lat, max_lat)
            ax.set_title(f"{pd.to_datetime(dt)}")
            ax.grid(True)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

            if mesh is not None:
                cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", fraction=0.04, pad=0.02)
                cbar.set_label("x_sea_water_velocity (m/s)")

            fname = out_dir / f"{pd.to_datetime(dt):%Y%m%dT%H%M}.png"
            fig.savefig(fname, dpi=200)
            plt.close(fig)
        return

    if overlay:
        ax = axes[0]
        if not only_spills:
            idx = int((abs(water_times - times[0])).argmin())
            ds_t = ds.isel({time_name: idx})
            u = _slice_surface(ds_t[u_name]).squeeze()
            v = _slice_surface(ds_t[v_name]).squeeze()

            u = u.transpose(lat_name, lon_name)
            v = v.transpose(lat_name, lon_name)
            lats = u[lat_name].values
            lons = u[lon_name].values
            u_vals = u.values
            v_vals = v.values

            mesh = ax.pcolormesh(lons, lats, u_vals, vmin=vmin, vmax=vmax, cmap="coolwarm", shading="auto")
            stride = max(1, int(max(u_vals.shape) / 25))
            ax.quiver(
                lons[::stride],
                lats[::stride],
                u_vals[::stride, ::stride],
                v_vals[::stride, ::stride],
                scale=quiver_scale,
                width=0.002,
                color="k",
            )

        for dt, color in zip(times, colors):
            subset = real[real["datetime"] == dt]
            subset.plot(ax=ax, color=color, alpha=0.6, edgecolor="none", label=str(pd.to_datetime(dt)))
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        ax.set_title("Observed spills (overlay)")
        ax.grid(True)
        ax.legend()
    else:
        for ax, dt in zip(axes, times):
            if not only_spills:
                idx = int((abs(water_times - dt)).argmin())
                ds_t = ds.isel({time_name: idx})
                u = _slice_surface(ds_t[u_name]).squeeze()
                v = _slice_surface(ds_t[v_name]).squeeze()

                u = u.transpose(lat_name, lon_name)
                v = v.transpose(lat_name, lon_name)
                lats = u[lat_name].values
                lons = u[lon_name].values
                u_vals = u.values
                v_vals = v.values

                mesh = ax.pcolormesh(lons, lats, u_vals, vmin=vmin, vmax=vmax, cmap="coolwarm", shading="auto")

                stride = max(1, int(max(u_vals.shape) / 25))
                ax.quiver(
                    lons[::stride],
                    lats[::stride],
                    u_vals[::stride, ::stride],
                    v_vals[::stride, ::stride],
                    scale=quiver_scale,
                    width=0.002,
                    color="k",
                )

            subset = real[real["datetime"] == dt]
            subset.plot(ax=ax, color="red", alpha=0.5, edgecolor="none")
            if sim_ds is not None:
                sim_idx = int((abs(sim_times - dt)).argmin())
                sim_lons = sim_ds["lon"].isel(time=sim_idx).values
                sim_lats = sim_ds["lat"].isel(time=sim_idx).values
                sim_lons = np.ma.filled(sim_lons, np.nan).astype(float).ravel()
                sim_lats = np.ma.filled(sim_lats, np.nan).astype(float).ravel()
                mask = np.isfinite(sim_lons) & np.isfinite(sim_lats)
                sim_lons = sim_lons[mask]
                sim_lats = sim_lats[mask]
                if sim_max_points and len(sim_lons) > sim_max_points:
                    step = max(1, int(len(sim_lons) / sim_max_points))
                    sim_lons = sim_lons[::step]
                    sim_lats = sim_lats[::step]
                ax.scatter(sim_lons, sim_lats, s=6, color="blue", alpha=sim_alpha, label="Simulated")

            ax.set_xlim(min_lon, max_lon)
            ax.set_ylim(min_lat, max_lat)
            ax.set_title(f"{pd.to_datetime(dt)}")
            ax.grid(True)

    if not overlay:
        for ax in axes[len(times) :]:
            ax.axis("off")

    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes.tolist(), orientation="vertical", fraction=0.02, pad=0.02)
        cbar.set_label("x_sea_water_velocity (m/s)")
        fig.suptitle("Observed spills over currents", fontsize=12)
    else:
        fig.suptitle("Observed spills", fontsize=12)
    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()
