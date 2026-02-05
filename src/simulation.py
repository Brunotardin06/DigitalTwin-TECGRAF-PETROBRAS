from opendrift.models.openoil import OpenOil
from opendrift.readers import reader_netCDF_CF_generic
from pathlib import Path
from datetime import timedelta
import geopandas as gpd
import pandas as pd
from utils.aux_func import generate_random_points_in_polygon
from datetime import datetime
import os


def simulate_drift(
    manchas,
    out_filename,
    config,
    skip_animation,
    padding_animation_frame,
    wind_drift_factor=None,
    stokes_drift=None,
    current_drift_factor=None,
    oil_type=None,
    horizontal_diffusivity=None,
    processes_dispersion=None,
    processes_evaporation=None,
):
    """
    Simulates oil drift using the OpenOil model.
    Parameters:
        manchas (GeoPandas): oil spills described by many attributes
        out_filename (str): Base filename for output files (NetCDF and GIF).
        config (object): Configuration object containing simulation parameters:
            - copernicusmarine.specifities: contains all environment information for readers
            - simulation.num_seed_elements (int): Number of oil elements to seed.
            - simulation.duration_days (int): Total duration of the simulation in days.
            - simulation.time_step_minutes (int): Time step for the simulation in minutes.
            - simulation.output_time_step_minutes (int): Time step for output data in minutes.
            - simulation.export_variables (list): List of variables to export in the NetCDF file.
            - paths.simulation_data (str): Directory path to save simulation data.
            - simulation.name (str): Name of the simulation scenario.

        skip_animation (bool): Whether to generate a GIF animation of the simulation.
        wind_drift_factor (float | None): Override wind drift factor if provided.
        stokes_drift (bool | None): Override Stokes drift flag if provided.
        current_drift_factor (float | None): Override current drift factor if provided.
        oil_type (str | None): Override oil type if provided.
        horizontal_diffusivity (float | None): Override horizontal diffusivity if provided.
        processes_dispersion (bool | None): Override dispersion process toggle if provided.
        processes_evaporation (bool | None): Override evaporation process toggle if provided.
    Returns:
        OpenOil object containing the simulation results.

    Outputs:
        - NetCDF file containing simulation results.
        - GIF animation visualizing the drift with sea water velocity as background.
    Notes:
        - Uses marine Copernicus water and wind data as input.
        - Exports density and water content variables.
        - Animation is saved with sea water velocity as background.
    """
    


    

    # Ensure datetime field exists for different shapefile schemas
    if "datetime" not in manchas.columns:
        if "DATA_HORA1" in manchas.columns and "TEMPO_ENTR" in manchas.columns:
            manchas["date"] = pd.to_datetime(manchas["DATA_HORA1"], format="%d/%m/%Y")
            manchas["time"] = pd.to_datetime(manchas["TEMPO_ENTR"], format="%H:%M")
            manchas["datetime"] = manchas.apply(
                lambda row: datetime.combine(row["date"].date(), row["time"].time()),
                axis=1,
            )
            offset_hours = getattr(config.copernicusmarine.specificities, "datetime_offset_hours", 0)
            if offset_hours:
                manchas["datetime"] = manchas["datetime"] + pd.Timedelta(hours=float(offset_hours))
        elif "Data/Hora" in manchas.columns:
            manchas["datetime"] = pd.to_datetime(
                manchas["Data/Hora"],
                dayfirst=True,
                errors="raise",
            )
            offset_hours = getattr(config.copernicusmarine.specificities, "datetime_offset_hours", 0)
            if offset_hours:
                manchas["datetime"] = manchas["datetime"] + pd.Timedelta(hours=float(offset_hours))
        else:
            raise ValueError(
                "Missing datetime fields. Expected DATA_HORA1/TEMPO_ENTR or Data/Hora."
            )
    manchas.sort_values("datetime", inplace=True)

    # SELECT THE INITIAL SHAPE (first timestamp)
    seed_time_start = manchas["datetime"].iloc[0]
    initial_subset = manchas[manchas["datetime"] == seed_time_start]
    mancha_inicial_geo = (
        initial_subset.geometry.union_all()
        if hasattr(initial_subset.geometry, "union_all")
        else initial_subset.geometry.unary_union
    )

    end_time = manchas["datetime"].max()

    print("First arrival datetime:", seed_time_start)
    print("Most recent spill datetime:", end_time)

    points = generate_random_points_in_polygon(mancha_inicial_geo, config.simulation.num_seed_elements) #Points inside the polygon representing the particules
    elements = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")


    # MODEL DEFINITION
    o = OpenOil(loglevel=50)

    # READING WIND AND CURRENT DATA
    print("Calling readers...")
    readers = [
        reader_netCDF_CF_generic.Reader(Path(config.copernicusmarine.specificities.water_dataset_path)),
        reader_netCDF_CF_generic.Reader(Path(config.copernicusmarine.specificities.wind_dataset_path)),
        reader_netCDF_CF_generic.Reader(Path(config.copernicusmarine.specificities.wave_dataset_path)),
        reader_netCDF_CF_generic.Reader(Path(config.copernicusmarine.specificities.sal_temp_dataset_path)),
    ]
    o.add_reader(readers)


    # ADAPT THE OUTPUT GIF FRAME
    minlon, minlat, maxlon, maxlat = manchas.total_bounds
    padding_lon = (maxlon - minlon) * padding_animation_frame
    padding_lat = (maxlat - minlat) * padding_animation_frame
    min_lon = minlon - padding_lon
    max_lon = maxlon + padding_lon
    min_lat = minlat - padding_lat
    max_lat = maxlat + padding_lat


    print("Seeding elements...")
    o.set_config('drift:advection_scheme', 'runge-kutta4')
    if stokes_drift is None:
        stokes_drift = getattr(config.simulation, "stokes_drift", True)
    o.set_config('drift:stokes_drift', bool(stokes_drift))
    default_wdf = 0.015
    wdf = wind_drift_factor if wind_drift_factor is not None else getattr(config.simulation, "wind_drift_factor", default_wdf)
    o.set_config('seed:wind_drift_factor', wdf)
    if current_drift_factor is None:
        current_drift_factor = getattr(config.simulation, "current_drift_factor", None)
    if current_drift_factor is not None:
        o.set_config('seed:current_drift_factor', float(current_drift_factor))
    if horizontal_diffusivity is not None:
        o.set_config('drift:horizontal_diffusivity', float(horizontal_diffusivity))
    if processes_dispersion is not None:
        o.set_config('processes:dispersion', bool(processes_dispersion))
    if processes_evaporation is not None:
        o.set_config('processes:evaporation', bool(processes_evaporation))
    o.set_config('wave_entrainment:entrainment_rate', 'Li et al. (2017)')
    o.set_config('wave_entrainment:droplet_size_distribution', 'Johansen et al. (2015)')

    #print(o.oiltypes)
    selected_oil = oil_type or getattr(config.simulation, "oil_type", "SOCKEYE SWEET")
    o.seed_from_geopandas(
        geodataframe = elements,
        time = seed_time_start,
        oil_type = selected_oil
        )

    simulation_end_time = seed_time_start + timedelta(days=config.simulation.duration_days)
    #out_filename = f'{out_filename}_{seed_time_start.strftime("%Y%m%dT%H%M")}_{simulation_end_time.strftime("%Y%m%dT%H%M")}'
    
    if os.path.exists(Path(config.paths.simulation_data) / config.simulation.name):
        fname = Path(config.paths.simulation_data) / config.simulation.name / f'{out_filename}.nc' #Relative path to result file name
        print("Running...")
        o.prepare_run()
        o.run(
            end_time=end_time, 
            time_step=config.simulation.time_step_minutes*60, 
            time_step_output=config.simulation.output_time_step_minutes*60,
            outfile=str(fname.absolute()))
        o.elements
    else:
        print("Erro o.run(): o path do arquivo de saída provavelmente não é correto.")

    print(f"Simulation {out_filename} completed successfully.")
    if skip_animation:
        return o
    fname = Path(config.paths.simulation_data) / config.simulation.name / f'{out_filename}.gif'

    
    o.animation(
        filename=str(fname.absolute()),
        background=["x_sea_water_velocity", "y_sea_water_velocity"],
        corners=[min_lon, max_lon, min_lat, max_lat],
        vmin=-1,
        vmax=1,
        fast=True,
        fps=6,
    )
    

    
