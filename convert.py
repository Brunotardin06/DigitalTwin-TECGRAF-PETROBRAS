import click
from hydra import compose, initialize
from pathlib import Path
from shutil import rmtree
from xarray import open_dataset
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import warnings
import numpy as np
from src.utils import create_grid_gpd, dataset_to_geodataframe, save_geopandas_grid_to_geotiff
import pandas as pd
from multiprocessing import Pool


warnings.filterwarnings("ignore", category=UserWarning)  
def convert_data(config, simulated_file):
    output_path = Path(config.paths.converted_data) / config.simulation.name / config.grid_format.name / simulated_file.stem
    output_path.unlink(missing_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    drift_simulation_ds = open_dataset(simulated_file)
    
    grid_cells = create_grid_gpd(
        drift_simulation_ds.geospatial_lon_min - config.grid_format.margin, 
        drift_simulation_ds.geospatial_lon_max + config.grid_format.margin, 
        drift_simulation_ds.geospatial_lat_min - config.grid_format.margin, 
        drift_simulation_ds.geospatial_lat_max + config.grid_format.margin, 
        config.grid_format.spatial_resolution
    )

    grid_centroids_gdf = gpd.GeoDataFrame(geometry=grid_cells.geometry.centroid, crs="EPSG:4326")

    # minx, miny, maxx, maxy = grid_centroids_gdf.total_bounds

    #open water and wind datasets
    water_ds = open_dataset(config.copernicusmarine.water_dataset_path)
    wind_ds = open_dataset(config.copernicusmarine.wind_dataset_path)
    
    # subselect water and wind data to the bounding box of the drift simulation
    water_ds = water_ds.sel(
        longitude=slice(drift_simulation_ds.lon.minval-0.1, drift_simulation_ds.lon.maxval+0.1),
        latitude=slice(drift_simulation_ds.lat.minval-0.1, drift_simulation_ds.lat.maxval+0.1),
        time=slice(drift_simulation_ds.time.values[0], drift_simulation_ds.time.values[-1])
    )
    wind_ds = wind_ds.sel(
        longitude=slice(drift_simulation_ds.lon.minval-0.1, drift_simulation_ds.lon.maxval+0.1),
        latitude=slice(drift_simulation_ds.lat.minval-0.1, drift_simulation_ds.lat.maxval+0.1),
        time=slice(drift_simulation_ds.time.values[0], drift_simulation_ds.time.values[-1])
    )

    # resample water and wind data to the grid centroids
    water_ds_resampled = water_ds.interp(
        longitude=grid_centroids_gdf.geometry.x.unique(),
        latitude=grid_centroids_gdf.geometry.y.unique(),
        method="linear"
    )
    wind_ds_resampled = wind_ds.interp(
        longitude=grid_centroids_gdf.geometry.x.unique(),
        latitude=grid_centroids_gdf.geometry.y.unique(),
        method="linear"
    )

    # count points in each cell for each time step and rasterize the results with water and wind data
    for t in tqdm(drift_simulation_ds.time.values, desc=f"Converting simulation {simulated_file.stem}", unit="time step"):
        drift_simulation_ds_t = drift_simulation_ds.sel(time=t)
        drift_simulation_df_t = drift_simulation_ds_t.to_dataframe().reset_index()
        drift_simulation_df_t = drift_simulation_df_t.dropna(subset=['lon'])  # Drop rows where trajectory is NaN
        drift_simulation_gdf_t = gpd.GeoDataFrame(drift_simulation_df_t, geometry=gpd.points_from_xy(drift_simulation_df_t.lon, drift_simulation_df_t.lat), crs="EPSG:4326")
        drift_simulation_gdf_t = drift_simulation_gdf_t.drop(columns=['lon', 'lat', 'time', 'status', 'trajectory'])
        # drift_simulation_gdf_t['value'] = 1.0
        joined = drift_simulation_gdf_t.sjoin(grid_cells, how='left', predicate='intersects')
        joined = joined.drop(columns=['geometry'])
        joined = joined['index_right'].value_counts().reset_index()
        
        count_cells = grid_cells.copy()
        count_cells['count'] = np.uint16(0)
        for _, row in joined.iterrows():
            count_cells.at[row['index_right'], 'count'] = row['count']
            
        count_cells.geometry = count_cells.geometry.centroid
        
        water_ds_resampled_t = water_ds_resampled.sel(time=t)
        wind_ds_resampled_t = wind_ds_resampled.sel(time=t)
        
        water_gdf_t = dataset_to_geodataframe(water_ds_resampled_t, 'longitude', 'latitude')
        wind_gdf_t = dataset_to_geodataframe(wind_ds_resampled_t, 'longitude', 'latitude')

        water_gdf_t = water_gdf_t.drop(columns=['longitude', 'latitude', 'time', 'depth'])
        wind_gdf_t = wind_gdf_t.drop(columns=['longitude', 'latitude', 'time'])

        water_gdf_t['vo'] = water_gdf_t['vo'].fillna(0)
        water_gdf_t['uo'] = water_gdf_t['uo'].fillna(0)
        wind_gdf_t['northward_wind'] = wind_gdf_t['northward_wind'].fillna(0)
        wind_gdf_t['eastward_wind'] = wind_gdf_t['eastward_wind'].fillna(0)
        
        water_gdf_t['vo'] = water_gdf_t['vo'].clip(-100, 100)
        water_gdf_t['uo'] = water_gdf_t['uo'].clip(-100, 100)
        wind_gdf_t['northward_wind'] = wind_gdf_t['northward_wind'].clip(-300, 300)
        wind_gdf_t['eastward_wind'] = wind_gdf_t['eastward_wind'].clip(-300, 300)
        
        water_gdf_t['vo'] = (water_gdf_t['vo']*100).round().astype(np.int16)
        water_gdf_t['uo'] = (water_gdf_t['uo']*100).round().astype(np.int16)
        wind_gdf_t['northward_wind'] = (wind_gdf_t['northward_wind']*100).round().astype(np.int16)
        wind_gdf_t['eastward_wind'] = (wind_gdf_t['eastward_wind']*100).round().astype(np.int16)

        drift_simulation_gdf_t.to_file(output_path / f'drift_points.gpkg', layer = f'drift_t{pd.to_datetime(t).strftime("%Y%m%d%H")}', driver='GPKG', mode='a')

        params = [
            [count_cells, 'count', output_path / f'drift_count_{pd.to_datetime(t).strftime("%Y%m%d%H")}.tif', config.grid_format.spatial_resolution],
            [count_cells, 'count', output_path / f'drift_gaussian_count_{pd.to_datetime(t).strftime("%Y%m%d%H")}.tif', config.grid_format.spatial_resolution, 'gaussian'],
            [count_cells, 'count', output_path / f'drift_binary_count_{pd.to_datetime(t).strftime("%Y%m%d%H")}.tif', config.grid_format.spatial_resolution, 'binary'],
            [water_gdf_t, 'vo', output_path / f'water_north_{pd.to_datetime(t).strftime("%Y%m%d%H")}.tif', config.grid_format.spatial_resolution],
            [water_gdf_t, 'uo', output_path / f'water_east_{pd.to_datetime(t).strftime("%Y%m%d%H")}.tif', config.grid_format.spatial_resolution],
            [wind_gdf_t, 'northward_wind', output_path / f'wind_north_{pd.to_datetime(t).strftime("%Y%m%d%H")}.tif', config.grid_format.spatial_resolution],
            [wind_gdf_t, 'eastward_wind', output_path / f'wind_east_{pd.to_datetime(t).strftime("%Y%m%d%d%H")}.tif', config.grid_format.spatial_resolution]
        ]
        
        # with Pool(5) as p:
        #     p.starmap(save_geopandas_grid_to_geotiff, params)
        
        for param in params:
            save_geopandas_grid_to_geotiff(*param)    
        
        

@click.command()
@click.option('--config-name', default='main')
@click.option('--skip-animation', is_flag=True, default=False)
@click.option('--number-of-workers', default=4, type=int)
def convert(config_name, skip_animation, number_of_workers):
    with initialize(version_base=None, config_path="conf"):
        config = compose(config_name=config_name, overrides=[])

    simulated_data_path = Path(config.paths.simulation_data) / config.simulation.name
    converted_data_path = Path(config.paths.converted_data) / config.simulation.name / config.grid_format.name
    
    rmtree(converted_data_path, ignore_errors=True)
    converted_data_path.mkdir(parents=True, exist_ok=True)
    
    simulation_data_files = list(simulated_data_path.glob('*.nc'))
    
    for simulated_file in simulation_data_files:
        convert_data(config, simulated_file)

if __name__ == '__main__':
    convert()