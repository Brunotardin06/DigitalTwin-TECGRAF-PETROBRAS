from math import ceil, floor
import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_origin
import rasterio
from skimage.filters import gaussian
from skimage.morphology import binary_dilation, binary_erosion, binary_opening, disk


def dataset_to_geodataframe(ds, lon_name='lon', lat_name='lat'):
    """
    Convert an xarray Dataset to a GeoDataFrame.
    
    Parameters:
    - ds: xarray Dataset
    - lon_name: str, name of the longitude variable in the dataset
    - lat_name: str, name of the latitude variable in the dataset
    
    Returns:
    - GeoDataFrame with geometry column
    """
    df = ds.to_dataframe().reset_index()
    gpd_df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_name], df[lat_name]), crs="EPSG:4326")
    return gpd_df

def dataset_to_matrix(ds, var_name):
    """
    Convert an xarray Dataset variable to a 2D numpy array.
    
    Parameters:
    - ds: xarray Dataset
    - var_name: str, name of the variable to convert
    
    Returns:
    - 2D numpy array
    """
    data_array = ds[var_name]
    return data_array.values

def create_grid_gpd(lon_min, lon_max, lat_min, lat_max, spatial_resolution):
    """
    Create a GeoDataFrame representing a grid of square polygons in the specified bounding box from fixed grid values (multiples of spatial_resolution).
    
    Parameters:
    - lon_min, lon_max: float, longitude bounds
    - lat_min, lat_max: float, latitude bounds
    - spatial_resolution: float, size of each grid cell
    
    Returns:
    - GeoDataFrame with grid cells
    """

    lon_min = floor(lon_min / spatial_resolution) * spatial_resolution
    lon_max = ceil(lon_max / spatial_resolution) * spatial_resolution
    lat_min = floor(lat_min / spatial_resolution) * spatial_resolution
    lat_max = ceil(lat_max / spatial_resolution) * spatial_resolution
    
    lon_bins, lat_bins = np.meshgrid(
        np.arange(lon_min, lon_max, spatial_resolution),
        np.arange(lat_min, lat_max, spatial_resolution)
    )
    
    zip_bins = list(zip(lon_bins.flatten(), lat_bins.flatten()))
    
    grid_gdf = gpd.GeoDataFrame(
        [], 
        geometry=gpd.points_from_xy([b[0] for b in zip_bins], [b[1] for b in zip_bins]), 
        crs="EPSG:4326"
    )

    grid_gdf.geometry = grid_gdf.buffer(spatial_resolution/2, cap_style='square')

    return grid_gdf

def save_geopandas_grid_to_geotiff(grid_gdf, value_column, output_path, spatial_resolution, filter = None):
    """
    Save a GeoDataFrame grid to a GeoTIFF file.
    
    Parameters:
    - grid_gdf: GeoDataFrame with grid cells and a column for values
    - value_column: str, name of the column with values to rasterize
    - output_path: str or Path, path to save the GeoTIFF
    - spatial_resolution: float, size of each grid cell
    - filter: str, type of filter to apply ('gaussian', 'binary' or None)
    
    Returns:
    - None
    """
    grid = grid_gdf.copy()

    # Define the transform
    minx, miny, maxx, maxy = grid.total_bounds
    transform = from_origin(minx - spatial_resolution/2, maxy + spatial_resolution/2, spatial_resolution, spatial_resolution)
    
    # Create an empty array for the raster
    n_cols = round((maxx - minx) / spatial_resolution) + 1
    n_rows = round((maxy - miny) / spatial_resolution) + 1

    assert len(grid) == n_cols * n_rows, "Grid GeoDataFrame size does not match calculated raster dimensions."

    grid['longitude'] = grid.geometry.centroid.x
    grid['latitude'] = grid.geometry.centroid.y
    grid = grid.sort_values(by=['latitude', 'longitude'], ascending=[False, True]).reset_index(drop=True)

    rasterized = grid[value_column].values.reshape((n_rows, n_cols))

    if filter=='gaussian':
        rasterized = rasterized*100
        rasterized = gaussian(rasterized, sigma=1, preserve_range=True)
        rasterized = gaussian(rasterized, sigma=1, preserve_range=True)
        rasterized = gaussian(rasterized, sigma=1, preserve_range=True)
        rasterized = gaussian(rasterized, sigma=1, preserve_range=True)
        rasterized = gaussian(rasterized, sigma=1, preserve_range=True)
        rasterized = rasterized.round().astype(np.uint16)
        
    elif filter=='binary':
        rasterized = rasterized*100
        for _ in range(5):
            rasterized = gaussian(rasterized, sigma=5, preserve_range=True)
        rasterized = (rasterized > 10)
        
        rasterized = binary_dilation(rasterized, footprint=disk(10))
        rasterized = binary_erosion(rasterized, footprint=disk(15))
        rasterized = binary_opening(rasterized, footprint=disk(10))
        
        rasterized = rasterized.astype(np.uint8)

    # Save to GeoTIFF
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=rasterized.shape[0],
        width=rasterized.shape[1],
        count=1,
        dtype=rasterized.dtype,
        crs='EPSG:4326',
        transform=transform,
        # nodata=0,
        compress='lzw'
    ) as dst:
        dst.write(rasterized, 1)