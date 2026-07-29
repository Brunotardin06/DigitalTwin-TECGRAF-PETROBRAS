"""Grade fixa, rasterizacao de particulas e agregacao em mapas de probabilidade."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import from_origin

from src.inputs.config import StochasticGridConfig


@dataclass(frozen=True)
class FixedGrid:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    spatial_resolution: float
    crs: str = "EPSG:4326"

    @property
    def width(self) -> int:
        return int(np.ceil((self.lon_max - self.lon_min) / self.spatial_resolution))

    @property
    def height(self) -> int:
        return int(np.ceil((self.lat_max - self.lat_min) / self.spatial_resolution))

    @property
    def transform(self):
        return from_origin(
            self.lon_min,
            self.lat_max,
            self.spatial_resolution,
            self.spatial_resolution,
        )

    def validate(self) -> None:
        if self.lon_min >= self.lon_max:
            raise ValueError("grid lon_min must be < lon_max")
        if self.lat_min >= self.lat_max:
            raise ValueError("grid lat_min must be < lat_max")
        if self.spatial_resolution <= 0:
            raise ValueError("grid spatial_resolution must be > 0")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("grid dimensions must be positive")


class DriftRasterConverter:
    """Converte posicoes de particulas do OpenDrift em arrays de grade fixa.

    A grade e fixa para que membros diferentes do ensemble sejam somaveis
    celula a celula, independente de onde cada um espalhou as particulas.
    """

    @staticmethod
    def fixed_grid_from_config(config: StochasticGridConfig) -> FixedGrid:
        margin = float(config.margin or 0.0)
        grid = FixedGrid(
            lon_min=float(config.lon_min) - margin,
            lon_max=float(config.lon_max) + margin,
            lat_min=float(config.lat_min) - margin,
            lat_max=float(config.lat_max) + margin,
            spatial_resolution=float(config.spatial_resolution),
            crs=config.crs,
        )
        grid.validate()
        return grid

    def convert_simulation_to_binary_array(
        self,
        simulation_path: Path,
        grid: FixedGrid,
        time_index: Optional[int] = None,
    ) -> np.ndarray:
        count = self.convert_simulation_to_count_array(simulation_path, grid, time_index=time_index)
        return (count > 0).astype(np.uint8)

    def convert_simulation_to_count_array(
        self,
        simulation_path: Path,
        grid: FixedGrid,
        time_index: Optional[int] = None,
    ) -> np.ndarray:
        grid.validate()
        simulation_path = Path(simulation_path)
        with xr.open_dataset(simulation_path) as dataset:
            lon = dataset["lon"]
            lat = dataset["lat"]
            if time_index is not None:
                lon_values = lon.isel(time=time_index).values
                lat_values = lat.isel(time=time_index).values
            else:
                lon_values = lon.values
                lat_values = lat.values

        return self.rasterize_points(lon_values, lat_values, grid)

    def convert_simulation_to_binary_arrays_for_times(
        self,
        simulation_path: Path,
        grid: FixedGrid,
        expected_times: Sequence[object],
        start_time_index: int = 1,
    ) -> list[tuple[int, str, np.ndarray]]:
        grid.validate()
        expected_time_values = np.asarray(list(expected_times), dtype="datetime64[ns]")
        if start_time_index < 0:
            start_time_index = max(0, len(expected_time_values) + start_time_index)
        if start_time_index >= len(expected_time_values):
            return []

        simulation_path = Path(simulation_path)
        with xr.open_dataset(simulation_path) as dataset:
            lon = dataset["lon"]
            lat = dataset["lat"]
            dataset_time_values = np.asarray(dataset["time"].values, dtype="datetime64[ns]")
            time_lookup = {
                int(time_value.astype(np.int64)): position
                for position, time_value in enumerate(dataset_time_values)
            }

            arrays: list[tuple[int, str, np.ndarray]] = []
            for time_index in range(start_time_index, len(expected_time_values)):
                expected_time = expected_time_values[time_index]
                dataset_time_index = time_lookup.get(int(expected_time.astype(np.int64)))
                if dataset_time_index is None:
                    binary = np.zeros((grid.height, grid.width), dtype=np.uint8)
                else:
                    count = self.rasterize_points(
                        lon.isel(time=dataset_time_index).values,
                        lat.isel(time=dataset_time_index).values,
                        grid,
                    )
                    binary = (count > 0).astype(np.uint8)
                arrays.append((time_index, self._format_time_value(expected_time), binary))
            return arrays

    @staticmethod
    def _format_time_value(value) -> str:
        if np.issubdtype(np.asarray(value).dtype, np.datetime64):
            return str(np.datetime_as_string(value, unit="s"))
        return str(value)

    def rasterize_points(
        self,
        lon_values,
        lat_values,
        grid: FixedGrid,
    ) -> np.ndarray:
        lon_array = np.ma.filled(np.asarray(lon_values), np.nan).astype(float).ravel()
        lat_array = np.ma.filled(np.asarray(lat_values), np.nan).astype(float).ravel()
        valid = (
            np.isfinite(lon_array)
            & np.isfinite(lat_array)
            & (lon_array >= grid.lon_min)
            & (lon_array <= grid.lon_max)
            & (lat_array >= grid.lat_min)
            & (lat_array <= grid.lat_max)
        )
        rows = np.floor((grid.lat_max - lat_array[valid]) / grid.spatial_resolution).astype(int)
        cols = np.floor((lon_array[valid] - grid.lon_min) / grid.spatial_resolution).astype(int)
        rows = np.clip(rows, 0, grid.height - 1)
        cols = np.clip(cols, 0, grid.width - 1)
        inside = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)

        count = np.zeros((grid.height, grid.width), dtype=np.uint32)
        if inside.any():
            np.add.at(count, (rows[inside], cols[inside]), 1)
        return count

    @staticmethod
    def save_geotiff(array: np.ndarray, output_path: Path, grid: FixedGrid, dtype=None) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        raster = np.asarray(array)
        if dtype is not None:
            raster = raster.astype(dtype)

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=raster.shape[0],
            width=raster.shape[1],
            count=1,
            dtype=raster.dtype,
            crs=grid.crs,
            transform=grid.transform,
            compress="lzw",
        ) as dataset:
            dataset.write(raster, 1)


@dataclass(frozen=True)
class EnsembleAggregationResult:
    valid_simulations: int
    hit_count: np.ndarray
    probability_map: np.ndarray
    hit_count_map_path: Optional[Path] = None
    probability_map_path: Optional[Path] = None


class EnsembleAggregator:
    """Soma as mascaras binarias dos membros e converte em probabilidade."""

    def __init__(self, converter: Optional[DriftRasterConverter] = None) -> None:
        self.converter = converter or DriftRasterConverter()

    def save_maps(
        self,
        result: EnsembleAggregationResult,
        output_dir: Path,
        grid: FixedGrid,
        hit_count_filename: str = "hit_count_map.tif",
        probability_filename: str = "probability_map.tif",
    ) -> EnsembleAggregationResult:
        output_dir = Path(output_dir)
        hit_count_path = output_dir / hit_count_filename
        probability_path = output_dir / probability_filename

        self.converter.save_geotiff(result.hit_count, hit_count_path, grid, dtype=np.uint32)
        self.converter.save_geotiff(result.probability_map, probability_path, grid, dtype=np.float32)

        return EnsembleAggregationResult(
            valid_simulations=result.valid_simulations,
            hit_count=result.hit_count,
            probability_map=result.probability_map,
            hit_count_map_path=hit_count_path,
            probability_map_path=probability_path,
        )
