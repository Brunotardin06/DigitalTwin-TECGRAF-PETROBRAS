"""Leitura e normalizacao das manchas de oleo observadas (shapefile/zip)."""

from datetime import datetime
from typing import List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import Point, Polygon


def generate_random_points_in_polygon(polygon: Polygon, num_points: int) -> List[Point]:
    """Sorteia num_points dentro do poligono (rejeicao sobre a bounding box)."""
    min_x, min_y, max_x, max_y = polygon.bounds
    points = []

    while len(points) < num_points:
        random_point = Point(np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y))
        if polygon.contains(random_point):
            points.append(random_point)

    return points


class SpillRepository:
    """Normaliza as manchas observadas vindas do shapefile.

    Resolve a coluna de datetime (os shapefiles chegam em dois formatos
    diferentes) e reduz cada instante a um centroide para comparacao com a
    trajetoria simulada.
    """

    def centroid_lat_lon_from_group(self, group: pd.DataFrame) -> Tuple[float, float]:
        if not hasattr(group, "geometry") or group.geometry.empty:
            raise ValueError("Geometry is missing. Provide polygon geometry in the shapefile.")

        geometry = group.geometry.dropna()
        if geometry.empty:
            return np.nan, np.nan

        crs = getattr(group, "crs", None) or "EPSG:4326"
        gdf = gpd.GeoDataFrame(geometry=geometry, crs=crs)
        if gdf.crs is not None and gdf.crs.is_geographic:
            projected_crs = gdf.estimate_utm_crs()
            if projected_crs is None:
                projected_crs = "EPSG:3857"
            gdf_projected = gdf.to_crs(projected_crs)
        else:
            gdf_projected = gdf

        union_geometry = (
            gdf_projected.geometry.union_all()
            if hasattr(gdf_projected.geometry, "union_all")
            else gdf_projected.geometry.unary_union
        )
        centroid = union_geometry.centroid
        centroid_wgs84 = gpd.GeoSeries([centroid], crs=gdf_projected.crs).to_crs(epsg=4326).iloc[0]
        return float(centroid_wgs84.y), float(centroid_wgs84.x)

    def ensure_datetime_column(self, manchas: pd.DataFrame, offset_hours: float = 0.0) -> pd.DataFrame:
        if "datetime" in manchas.columns:
            return manchas

        columns = set(manchas.columns)
        if "DATA_HORA1" in columns and "TEMPO_ENTR" in columns:
            manchas["date"] = pd.to_datetime(manchas["DATA_HORA1"], format="%d/%m/%Y")
            manchas["time"] = pd.to_datetime(manchas["TEMPO_ENTR"], format="%H:%M")
            manchas["datetime"] = manchas.apply(
                lambda row: datetime.combine(row["date"].date(), row["time"].time()),
                axis=1,
            )
        elif "Data/Hora" in columns:
            manchas["datetime"] = pd.to_datetime(
                manchas["Data/Hora"],
                dayfirst=True,
                errors="raise",
            )
        else:
            raise ValueError(
                "Missing datetime fields. Expected DATA_HORA1/TEMPO_ENTR or Data/Hora."
            )

        if offset_hours:
            manchas["datetime"] = manchas["datetime"] + pd.Timedelta(hours=float(offset_hours))
        return manchas

    def build_observed_trajectory(self, manchas: pd.DataFrame) -> pd.DataFrame:
        if "datetime" not in manchas.columns:
            raise ValueError("Expected 'datetime' column in manchas")

        grouped = manchas.groupby("datetime", sort=True)
        rows = []
        for dt, group in grouped:
            lat, lon = self.centroid_lat_lon_from_group(group)
            rows.append({"time": pd.to_datetime(dt), "lon": lon, "lat": lat})
        return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
