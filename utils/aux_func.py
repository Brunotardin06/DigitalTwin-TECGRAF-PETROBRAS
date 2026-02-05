from shapely import Point, Polygon
from typing import List
import numpy as np

def generate_random_points_in_polygon(polygon: Polygon, num_points: int) -> List[Point]:
    min_x, min_y, max_x, max_y = polygon.bounds
    points = []

    while len(points) < num_points:
        random_point = Point(np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y))
        if polygon.contains(random_point):
            points.append(random_point)

    return points

