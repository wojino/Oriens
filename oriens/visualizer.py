import numpy as np
import matplotlib.pyplot as plt

import folium

from oriens.maploc.utils.geo import BoundaryBox


class Visualizer:
    def __init__(self, bbox: BoundaryBox):
        self.bbox = bbox
        self.map = folium.Map(
            location=self.bbox.center, zoom_start=17, tiles="OpenStreetMap"
        )

    def plot_gps(self, gps_points, color):
        for i, point in enumerate(gps_points):
            folium.CircleMarker(
                location=point,
                radius=2,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=1.0,
                tooltip=f"Point {i+1} ({point[0]:.6f}, {point[1]:.6f})",
            ).add_to(self.map)

        folium.PolyLine(gps_points, color=color).add_to(self.map)

    def save_map(self, path):
        self.map.save(path)
