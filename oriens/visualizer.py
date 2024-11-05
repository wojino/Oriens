import numpy as np
import matplotlib.pyplot as plt

import folium
from folium.plugins import MeasureControl

import math

from oriens.maploc.utils.geo import BoundaryBox


class Visualizer:
    def __init__(self, bbox: BoundaryBox):
        self.bbox = bbox
        self.map = folium.Map(
            location=self.bbox.center, zoom_start=17, tiles="Esri.WorldImagery"
        )
        folium.TileLayer("OpenStreetMap").add_to(self.map)
        folium.LayerControl().add_to(self.map)

        measure_control = MeasureControl(
            position="topright",
            primary_length_unit="meters",
            secondary_length_unit="kilometers",
            primary_area_unit="sqmeters",
            secondary_area_unit="hectares",
        )
        self.map.add_child(measure_control)

    def plot_gps(self, gps_points, color):
        # gps_points: List[Tuple[float, float, float]] (lat, lon, yaw)
        for i, point in enumerate(gps_points):
            lat, lon, yaw_tensor = point
            yaw = yaw_tensor.item()
            print(lat, lon, yaw)

            # 현재 점에서 yaw 방향으로 0.1m(약 0.000001의 위도 차이)에 해당하는 점을 계산
            distance = 2 / 111320  # 0.1미터를 약 위도 경도로 변환
            delta_lat = distance * math.cos(math.radians(yaw))
            delta_lon = (
                distance * math.sin(math.radians(yaw)) / math.cos(math.radians(lat))
            )

            # 새로운 점의 위치 계산
            end_lat = lat + delta_lat
            end_lon = lon + delta_lon

            print(lat, lon, end_lat, end_lon)

            # 시작점에서 끝점까지의 선 추가
            folium.PolyLine(
                [(lat, lon), (end_lat, end_lon)], color=color, weight=2
            ).add_to(self.map)

            # 원래 GPS 포인트에 마커 추가
            folium.CircleMarker(
                location=(lat, lon),
                radius=1,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=1.0,
                tooltip=f"Point {i+1} ({lat:.6f}, {lon:.6f}, yaw: {yaw})",
            ).add_to(self.map)

    def save_map(self, path):
        self.map.save(path)
