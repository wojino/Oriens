import matplotlib.pyplot as plt
import pandas as pd

from oriens.maploc.osm.viz import GeoPlotter, Colormap
from oriens.maploc.osm.tiling import TileManager
from oriens.maploc.utils.geo import BoundaryBox
import numpy as np


class Visualizer:
    def __init__(self):
        self.bbox = None

    def plot_gps(self, gps_df):
        plt.figure(figsize=(10, 8))

        plt.plot(
            gps_df["lon"],
            gps_df["lat"],
            marker="o",
            color="b",
            linestyle="-",
            markersize=2,
        )
        plt.scatter(gps_df["lon"][0], gps_df["lat"][0], color="r", s=100, label="Start")

        plt.ticklabel_format(useOffset=False)

        plt.title("GPS Data")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()

        plt.grid(True)
        plt.axis("equal")

        plt.savefig("oriens/experiments/gps.png", dpi=300, bbox_inches="tight")

    def plot_prediction(self, original, prediction):
        # (lat, lon) -> (lon, lat)
        del_x = original[0][0] - prediction[0][0]
        del_y = original[0][1] - prediction[0][1]

        original = [(lon, lat) for (lat, lon) in original]
        prediction = [(lon + del_x, lat + del_y) for (lat, lon) in prediction]

        plt.scatter(*zip(*original), color="b", label="Original")
        plt.scatter(*zip(*prediction), color="r", label="Prediction")

        plt.ticklabel_format(useOffset=False)

        plt.title("Prediction")
        plt.xlabel("Latitude")
        plt.ylabel("Longitude")

        plt.legend()

        plt.grid(True)
        plt.axis("equal")

        plt.savefig("oriens/experiments/prediction.png", dpi=300, bbox_inches="tight")
