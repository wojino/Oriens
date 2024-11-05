import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np

from oriens import logger
from oriens.bag_manager import BagManager
from oriens.visualizer import Visualizer
from oriens.localizer import Localizer

from oriens.maploc.osm.download import get_osm
from oriens.maploc.utils.geo import BoundaryBox
from pathlib import Path


if __name__ == "__main__":
    logger.info("Get GPS and image data from the bag file.")
    bm = BagManager("oriens/experiments/20240410-1.bag")

    gps = bm.get_gps_dataframe()
    image = bm.get_image_dataframe()
    attitute = bm.get_attitude_dataframe()
    logger.info("GPS and image data are loaded.")

    logger.info("Get OSM data.")
    CACHE_PATH = Path("cache/osm.json")
    bbox = BoundaryBox(min_=[35.5700, 129.1840], max_=[35.5770, 129.1930])
    osm_data = get_osm(boundary_box=bbox, cache_path=CACHE_PATH, overwrite=False)
    logger.info("OSM data is downloaded.")

    original = []
    prediction = []

    logger.info("Start loop for localization.")
    for idx, row in image.iterrows():
        gps_rows = gps[gps["sec"] == row["sec"]]
        if gps_rows.empty:
            continue
        gps_row = gps_rows.iloc[0]

        prior_latlonyaw = (gps_row["lat"], gps_row["lon"], gps_row["yaw"])
        image = row["img"]  # BGR

        latlonyaw = Localizer(image, prior_latlonyaw[:2]).localize(True)

        original.append(prior_latlonyaw)
        prediction.append(latlonyaw)

        break

    logger.info("Plot the GPS data.")
    vis = Visualizer(bbox)
    vis.plot_gps(original, "blue")
    vis.plot_gps(prediction, "red")

    vis.save_map("experiments/map.html")
