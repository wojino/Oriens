import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np

from oriens import logger
from oriens.bag_manager import BagManager
from oriens.visualizer import Visualizer
from oriens.localizer import Localizer


if __name__ == "__main__":
    logger.info("Get GPS and image data from the bag file.")
    bm = BagManager("oriens/experiments/20240410-1.bag")

    gps = bm.get_gps_dataframe()
    image = bm.get_image_dataframe()
    # attitute = bm.get_attitude_dataframe()

    original = []
    prediction = []

    for idx, row in image.iterrows():
        gps_rows = gps[gps["sec"] == row["sec"]]
        if gps_rows.empty:
            continue
        gps_row = gps_rows.iloc[0]

        prior_latlon = (gps_row["lat"], gps_row["lon"])
        image = row["img"]  # BGR

        latlon, yaw = Localizer(image, prior_latlon).localize()
        # Localizer(image, prior_latlon).localize_full()

        original.append(prior_latlon)
        prediction.append(latlon)

        break

    vis = Visualizer()
    # vis.plot_gps(gps)
    vis.plot_prediction(original, prediction)
