import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random

from oriens import logger
from oriens.bag_manager import BagManager
from oriens.visualizer import Visualizer
from oriens.localizer import Localizer

from oriens.maploc.osm.download import get_osm
from oriens.maploc.utils.geo import BoundaryBox
from pathlib import Path


def monte_carlo(df, idx, num_samples=100):
    prior_latlonyaw = (df.loc[idx, "lat"], df.loc[idx, "lon"], df.loc[idx, "yaw"])
    image = df.loc[idx, "img"]

    gps_latlon = prior_latlonyaw[:2]
    # 위도와 경도의 약 1미터 당 거리 변화 (단위: 위도 및 경도 1도 당 미터)
    meters_per_deg_lat = 111_320  # 위도 1도 당 약 111.32km
    meters_per_deg_lon = 111_320 * np.cos(
        np.radians(gps_latlon[0])
    )  # 경도 1도 당 거리 (위도에 따라 다름)

    latlonyaw = []
    for i in range(num_samples):
        logger.info(f"Monte Carlo simulation: {i+1}/{num_samples}")

        # 위도와 경도에 -4m ~ +4m 범위의 랜덤 오차 적용
        lat_offset = random.uniform(-4, 4) / meters_per_deg_lat
        lon_offset = random.uniform(-4, 4) / meters_per_deg_lon
        perturbed_latlon = (gps_latlon[0] + lat_offset, gps_latlon[1] + lon_offset)

        # 로컬라이제이션 수행 및 결과 저장
        latlonyaw.append(Localizer(image, perturbed_latlon).localize(False))

    lat_mean, lon_mean, yaw_mean = np.mean(latlonyaw, axis=0)
    lat_std, lon_std, yaw_std = np.std(latlonyaw, axis=0)
    lat_meter_std = lat_std * meters_per_deg_lat
    lon_meter_std = lon_std * meters_per_deg_lon

    logger.info(f"Monte Carlo simulation results for {num_samples} samples.")
    logger.info(
        f"Prior: ({prior_latlonyaw[0]:.6f}, {prior_latlonyaw[1]:.6f}, yaw: {prior_latlonyaw[2]:.2f})"
    )

    logger.info(f"Mean: ({lat_mean:.6f}, {lon_mean:.6f}, yaw: {yaw_mean:.2f})")
    logger.info(f"Std: ({lat_std:.6f}, {lon_std:.6f}, yaw: {yaw_std:.2f})")

    logger.info(f"Std in meters: ({lat_meter_std:.3f}, {lon_meter_std:.3f})")


def localization_loop(df):
    original = []
    prediction = []

    logger.info("Start loop for localization.")
    for idx, row in df.iterrows():
        prior_latlonyaw = (row["lat"], row["lon"], row["yaw"])
        image = row["img"]

        latlonyaw = Localizer(image, prior_latlonyaw[:2]).localize(False)

        original.append(prior_latlonyaw)
        prediction.append(latlonyaw)

        break

    logger.info("Plot the GPS data.")
    vis = Visualizer(bbox)
    vis.plot_gps(original, "blue")
    vis.plot_gps(prediction, "red")

    vis.save_map("experiments/map.html")


def localization(df, idx):
    prior_latlonyaw = (df.loc[idx, "lat"], df.loc[idx, "lon"], df.loc[idx, "yaw"])
    image = df.loc[idx, "img"]

    latlonyaw = Localizer(image, prior_latlonyaw[:2]).localize(True, idx)

    return latlonyaw


if __name__ == "__main__":
    logger.info("Get GPS and image data from the bag file.")
    bm = BagManager("experiments/20240410-1.bag")

    df = bm.get_synced_dataframe()
    logger.info("GPS and image data are loaded.")

    logger.info("Get OSM data.")
    CACHE_PATH = Path("cache/osm.json")
    bbox = BoundaryBox(min_=[35.5700, 129.1840], max_=[35.5770, 129.1930])
    osm_data = get_osm(boundary_box=bbox, cache_path=CACHE_PATH, overwrite=False)
    logger.info("OSM data is downloaded.")

    # monte_carlo(df, 164)
    # localization_loop(df)
    localization(df, 101)
