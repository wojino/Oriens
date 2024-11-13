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
from geopy.distance import geodesic


def init():
    logger.info("Get GPS and image data from the bag file.")
    bm = BagManager("experiments/20240410-1.bag")

    df = bm.get_synced_dataframe()
    logger.info("GPS and image data are loaded.")

    logger.info("Get OSM data.")
    CACHE_PATH = Path("cache/osm.json")
    bbox = BoundaryBox(min_=[35.5700, 129.1840], max_=[35.5770, 129.1930])
    get_osm(boundary_box=bbox, cache_path=CACHE_PATH, overwrite=False)
    logger.info("OSM data is downloaded.")

    return df


def monte_carlo(df, idx, num_samples=100):
    prior_latlonyaw = (df.loc[idx, "lat"], df.loc[idx, "lon"], df.loc[idx, "yaw"])
    image = df.loc[idx, "img"]

    gps_latlon = prior_latlonyaw[:2]

    start_point = (gps_latlon[0], gps_latlon[1])
    lat_perturbation = (
        geodesic(meters=1).destination(start_point, 0).latitude - start_point[0]
    )
    lon_perturbation = (
        geodesic(meters=1).destination(start_point, 90).longitude - start_point[1]
    )

    latlonyaw = []
    for i in range(num_samples):
        logger.info(f"Monte Carlo simulation in {idx+1}: {i+1}/{num_samples}")

        # 위도와 경도에 -4m ~ +4m 범위의 랜덤 오차 적용 (95퍼센트)
        lat_offset = np.random.normal(0, 2) * lat_perturbation
        lon_offset = np.random.normal(0, 2) * lon_perturbation
        perturbed_latlon = (gps_latlon[0] + lat_offset, gps_latlon[1] + lon_offset)

        # 로컬라이제이션 수행 및 결과 저장
        latlonyaw.append(Localizer(image, perturbed_latlon).localize(False))

    l = Localizer(image, gps_latlon)
    l.localize(True)
    overlay = l.overlay

    latlonyaw = np.array(latlonyaw)
    reference_location = np.mean(latlonyaw, axis=0)

    distances_from_reference = [
        geodesic(reference_location[:2], latlonyaw[:2]).meters
        for latlonyaw in latlonyaw
    ]
    yaw_differences = [
        abs(latlonyaw[2] - reference_location[2]) % 360 for latlonyaw in latlonyaw
    ]

    mean_distance = np.mean(distances_from_reference)
    std_distance = np.std(distances_from_reference)
    mean_yaw_error = np.mean(yaw_differences)
    std_yaw_error = np.std(yaw_differences)

    logger.info(f"Monte Carlo simulation results for {num_samples} samples.")

    # Prior lat, lon, yaw
    logger.info(
        f"Prior: ({prior_latlonyaw[0]:.6f}, {prior_latlonyaw[1]:.6f}, yaw: {prior_latlonyaw[2]:.2f})"
    )

    # Reference lat, lon, yaw
    logger.info(
        f"Reference: ({reference_location[0]:.6f}, {reference_location[1]:.6f}, yaw: {reference_location[2]:.2f})"
    )

    # Mean, std of distance from reference
    logger.info(f"Mean distance from reference: {mean_distance:.3f} meters")
    logger.info(
        f"Standard deviation of distance from reference: {std_distance:.3f} meters"
    )

    # Mean, std of yaw error from reference
    logger.info(f"Mean yaw error from reference: {mean_yaw_error:.3f} degrees")
    logger.info(
        f"Standard deviation of yaw error from reference: {std_yaw_error:.3f} degrees"
    )

    # Plot
    plot_origin = (
        geodesic(meters=32).destination(start_point, 180).latitude,
        geodesic(meters=32).destination(start_point, 270).longitude,
    )

    points_meters = [
        (
            geodesic(plot_origin, (latlonyaw[0], plot_origin[1])).meters,
            geodesic(plot_origin, (plot_origin[0], latlonyaw[1])).meters,
        )
        for latlonyaw in latlonyaw
    ]

    ref_point_meters = (
        geodesic(plot_origin, (reference_location[0], plot_origin[1])).meters,
        geodesic(plot_origin, (plot_origin[0], reference_location[1])).meters,
    )

    gps_point_meters = (
        geodesic(plot_origin, (gps_latlon[0], plot_origin[1])).meters,
        geodesic(plot_origin, (plot_origin[0], gps_latlon[1])).meters,
    )

    plt.figure(figsize=(10, 10))

    plt.imshow(overlay, cmap="gray", alpha=0.5, extent=[0, 64, 0, 64])

    plt.title(f"Monte Carlo Simulation for {idx+1}, {num_samples} samples")
    plt.xlabel("East (meters)")
    plt.ylabel("North (meters)")
    plt.tight_layout(pad=0.5)

    plt.grid(True)
    plt.axis("equal")
    plt.xlim(0, 64)
    plt.ylim(0, 64)
    # 간격 8m로 설정
    plt.xticks(np.arange(0, 65, 8))
    plt.yticks(np.arange(0, 65, 8))

    plt.scatter(*zip(*points_meters), c="blue", label="Monte Carlo Samples")
    plt.scatter(*ref_point_meters, c="red", label="Reference Location")
    plt.scatter(*gps_point_meters, c="green", label="GPS Location")
    plt.legend()

    # distance와 yaw의 mean 및 std 텍스트 표시
    plt.text(
        1,
        7,
        f"Mean Distance: {mean_distance:.2f} m, Std Distance: {std_distance:.2f} m",
    )
    plt.text(
        1,
        5,
        f"Mean Yaw Error: {mean_yaw_error:.2f}°, Std Yaw Error: {std_yaw_error:.2f}°",
    )

    # GPS 좌표와 Reference 좌표 텍스트 표시
    plt.text(
        1,
        3,
        f"GPS Coordinates: Lat: {start_point[0]:.6f}, Lon: {start_point[1]:.6f}, Yaw: {prior_latlonyaw[2]:.2f}",
    )
    plt.text(
        1,
        1,
        f"Reference Coordinates: Lat: {reference_location[0]:.6f}, Lon: {reference_location[1]:.6f}, Yaw: {reference_location[2]:.2f}",
    )

    plt.savefig(f"experiments/plots/{idx:03d}_m.png")


def monte_carlo_loop(df, num_samples=100):
    logger.info("Start loop for Monte Carlo simulation.")
    for idx, row in df.iterrows():
        monte_carlo(df, idx, num_samples)


def localization(df, idx):
    logger.info(f"Localization: {idx+1}/{len(df)}")

    prior_latlonyaw = (df.loc[idx, "lat"], df.loc[idx, "lon"], df.loc[idx, "yaw"])
    image = df.loc[idx, "img"]

    latlonyaw = Localizer(image, prior_latlonyaw[:2]).localize(True, idx)

    return latlonyaw


def localization_loop(df):
    original = []
    prediction = []

    logger.info("Start loop for localization.")
    for idx, row in df.iterrows():
        latlonyaw = localization(df, idx)

        original.append(prior_latlonyaw)
        prediction.append(latlonyaw)

    logger.info("Plot the GPS data.")
    vis = Visualizer(bbox)
    vis.plot_gps(original, "blue")
    vis.plot_gps(prediction, "red")

    vis.save_map("experiments/map.html")


if __name__ == "__main__":
    df = init()

    # monte_carlo(df, 164)
    monte_carlo_loop(df, 100)
    # localization_loop(df)
    # localization(df, 101)
