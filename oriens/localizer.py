import matplotlib.pyplot as plt
import cv2
import numpy as np

import time

from oriens.utils import timer

from oriens.maploc import logger
from oriens.maploc.demo import Demo
from oriens.maploc.osm.tiling import TileManager
from oriens.maploc.utils.exif import EXIF
from oriens.maploc.utils.geo import BoundaryBox, Projection

from oriens.maploc.osm.viz import GeoPlotter
from oriens.maploc.osm.tiling import TileManager
from oriens.maploc.osm.viz import Colormap, plot_nodes
from oriens.maploc.utils.viz_2d import plot_images, features_to_RGB
from oriens.maploc.utils.viz_localization import (
    likelihood_overlay,
    plot_dense_rotations,
    add_circle_inset,
)

from pathlib import Path


class Localizer:
    def __init__(
        self,
        image,
        prior_latlon,
        focal_length=368,
        tile_size_meters=32,
        num_rotations=256,
        device="cuda",
    ):
        self.demo = Demo(num_rotations=num_rotations, device=device)
        self.image = image
        self.prior_latlon = prior_latlon
        self.focal_length = focal_length
        self.tile_size_meters = tile_size_meters

    def generate_bbox(self, center):
        return BoundaryBox(center, center) + self.tile_size_meters

    def get_image_data(self):  # Reconstructed function of read_input_image
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        start = time.time()
        gravity, camera = self.demo.calibrator.run(image, self.focal_length)
        logger.info(f"Using (roll, pitch) {gravity}.")
        logger.info(f"Time taken for calibrator: {time.time() - start}")

        latlon = self.prior_latlon
        proj = Projection(*latlon)
        center = proj.project(latlon)
        bbox = self.generate_bbox(center)

        return image, camera, gravity, proj, bbox

    @timer
    def localize(self, image=None):
        start = time.time()
        # Get the image data
        image, camera, gravity, proj, bbox = self.get_image_data()

        start = time.time()
        # Query OpenStreetMap for this area
        tiler = TileManager.from_bbox(
            proj,
            bbox + 10,
            self.demo.config.data.pixel_per_meter,
            path=Path("cache/osm.json"),
        )
        canvas = tiler.query(bbox)

        start = time.time()
        # Run the inference
        uv, yaw, prob, neural_map, image_rectified = self.demo.localize(
            image, camera, canvas, gravity=gravity
        )

        lat, lon = proj.unproject(canvas.to_xy(uv))
        yaw = yaw.item()

        if image is not None:  # If image is provided, visualize the results
            map_viz = Colormap.apply(canvas.raster)
            overlay = likelihood_overlay(
                prob.numpy().max(-1), map_viz.mean(-1, keepdims=True)
            )
            (neural_map_rgb,) = features_to_RGB(neural_map.numpy())

            # Visualize the results
            plot_images(
                [image, map_viz, overlay, neural_map_rgb],
                titles=[
                    "Input Image",
                    "OpenStreetMap raster",
                    "Prediction",
                    "Neural Map",
                ],
            )
            ax = plt.gcf().axes[2]
            ax.scatter(*canvas.to_uv(bbox.center), s=5, c="red")
            plot_dense_rotations(ax, prob, w=0.005, s=1 / 25)
            # add_circle_inset(ax, uv)
            plt.savefig("experiments/all.png")

        return (lat, lon, yaw)
