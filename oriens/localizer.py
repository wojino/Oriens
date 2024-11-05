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
        tile_size_meters=2,
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
        logger.info("Using (roll, pitch) %s.", gravity)
        print("Time taken for calibrator:", time.time() - start)

        latlon = self.prior_latlon
        proj = Projection(*latlon)
        center = proj.project(latlon)
        bbox = self.generate_bbox(center)

        return image, camera, gravity, proj, bbox

    @timer
    def localize(self, plot: bool):
        if plot:
            return self.localize_with_plot()
        else:
            return self.localize_without_plot()

    def localize_without_plot(self):
        start = time.time()
        # Get the image data
        image, camera, gravity, proj, bbox = self.get_image_data()
        print("Time taken for image data:", time.time() - start)

        start = time.time()
        # Query OpenStreetMap for this area
        tiler = TileManager.from_bbox(
            proj,
            bbox + 10,
            self.demo.config.data.pixel_per_meter,
            path=Path("cache/osm.json"),
        )
        canvas = tiler.query(bbox)
        print("Time taken for querying OSM:", time.time() - start)

        start = time.time()
        # Run the inference
        uv, yaw, prob, neural_map, image_rectified = self.demo.localize(
            image, camera, canvas, gravity=gravity
        )

        latlon = proj.unproject(uv)
        print("Time taken for inference:", time.time() - start)

        return latlon, yaw

    def localize_with_plot(self):
        start = time.time()
        # Get the image data
        image, camera, gravity, proj, bbox = self.get_image_data()
        print("Time taken for image data:", time.time() - start)

        start = time.time()
        # Query OpenStreetMap for this area
        tiler = TileManager.from_bbox(
            proj,
            bbox + 10,
            self.demo.config.data.pixel_per_meter,
            path=Path("cache/osm.json"),
        )
        canvas = tiler.query(bbox)
        print("Time taken for querying OSM:", time.time() - start)

        map_viz = Colormap.apply(canvas.raster)
        plot_images([map_viz], titles=["OpenStreetMap raster"])
        # plot_nodes(0, canvas.raster[2], fontsize=6, size=10)
        plt.savefig("map_viz.png")

        start = time.time()
        # Run the inference
        uv, yaw, prob, neural_map, image_rectified = self.demo.localize(
            image, camera, canvas, gravity=gravity
        )

        lat, lon = proj.unproject(uv)
        print("Time taken for inference:", time.time() - start)

        return (lat, lon, yaw)
