import matplotlib.pyplot as plt
import cv2

from oriens.maploc import logger
from oriens.maploc.demo import Demo
from oriens.maploc.osm.tiling import TileManager
from oriens.maploc.utils.exif import EXIF
from oriens.maploc.utils.geo import BoundaryBox, Projection


class Localizer:
    def __init__(
        self,
        image,
        prior_latlon,
        focal_length=368,
        tile_size_meters=64,
        num_rotations=256,
        device="cuda",
    ):
        self.demo = Demo(num_rotations=num_rotations, device=device)
        self.image = image
        self.prior_latlon = prior_latlon
        self.focal_length = focal_length
        self.tile_size_meters = tile_size_meters

    def get_image_data(self):  # Reconstructed function of read_input_image
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        gravity, camera = self.demo.calibrator.run(image, self.focal_length)
        logger.info("Using (roll, pitch) %s.", gravity)

        latlon = self.prior_latlon
        proj = Projection(*latlon)
        center = proj.project(latlon)
        bbox = BoundaryBox(center, center) + self.tile_size_meters
        return image, camera, gravity, proj, bbox

    def localize(self):
        # Get the image data
        image, camera, gravity, proj, bbox = self.get_image_data()

        # Query OpenStreetMap for this area
        tiler = TileManager.from_bbox(
            proj, bbox + 10, self.demo.config.data.pixel_per_meter
        )
        canvas = tiler.query(bbox)

        # Run the inference
        uv, yaw, prob, neural_map, image_rectified = self.demo.localize(
            image, camera, canvas, gravity=gravity
        )

        latlon = proj.unproject(uv)

        return latlon, yaw
