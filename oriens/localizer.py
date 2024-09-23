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
        focal_length=None,
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

        # from oriens.maploc.osm.viz import GeoPlotter
        # from oriens.maploc.osm.viz import Colormap, plot_nodes
        # from oriens.maploc.utils.viz_2d import plot_images

        # plot = GeoPlotter(zoom=16)
        # plot.points(proj.latlonalt[:2], "red", name="location prior", size=10)
        # plot.bbox(proj.unproject(bbox), "blue", name="map tile")
        # plot.fig.show()

        # tiler = TileManager.from_bbox(
        #     proj, bbox + 10, self.demo.config.data.pixel_per_meter
        # )
        # map_viz = Colormap.apply(canvas.raster)
        # plot_images([image, map_viz], titles=["input image", "OpenStreetMap raster"])
        # plot_nodes(1, canvas.raster[2], fontsize=6, size=10)

        # from oriens.maploc.utils.viz_localization import (
        #     likelihood_overlay,
        #     plot_dense_rotations,
        #     add_circle_inset,
        # )
        # from oriens.maploc.utils.viz_2d import features_to_RGB

        # # Visualize the predictions
        # overlay = likelihood_overlay(
        #     prob.numpy().max(-1), map_viz.mean(-1, keepdims=True)
        # )
        # (neural_map_rgb,) = features_to_RGB(neural_map.numpy())
        # plot_images([overlay, neural_map_rgb], titles=["prediction", "neural map"])
        # ax = plt.gcf().axes[0]
        # ax.scatter(*canvas.to_uv(bbox.center), s=5, c="red")
        # plot_dense_rotations(ax, prob, w=0.005, s=1 / 25)
        # add_circle_inset(ax, uv)
        # plt.savefig("oriens/experiments/localization.png", dpi=300, bbox_inches="tight")

        # # Plot as interactive figure
        # bbox_latlon = proj.unproject(canvas.bbox)
        # plot = GeoPlotter(zoom=16.5)
        # plot.raster(map_viz, bbox_latlon, opacity=0.5)
        # plot.raster(likelihood_overlay(prob.numpy().max(-1)), proj.unproject(bbox))
        # plot.points(proj.latlonalt[:2], "red", name="location prior", size=10)
        # plot.points(proj.unproject(canvas.to_xy(uv)), "black", name="argmax", size=10)
        # plot.bbox(bbox_latlon, "blue", name="map tile")
        # plot.fig.show()

        return latlon, yaw