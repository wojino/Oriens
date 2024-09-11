import matplotlib.pyplot as plt

from maploc.demo import Demo
from maploc.osm.viz import GeoPlotter
from maploc.osm.tiling import TileManager
from maploc.osm.viz import Colormap, plot_nodes
from maploc.utils.viz_2d import plot_images, features_to_RGB
from maploc.utils.viz_localization import (
    likelihood_overlay,
    plot_dense_rotations,
    add_circle_inset,
)


# Increasing the number of rotations increases the accuracy but requires more GPU memory.
# The highest accuracy is achieved with num_rotations=360
# but num_rotations=64~128 is often sufficient.
# To reduce the memory usage, we can reduce the tile size in the next cell.
demo = Demo(num_rotations=256, device="cpu")

image_path = "experiments/query_zurich_1.jpg"
prior_address = "ETH CAB Zurich"

image, camera, gravity, proj, bbox = demo.read_input_image(
    image_path,
    prior_address=prior_address,
    tile_size_meters=128,  # try 64, 256, etc.
)

# Query OpenStreetMap for this area
tiler = TileManager.from_bbox(proj, bbox + 10, demo.config.data.pixel_per_meter)
canvas = tiler.query(bbox)

# Show the inputs to the model: image and raster map
map_viz = Colormap.apply(canvas.raster)
plot_images([image, map_viz], titles=["input image", "OpenStreetMap raster"])
plot_nodes(1, canvas.raster[2], fontsize=6, size=10)

# Run the inference
uv, yaw, prob, neural_map, image_rectified = demo.localize(
    image, camera, canvas, gravity=gravity
)

# Visualize the predictions
overlay = likelihood_overlay(prob.numpy().max(-1), map_viz.mean(-1, keepdims=True))
(neural_map_rgb,) = features_to_RGB(neural_map.numpy())
plot_images([overlay, neural_map_rgb], titles=["prediction", "neural map"])
ax = plt.gcf().axes[0]
ax.scatter(*canvas.to_uv(bbox.center), s=5, c="red")
plot_dense_rotations(ax, prob, w=0.005, s=1 / 25)
add_circle_inset(ax, uv)
plt.savefig("experiments/localization.png", dpi=300, bbox_inches="tight")
