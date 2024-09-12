import rosbag
import pandas as pd
import sensor_msgs.msg
import cv2
from cv_bridge import CvBridge


class BagManager:
    def __init__(self, bag_file):
        self.bag = rosbag.Bag(bag_file)
        self.gps_data = []
        self.camera_data = []

    def get_gps_dataframe(self):
        for _, msg, t in self.bag.read_messages(topics=["/mavros/sdcstatus/gpsrawint"]):
            self.gps_data.append(
                [
                    msg.header.stamp.secs,
                    msg.header.stamp.nsecs,
                    msg.lat / 1e7,
                    msg.lon / 1e7,
                ]
            )
        self.gps_data = pd.DataFrame(
            self.gps_data, columns=["sec", "nsec", "lat", "lon"]
        )
        return self.gps_data

    def get_image_dataframe(self):
        bridge = CvBridge()

        for _, msg, t in self.bag.read_messages(
            topics=["/zedx/zed_node/left/image_rect_color"]
        ):
            img_msg = sensor_msgs.msg.Image()
            img_msg.height = msg.height
            img_msg.width = msg.width
            img_msg.encoding = msg.encoding
            img_msg.is_bigendian = msg.is_bigendian
            img_msg.step = msg.step
            img_msg.data = msg.data

            img = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

            self.camera_data.append(
                [msg.header.stamp.secs, msg.header.stamp.nsecs, img]
            )
        self.camera_data = pd.DataFrame(
            self.camera_data, columns=["sec", "nsec", "img"]
        )
        return self.camera_data
