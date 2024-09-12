import rosbag
import pandas as pd


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
                    msg.lat,
                    msg.lon,
                    msg.alt,
                ]
            )
        self.gps_data = pd.DataFrame(
            self.gps_data, columns=["sec", "nsec", "lat", "lon", "alt"]
        )
        return self.gps_data

    def get_image_dataframe(self):
        for _, msg, t in self.bag.read_messages(
            topics=["/zedx/zed_node/left/image_rect_color"]
        ):
            self.camera_data.append(
                [
                    msg.header.stamp.secs,
                    msg.header.stamp.nsecs,
                    msg.data,
                ]
            )
        self.camera_data = pd.DataFrame(
            self.camera_data, columns=["sec", "nsec", "data"]
        )
        return self.camera_data
