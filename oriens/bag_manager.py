import bagpy
from bagpy import bagreader
import pandas as pd


class BagManager:
    def __init__(self, bag_file):
        self.bag_reader = bagreader(bag_file)
        self.topic_table = self.bag_reader.topic_table

    def get_dataframe(self, topic):
        return pd.read_csv(self.bag_reader.message_by_topic(topic))
