import json
import codecs

PEDESTRIAN_DATA_PATH = None
with codecs.open(r"config/config.json", 'r', 'utf-8') as config_file:
    config_data = json.load(config_file)
    PEDESTRIAN_DATA_PATH = config_data["Data-Path"]
