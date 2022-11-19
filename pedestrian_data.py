import os
import json
import codecs

PEDESTRIAN_DATA_PATH = None
with codecs.open(r"config/config.json", 'r', 'utf-8') as config_file:
    config_data = json.load(config_file)
    PEDESTRIAN_DATA_PATH = config_data["Data-Path"]


__SCENARIOS_FILTERED = {".git"}
scenarios = {scenario: [sample for sample in os.listdir(os.path.join(PEDESTRIAN_DATA_PATH, scenario))
                        if os.path.isdir(os.path.join(PEDESTRIAN_DATA_PATH, scenario, sample))] for scenario in
             filter(lambda f: f not in __SCENARIOS_FILTERED, os.listdir(PEDESTRIAN_DATA_PATH))
             if os.path.isdir(os.path.join(PEDESTRIAN_DATA_PATH, scenario))}