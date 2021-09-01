# This seed is used for the train/test splits. Hold it constant
SEED = 888
MIN_INTERACTIONS_PER_USER = 10
NAN_VAL = -888

# identifiers for squirrel events
READING = 1
WATCH_VIDEO = 5
SKIP_VIDEO = 6

# List of available features
ONE_HOT_FEATURES = ['u', 'i', 's', 'sch', 'tea', 'sm', 'c', 'd', 'at', 't',
                    'bundle', 'part', 'ss', 'age', 'gender', 'user_skill']
COUNT_FEATURES = ['tcA', 'tcW', 'scA', 'scW', 'icA', 'icW', 'partcA', 'partcW']
GRAPH_FEATURES = ['pre', 'post', 'precA', 'precW', 'postcA', 'postcW']
TIME_FEATURES = ['resp_time', 'resp_time_cat', 'prev_resp_time_cat',
                 'lag_time', 'lag_time_cat', 'prev_lag_time_cat']
TIME_WINDOW_FEATURES = ['tcA_TW', 'tcW_TW', 'scA_TW', 'scW_TW',
                        'icA_TW', 'icW_TW']
DATETIME_FEATURES = ['month', 'week', 'day', 'hour', 'weekend', 'part_of_day']
VIDEO_FEATURES = ['vw', 'vs', 'vt']
STUDY_MODULE_FEATURES = ['smA', 'smW']
READING_FEATURES = ["rc", "rt"]
ALL_FEATURES = ONE_HOT_FEATURES + COUNT_FEATURES + TIME_FEATURES \
   + TIME_WINDOW_FEATURES + VIDEO_FEATURES \
   + STUDY_MODULE_FEATURES + DATETIME_FEATURES \
   + READING_FEATURES + GRAPH_FEATURES + ["user_avg_correct", "n_gram", "ones"]

# Paths for data access
PREPARATION_PATH = "./data/preparation/"
DATASET_PATH = {
    "squirrel": "./data/squirrel/",
    "ednet_kt3": "./data/ednet_kt3/",
    "junyi_15": "./data/junyi_15/",
    "junyi_20": "./data/junyi_20/",
    "eedi": "./data/eedi/",
}
N_SM = {
    "squirrel": 6,
    "ednet_kt3": 8,
    "eedi": 59,
    "junyi_15": 8,
}
