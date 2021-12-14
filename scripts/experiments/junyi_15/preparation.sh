#!/bin/bash
###############################################################################
## Junyi 15: prepare data and extract all features                           ##
###############################################################################

# parameter
export PYTHONPATH="."
DATASET="junyi_15"
NTHREADS=31
SPLITS=5


# prepare data file
echo "starting data preparation"
python ./src/preparation/prepare_data.py \
    --dataset=$DATASET \
    --n_splits=$SPLITS


# available features classes
OH_FEATURES="-i -s -sm -part"
COUNT_FEATURES="-tcA -tcW -scA -scW -icA -icW -partcA -partcW"
TW_FEATURES="-tcA_TW -tcW_TW -scA_TW -scW_TW -icA_TW -icW_TW"
IT1_FEATURES="-resp_time -lag_time"
IT2_FEATURES="-resp_time_cat -prev_resp_time_cat -lag_time_cat -prev_lag_time_cat"
DATE_FEATURES="-month -week -day -hour -weekend -part_of_day"
GRAPH_FEATURE="-pre -post -precA -precW -postcA -postcW"
READING_FEATURE="-rc -rt"
SM_FEATURES="-smA -smW"
AVERAGE_CORRRECT="-user_avg_correct"
NGRAM="-n_gram"
RPFA="-rpfa_F -rpfa_R"
PPE="-ppe"

# select feature classes for extraction
FS=(
    "$OH_FEATURES"
    "$COUNT_FEATURES"
    "$TW_FEATURES"
    "$IT1_FEATURES"
    "$IT2_FEATURES"
    "$DATE_FEATURES"
    "$GRAPH_FEATURE"
    "$READING_FEATURE"
    "$SM_FEATURES"
    "$AVERAGE_CORRRECT"
    "$NGRAM"
    "$RPFA"
    "$PPE"
)


# extract features
echo "starting feature extraction"
for f in "${FS[@]}"; do
    python ./src/preprocessing/extract_features.py \
        --dataset=$DATASET \
        --num_threads=$NTHREADS \
        -recompute \
        $f
done
