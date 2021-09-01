###############################################################################
## Preprare data for feature extraction                                      ##
###############################################################################
# Args:                                                                       #
#    dataset (string): dataset name, available datasets:                      #
#      - squirrel                                                             #
#      - ednet_kt3                                                            #
#      - junyi_15                                                             #
#      - junyi_20                                                             #
#      - eedi                                                                 #
#    n_splits (int): number of splits for cross-validation                    #
###############################################################################
export PYTHONPATH="."

dataset="squirrel"

python ./src/preparation/prepare_data.py \
    --dataset=$dataset \
    --n_splits=5
