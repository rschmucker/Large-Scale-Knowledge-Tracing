###############################################################################
## Preprare data for feature extraction                                      ##
###############################################################################
# Args:                                                                       #
#    dataset (string): dataset name, available datasets:                      #
#      - elemmath_2021                                                        #
#      - ednet_kt3                                                            #
#      - junyi_15                                                             #
#      - junyi_20                                                             #
#      - eedi                                                                 #
#    n_splits (int): number of splits for cross-validation                    #
###############################################################################
export PYTHONPATH="."

dataset="elemmath_2021"

python ./src/preparation/prepare_data.py \
    --dataset=$dataset \
    --n_splits=5
