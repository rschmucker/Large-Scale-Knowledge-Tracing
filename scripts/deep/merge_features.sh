###############################################################################
## Additional prepartion step for SAINT+                                     ##
###############################################################################
# Args:                                                                       #
#    dataset (string): dataset name, available datasets:                      #
#      - squirrel                                                             #
#      - ednet_kt3                                                            #
#      - junyi_15                                                             #
#      - eedi                                                                 #
###############################################################################

export PYTHONPATH="."
DATASET="squirrel"

python ./src/preparation/merge_feature.py --dataset=$DATASET
