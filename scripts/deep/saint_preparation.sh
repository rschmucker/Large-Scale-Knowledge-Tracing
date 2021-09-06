###############################################################################
## Additional prepartion step for SAINT+                                     ##
###############################################################################
# Args:                                                                       #
#    dataset (string): dataset name, available datasets:                      #
#      - elemmath_2021                                                        #
#      - ednet_kt3                                                            #
#      - junyi_15                                                             #
#      - eedi                                                                 #
###############################################################################

export PYTHONPATH="."
DATASET="elemmath_2021"

python ./src/preparation/prepare_saint.py --dataset=$DATASET
