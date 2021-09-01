###############################################################################
## Compute dataset sequence entropy and sequence predictabilty               ##
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

python ./src/analysis/sequence_entropy.py \
    --dataset=$dataset
