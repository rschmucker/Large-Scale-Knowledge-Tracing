###############################################################################
## Analysis of feature evaluation experiments                                ##
###############################################################################
# Args:                                                                       #
#    dataset (string): dataset name, available datasets:                      #
#      - squirrel                                                             #
#      - ednet_kt3                                                            #
#      - junyi_15                                                             #
#      - junyi_20                                                             #
#      - eedi                                                                 #
#    exp_name (string): name of evaluation experiment:                        #
#      - single_feature_evaluation                                            #
#      - bestlr_feature_evaluation                                            #
#      - lr_baselines                                                         #
#    exp_name (int): number of cross-validation splits                        #
###############################################################################

export PYTHONPATH="."
DATASET="squirrel"
EXPNAME="single_feature_evaluation"
NSPLITS=5


python ./src/analysis/feature_evaluation.py \
    --dataset=$DATASET \
    --exp_name=$EXPNAME \
    --n_splits=$NSPLITS
