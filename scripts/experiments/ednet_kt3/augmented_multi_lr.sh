#!/bin/bash
###############################################################################
## Experiment: Train multiple augmented models that cover the dataset        ##
###############################################################################

# parameters
export PYTHONPATH="."
DATASET="ednet_kt3"
NPROCESSES=8
NTHREADS=3
SPLITS=5

cmd=""

EDLR="-i -s \
      -icA_TW -icW_TW -scA_TW -scW_TW -tcA_TW -tcW_TW \
      -sm \
      -lag_time_cat -prev_resp_time_cat \
      -partcA -partcW -vw\
      -user_avg_correct -n_gram"

#---------------------------------------------------------#
# Partitionings                                           #
#---------------------------------------------------------#
for f in "i" "hashed_skill_id" "sm" "bundle_id" "part_id" "at" "single" "time"; do
for (( i=0; i<$SPLITS; i++ )); do
cmd+="python ./src/training/compute_multi_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --col=$f $EDLR\n"
done
# -------------------------------------------------------#
done


echo -e $cmd
echo "============================="
echo "Waiting 10s before starting jobs...."
sleep 10s
echo -e $cmd | xargs -n1 -P$NPROCESSES -I{} -- bash -c '{}'
