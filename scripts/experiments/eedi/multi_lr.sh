#!/bin/bash
###############################################################################
## Experiment: Train multiple regression models that cover the dataset       ##
###############################################################################

# parameters
export PYTHONPATH="."
DATASET="eedi"
NPROCESSES=8
NTHREADS=3
SPLITS=5

BESTLR="-i -s -scA -scW -tcA -tcW"

for (( i=0; i<$SPLITS; i++ )); do
#---------------------------------------------------------#
# Partitionings                                           #
#---------------------------------------------------------#
for f in "i" "hashed_skill_id" "sm" "tea" "bundle_id" "single" "time"; do
cmd+="python ./src/training/compute_multi_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --col=$f $BESTLR\n"
done
# -------------------------------------------------------#
done


echo -e $cmd
echo "============================="
echo "Waiting 10s before starting jobs...."
sleep 10s
echo -e $cmd | xargs -n1 -P$NPROCESSES -I{} -- bash -c '{}'
