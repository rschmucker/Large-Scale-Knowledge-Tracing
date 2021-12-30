#!/bin/bash
###############################################################################
## Experiment: Train multiple regression models that cover the dataset       ##
###############################################################################

# parameters
export PYTHONPATH="."
DATASET="junyi_15"
NPROCESSES=8
NTHREADS=3
SPLITS=5

BESTLR="-i -s -scA -scW -tcA -tcW"

#---------------------------------------------------------#
# Partitionings                                           #
#---------------------------------------------------------#
for f in "i" "s" "sm" "part_id" "single" "time"; do
for (( i=0; i<$SPLITS; i++ )); do
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
