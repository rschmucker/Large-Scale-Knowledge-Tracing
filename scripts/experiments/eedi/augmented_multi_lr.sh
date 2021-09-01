#!/bin/bash
###############################################################################
## Experiment: Train multiple augmented models that cover the dataset        ##
###############################################################################

# parameters
export PYTHONPATH="."
DATASET="eedi"
NPROCESSES=8
NTHREADS=3
SPLITS=5

cmd=""

EELR="-i -s \
      -icA_TW -icW_TW -scA_TW -scW_TW -tcA_TW -tcW_TW \
      -sm \
      -tea -bundle \
      -precA -precW \
      -user_avg_correct -n_gram"

for (( i=0; i<$SPLITS; i++ )); do
#---------------------------------------------------------#
# Partitionings                                           #
#---------------------------------------------------------#
for f in "i" "hashed_skill_id" "sm" "tea" "bundle_id" "single" "time"; do
cmd+="python ./src/training/compute_multi_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --col=$f $EELR\n"
done
# -------------------------------------------------------#
done


echo -e $cmd
echo "============================="
echo "Waiting 10s before starting jobs...."
sleep 10s
echo -e $cmd | xargs -n1 -P$NPROCESSES -I{} -- bash -c '{}'
