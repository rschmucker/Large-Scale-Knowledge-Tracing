#!/bin/bash
###############################################################################
## Experiment: Evaluate rich logistic regression                             ##
###############################################################################

# parameters
export PYTHONPATH="."
EXPNAME="lr_baselines"
DATASET="elemmath_2021"
NPROCESSES=5     
NTHREADS=5
SPLITS=5

cmd=""

SQLR="-i -s \
      -icA_TW -icW_TW -scA_TW -scW_TW -tcA_TW -tcW_TW \
      -sm
      -lag_time_cat -prev_resp_time_cat \
      -t -precA -precW -postcA -postcW -rc -vw \
      -user_avg_correct -n_gram -rpfa_F -rpfa_R -ppe"

for (( i=0; i<$SPLITS; i++ )); do
#-----------------------------------------------------------
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $SQLR \n"
#-----------------------------------------------------------
done


echo -e $cmd
echo "============================="
echo "Waiting 10s before starting jobs...."
sleep 10s
echo -e $cmd | xargs -n1 -P$NPROCESSES -I{} -- bash -c '{}'
