#!/bin/bash
###############################################################################
## Experiment: Evaluate augmented logistic regression                        ##
###############################################################################

# parameters
export PYTHONPATH="."
EXPNAME="lr_baselines"
DATASET="ednet_kt3"
NPROCESSES=5     
NTHREADS=5
SPLITS=5

cmd=""

EDLR="-i -s \
      -icA_TW -icW_TW -scA_TW -scW_TW -tcA_TW -tcW_TW \
      -sm \
      -lag_time_cat -prev_resp_time_cat \
      -partcA -partcW -vw -vs \
      -user_avg_correct -n_gram -rpfa_F -rpfa_R -ppe"

for (( i=0; i<$SPLITS; i++ )); do
#-----------------------------------------------------------
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $EDLR \n"
#-----------------------------------------------------------
done


echo -e $cmd
echo "============================="
echo "Waiting 10s before starting jobs...."
sleep 10s
echo -e $cmd | xargs -n1 -P$NPROCESSES -I{} -- bash -c '{}'
