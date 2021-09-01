#!/bin/bash
###############################################################################
## Experiment: Evaluation of individual features                             ##
###############################################################################

# parameters
export PYTHONPATH="."
DATASET="junyi_15"
EXPNAME="single_feature_evaluation"
NPROCESSES=1     
NTHREADS=4
SPLITS=5

cmd=""
for (( i=0; i<$SPLITS; i++ )); do
#---------------------------------------------------------#
# One-hot features                                        #
#---------------------------------------------------------#
OH_FEATURES="-i -s -sm -part"
for f in $OH_FEATURES; do
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $f\n"
done

#---------------------------------------------------------#
# Count features                                          #
#---------------------------------------------------------#
# Comes in pairs and combined
for f in "-tcA -tcW" "-scA -scW" "-icA -icW" "-partcA -partcW"; do
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $f\n"
done

COUNT_FEATURES="-tcA -tcW -scA -scW -icA -icW"
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $COUNT_FEATURES\n"

#---------------------------------------------------------#
# Time window features                                    #
#---------------------------------------------------------#
# Comes in pairs and combined
TW_FEATURES="-tcA_TW -tcW_TW -scA_TW -scW_TW -icA_TW -icW_TW"
for f in "-tcA_TW -tcW_TW" "-scA_TW -scW_TW" "-icA_TW -icW_TW"; do
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $f\n"
done

cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $TW_FEATURES\n"

#---------------------------------------------------------#
# Interaction time features                               #
#---------------------------------------------------------#
IT_FEATURES="-resp_time_cat -prev_resp_time_cat -lag_time_cat -prev_lag_time_cat"
for f in $IT_FEATURES; do
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $f\n"
done

#---------------------------------------------------------#
# Datetime features                                       #
#---------------------------------------------------------#
# month, week, day, hour, weekend, part of day
DATE_FEATURES="-month -week -day -hour -weekend -part_of_day"
for f in $DATE_FEATURES; do
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $f\n"
done

#---------------------------------------------------------#
# Graph features                                          #
#---------------------------------------------------------#
for f in "-pre" "-post" "-precA -precW" "-postcA -postcW"; do
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $f\n"
done

#---------------------------------------------------------#
# Reading features                                        #
#---------------------------------------------------------#
READING_FEATURE="-rc -rt"
for f in $READING_FEATURE; do
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $f\n"
done

#---------------------------------------------------------#
# study module/part specific counts                       #
#---------------------------------------------------------#
SM_FEATURES="-smA -smW"
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $SM_FEATURES\n"

#---------------------------------------------------------#
# Student average correct                                 #
#---------------------------------------------------------#
AVERAGE_CORRRECT="-user_avg_correct"
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $AVERAGE_CORRRECT\n"

#---------------------------------------------------------#
# n-gram feature                                          #
#---------------------------------------------------------#
NGRAM="-n_gram"
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $NGRAM\n"

# -------------------------------------------------------#
done


echo -e $cmd
echo "============================="
echo "Waiting 10s before starting jobs...."
sleep 10s
echo -e $cmd | xargs -n1 -P$NPROCESSES -I{} -- bash -c '{}'
