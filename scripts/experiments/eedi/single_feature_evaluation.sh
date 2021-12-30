#!/bin/bash
###############################################################################
## Experiment: Evaluation of individual features                             ##
###############################################################################

# parameters
export PYTHONPATH="."
DATASET="eedi"
EXPNAME="single_feature_evaluation"
NPROCESSES=1
NTHREADS=4
SPLITS=5

cmd=""
for (( i=0; i<$SPLITS; i++ )); do
#---------------------------------------------------------#
# One-hot features                                        #
#---------------------------------------------------------#
OH_FEATURES="-i -s -tea -sm -bundle -age -gender -ss"
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
COUNT_FEATURES="-tcA -tcW -scA -scW -icA -icW"
for f in "-tcA -tcW" "-scA -scW" "-icA -icW"; do
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

#---------------------------------------------------------#
# PPE feature                                             #
#---------------------------------------------------------#
PPE="-ppe"
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $PPE\n"

#---------------------------------------------------------#
# RPFA features                                           #
#---------------------------------------------------------#
# Comes in pairs and combined
RPFA="-rpfa_F -rpfa_R"
for f in "-rpfa_F" "-rpfa_R"; do
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
    --exp_name=$EXPNAME $RPFA\n"

# -------------------------------------------------------#

done


echo -e $cmd
echo "============================="
echo "Waiting 10s before starting jobs...."
sleep 10s
echo -e $cmd | xargs -n1 -P$NPROCESSES -I{} -- bash -c '{}'
