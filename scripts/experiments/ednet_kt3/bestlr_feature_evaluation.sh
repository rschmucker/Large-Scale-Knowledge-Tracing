#!/bin/bash
###############################################################################
## Experiment: Evaluation of features combined with BestLR                   ##
###############################################################################

# parameters
export PYTHONPATH="."
DATASET="ednet_kt3"
EXPNAME="bestlr_feature_evaluation"
NPROCESSES=1
NTHREADS=4
SPLITS=5

BESTLR="-i -s -scA -scW -tcA -tcW"

cmd=""
for (( i=0; i<$SPLITS; i++ )); do
#---------------------------------------------------------#
# One-hot features                                        #
#---------------------------------------------------------#
OH_FEATURES="-sm -at -bundle -part"
for f in $OH_FEATURES; do
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $BESTLR $f\n"
done

#---------------------------------------------------------#
# Count features                                          #
#---------------------------------------------------------#
COUNT_FEATURES="-partcA -partcW"
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $BESTLR $COUNT_FEATURES\n"

COUNT_FEATURES="-icA -icW "
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $BESTLR $COUNT_FEATURES\n"

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
    --exp_name=$EXPNAME $BESTLR $f\n"
done

cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $BESTLR $TW_FEATURES\n"

#---------------------------------------------------------#
# Interaction time features                               #
#---------------------------------------------------------#
IT_FEATURES="-resp_time_cat -prev_resp_time_cat -lag_time_cat -prev_lag_time_cat"
for f in $IT_FEATURES; do
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $BESTLR $f\n"
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
    --exp_name=$EXPNAME $BESTLR $f\n"
done

#---------------------------------------------------------#
# Video features                                          #
#---------------------------------------------------------#
VIDEO_FEATURES="-vw -vs -vt"
for f in $VIDEO_FEATURES; do
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $BESTLR $f\n"
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
    --exp_name=$EXPNAME $BESTLR $f\n"
done

#---------------------------------------------------------#
# study module/part specific counts                       #
#---------------------------------------------------------#
SM_FEATURES="-smA -smW"
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $BESTLR $SM_FEATURES\n"

#---------------------------------------------------------#
# Student average correct                                 #
#---------------------------------------------------------#
AVERAGE_CORRRECT="-user_avg_correct"
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $BESTLR $AVERAGE_CORRRECT\n"

#---------------------------------------------------------#
# n-gram feature                                          #
#---------------------------------------------------------#
NGRAM="-n_gram"
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $BESTLR $NGRAM\n"

#---------------------------------------------------------#
# PPE feature                                             #
#---------------------------------------------------------#
PPE="-ppe"
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $BESTLR $PPE\n"

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
    --exp_name=$EXPNAME $BESTLR $f\n"
done

cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $BESTLR $RPFA\n"

# -------------------------------------------------------#
done


echo -e $cmd
echo "============================="
echo "Waiting 10s before starting jobs...."
sleep 10s
echo -e $cmd | xargs -n1 -P$NPROCESSES -I{} -- bash -c '{}'
