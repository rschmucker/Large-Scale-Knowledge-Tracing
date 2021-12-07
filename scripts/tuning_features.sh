###############################################################################
## Tune average correct and n-gram features                                  ##
###############################################################################
# Args:                                                                       #
#    dataset (string): dataset name, available datasets:                      #
#      - elemmath_2021                                                        #
#      - ednet_kt3                                                            #
#      - junyi                                                                #
#      - eedi                                                                 #
#    split_id (int): train/test split to use                                  #
#    num_iterations (int): number of iterations to perform                    #
#                                                                             #
# One-hot features:                                                           #
#    -u: user one-hot                                                         #
#    -i: item one-hot                                                         #
#    -s: skill one-hot                                                        #
#    -sch: school one-hot                                                     #
#    -tea: teacher one-hot                                                    #
#    -sm: study module one-hot                                                #
#    -c: course id one-hot                                                    #
#    -d: difficulty one-hot                                                   #
#    -at: app-type one-hot                                                    #
#    -t: topic one-hot                                                        #
#    -bundle: bundle one-hot                                                  #
#    -part: TOIEC part one-hot                                                #
#                                                                             #
# User history features:                                                      #
#    -tcA: total count of previous attemts                                    #
#    -tcW: total count of previous wins                                       #
#    -scA: skill count of previous attemts                                    #
#    -scW: skill count of previous wins                                       #
#    -icA: item count of previous attemts                                     #
#    -icW: item count of previous wins                                        #
#                                                                             #
# Graph features:                                                             #
#    -pre: pre-req skill one-hot                                              #
#    -post: pre-req skill one-hot                                             #
#    -precA: pre-req skill count of previous attemts                          #
#    -precW: pre-req skill count of previous wins                             #
#    -postcA: post-req skill count of previous attemts                        #
#    -postcW: post-req skill count of previous wins                           #
#                                                                             #
# Time features:                                                              #
#    -resp_time: user response time in seconds                                #
#    -resp_time_cat: response time phi and categories                         # 
#    -lag_time: user lag time in seconds                                      #
#    -lag_time_cat: lag time phi and categories                               # 
#                                                                             #
# Datetime features:                                                          #
#    -month: month one-hot                                                    #
#    -week: week one-hot                                                      #
#    -day: day one-hot                                                        #
#    -hour: hour one-hot                                                      #
#    -weekend: weekend one-hot                                                #
#    -part_of_day: part of day one-hot                                        #
#                                                                             #
# Higher-order features:                                                      #
#    -user_avg_correct: average user correctness over time                    #
#    -n_gram: correctness patterns in sequence                                #
#                                                                             #
# Reading features:                                                           #
#    -rc: count readings on skill and total level                             #
#    -rt: total reading time on skill and total level                         #
#                                                                             #
# Video features:                                                             #
#    -vw: count watched videos                                                #
#    -vs: count skipped videos                                                #
#    -vt: watching time on skill and total level                              #
#                                                                             #
# Study module count features:                                                #
#    -smA: count previous attempts in this study module                       #
#    -smW: count previous wins in this study module                           #
#                                                                             #
###############################################################################
export PYTHONPATH="."
DATASET="elemmath_2021"
EXPNAME="tuning_rpfa"
NTHREADS=23

# RPFA:
features="-s -rpfa_F -rpfa_R"

python ./src/analysis/feature_tuning.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=0 --exp_name=$EXPNAME $features
