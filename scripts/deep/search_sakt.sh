# Search for best SAKT parameters

clear
export PYTHONPATH="."

SPLITS=5
NUM_EPOCHS=100

# 0 1 2 3
device_id=1
# 128 256
BATCH_SIZE=128
# 1 2
LAYERS=1


for nodes in 50 100 200 500; do
for dropout in 0 0.20 0.5; do
#---------------------------------------------------------#
for dataset in elemmath_2021 ednet_kt3 eedi junyi_15
do
  CUDA_VISIBLE_DEVICES=$device_id python ./src/training/train_sakt.py \
    --dataset=$dataset \
    --total_split=$SPLITS \
    --num_epochs=$NUM_EPOCHS \
    --batch_size=$BATCH_SIZE \
    --num_attn_layers=$LAYERS \
    --embed_size=$nodes \
    --drop_prob=$dropout
done
#---------------------------------------------------------#
done
done
