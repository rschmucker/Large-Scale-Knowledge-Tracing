# Search for best SAINT parameters

clear
export PYTHONPATH="."

SPLITS=5
NUM_EPOCHS=100

# 0 1 2 3
device_id=2
# 128 256
BATCH_SIZE=128
# 2 4 6
LAYERS=4

for nodes in 64 128 256 512; do
#---------------------------------------------------------#
for dataset in elemmath_2021 ednet_kt3 eedi junyi_15; do
  CUDA_VISIBLE_DEVICES=$device_id python ./src/training/train_saint.py \
    --dataset=$dataset \
    --total_split=$SPLITS \
    --num_epochs=$NUM_EPOCHS \
    --batch_size=$BATCH_SIZE \
    --encoder_layer=$LAYERS \
    --decoder_layer=$LAYERS \
    --model_size=$nodes
done
#---------------------------------------------------------#
done
