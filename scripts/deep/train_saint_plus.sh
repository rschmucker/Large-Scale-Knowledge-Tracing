clear
export PYTHONPATH="."

SPLITS=5
NUM_EPOCHS=100
DEVICE_ID=3


DATASET="elemmath_2021"
BATCH_SIZE=256
LAYERS=6
NODES=64
CUDA_VISIBLE_DEVICES=$DEVICE_ID python ./src/training/train_saint_plus.py \
  --dataset=$DATASET \
  --total_split=$SPLITS \
  --num_epochs=$NUM_EPOCHS \
  --batch_size=$BATCH_SIZE \
  --encoder_layer=$LAYERS \
  --decoder_layer=$LAYERS \
  --model_size=$NODES


DATASET="ednet_kt3"
BATCH_SIZE=256
LAYERS=6
NODES=64
CUDA_VISIBLE_DEVICES=$DEVICE_ID python ./src/training/train_saint_plus.py \
  --dataset=$DATASET \
  --total_split=$SPLITS \
  --num_epochs=$NUM_EPOCHS \
  --batch_size=$BATCH_SIZE \
  --encoder_layer=$LAYERS \
  --decoder_layer=$LAYERS \
  --model_size=$NODES


DATASET="junyi_15"
BATCH_SIZE=256
LAYERS=6
NODES=64
CUDA_VISIBLE_DEVICES=$DEVICE_ID python ./src/training/train_saint_plus.py \
  --dataset=$DATASET \
  --total_split=$SPLITS \
  --num_epochs=$NUM_EPOCHS \
  --batch_size=$BATCH_SIZE \
  --encoder_layer=$LAYERS \
  --decoder_layer=$LAYERS \
  --model_size=$NODES
