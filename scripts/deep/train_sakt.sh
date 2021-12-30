clear
export PYTHONPATH="."

SPLITS=5
NUM_EPOCHS=100
DEVICE_ID=1


DATASET="elemmath_2021"
BATCH_SIZE=128
LAYERS=2
NODES=200
DROPOUT=0.2
CUDA_VISIBLE_DEVICES=$DEVICE_ID python ./src/training/train_sakt.py \
  --dataset=$DATASET \
  --total_split=$SPLITS \
  --num_epochs=$NUM_EPOCHS \
  --batch_size=$BATCH_SIZE \
  --num_attn_layers=$LAYERS \
  --embed_size=$NODES \
  --drop_prob=$DROPOUT


DATASET="ednet_kt3"
BATCH_SIZE=128
LAYERS=1
NODES=100
DROPOUT=0.5
CUDA_VISIBLE_DEVICES=$DEVICE_ID python ./src/training/train_sakt.py \
  --dataset=$DATASET \
  --total_split=$SPLITS \
  --num_epochs=$NUM_EPOCHS \
  --batch_size=$BATCH_SIZE \
  --num_attn_layers=$LAYERS \
  --embed_size=$NODES \
  --drop_prob=$DROPOUT


DATASET="eedi"
BATCH_SIZE=256
LAYERS=2
NODES=200
DROPOUT=0.2
CUDA_VISIBLE_DEVICES=$DEVICE_ID python ./src/training/train_sakt.py \
  --dataset=$DATASET \
  --total_split=$SPLITS \
  --num_epochs=$NUM_EPOCHS \
  --batch_size=$BATCH_SIZE \
  --num_attn_layers=$LAYERS \
  --embed_size=$NODES \
  --drop_prob=$DROPOUT



DATASET="junyi_15"
BATCH_SIZE=256
LAYERS=2
NODES=100
DROPOUT=0.0
CUDA_VISIBLE_DEVICES=$DEVICE_ID python ./src/training/train_sakt.py \
  --dataset=$DATASET \
  --total_split=$SPLITS \
  --num_epochs=$NUM_EPOCHS \
  --batch_size=$BATCH_SIZE \
  --num_attn_layers=$LAYERS \
  --embed_size=$NODES \
  --drop_prob=$DROPOUT

