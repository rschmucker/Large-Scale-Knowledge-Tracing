clear
export PYTHONPATH="."
device_id=3
for dataset in junyi_15 squirrel ednet_kt3
do
  CUDA_VISIBLE_DEVICES=$device_id python ./src/training/train_saint_plus.py \
    --dataset=$dataset \
    --batch_size=128 \
    --total_split=5
done
