clear
export PYTHONPATH="."
device_id=2
for dataset in elemmath_2021 ednet_kt3 eedi junyi_15
do
  CUDA_VISIBLE_DEVICES=$device_id python ./src/training/train_saint.py \
    --dataset=$dataset \
    --batch_size=128 \
    --total_split=5
done
