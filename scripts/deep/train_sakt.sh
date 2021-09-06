clear
export PYTHONPATH="."
device_id=1
for dataset in elemmath_2021 ednet_kt3 eedi junyi_15
do
  CUDA_VISIBLE_DEVICES=$device_id python ./src/training/train_sakt.py \
    --dataset=$dataset \
    --total_split=5 \
    --batch_size=500
done