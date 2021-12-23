clear
export PYTHONPATH="."
device_id=0
for dataset in elemmath_2021 ednet_kt3 eedi junyi_15
do
  CUDA_VISIBLE_DEVICES=$device_id python ./src/training/train_dkt2.py \
    --dataset=$dataset \
    --total_split=5
done
