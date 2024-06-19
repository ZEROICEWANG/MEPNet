# echo 1 > /proc/sys/vm/drop_caches
# echo 2 > /proc/sys/vm/drop_caches
# echo 3 > /proc/sys/vm/drop_caches

python sleep.py --base 0 --subbase 0 --iter 0
python -m torch.distributed.launch --nproc_per_node 2 train.py --config-file ./config/standard.yaml  --gpus '0,1'