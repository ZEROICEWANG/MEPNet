
python sleep.py --base 0 --subbase 7.8  --iter 0
dir=($(ls -A ./models/RES_Model/))

python predict.py --name ${dir[-1]} --model CPD_RES_PA --config-file 'config/standard.yaml'
