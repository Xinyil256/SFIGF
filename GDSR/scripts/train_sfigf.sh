set -euxo pipefail

# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main.py --name sfigf_sca_ch48_4 --model GIRNet_sca --scale 4 --sample_q 30720 --input_size 256 --train_batch 1 --epoch 400 --eval_interval 10 --lr 0.0003 --lr_step 60 --lr_gamma 0.2 --train_batch=1 --base_channel 48 
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main.py --name sfigf_sca_ch48_8 --model GIRNet_sca --scale 8 --sample_q 30720 --input_size 256 --train_batch 1 --epoch 400 --eval_interval 10 --lr 0.0003 --lr_step 60 --lr_gamma 0.2 --train_batch=1 --base_channel 48
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main.py --name girnet_sca_ch32_0802_16 --model GIRNet_sca --scale 16 --sample_q 30720 --input_size 256 --train_batch 1 --epoch 400 --eval_interval 10 --lr 0.0003 --lr_step 60 --lr_gamma 0.2 --train_batch 1 --base_channel 32