#!/bin/bash -l

#SBATCH --job-name=Co2L_res18_cifar10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32gb

#SBATCH --gres=gpu:1
#SBATCH --time=3:59:59

#SBATCH --mail-type=ALL
#SBATCH --mail-user=roy034@csiro.au

source /scratch2/roy034/anaconda3/etc/profile.d/conda.sh
module load cuda/10.2.89 cudnn/7.6.5-cuda102
conda activate py39

# python main_general.py --resume_target_task 1 --batch_size 512 --model resnet18 --dataset cifar10 --mem_size 200 --epochs 100 --start_epoch 500 --learning_rate 0.5 --temp 0.5 --current_temp 0.2 --past_temp 0.01  --cosine --syncBN

python main_linear_buffer.py --learning_rate 1 --target_task 0
python main_linear_buffer.py --learning_rate 1 --target_task 1
python main_linear_buffer.py --learning_rate 1 --target_task 2
python main_linear_buffer.py --learning_rate 1 --target_task 3
python main_linear_buffer.py --learning_rate 1 --target_task 4
