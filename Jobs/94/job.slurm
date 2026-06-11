#!/bin/bash
#SBATCH --job-name=FemurReg-94
#SBATCH --mail-type=ALL
#SBATCH --mail-user=baptiste.tron@insa-lyon.fr
#SBATCH --partition=mesonet
#SBATCH --account=M25233
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --output=/home/bapttron/prjcapstone/Femur-Regression-Project/Jobs/94/log.out
#SBATCH --error=/home/bapttron/prjcapstone/Femur-Regression-Project/Jobs/94/log.out

cd /home/bapttron/prjcapstone/Femur-Regression-Project
apptainer exec --nv /home/bapttron/prjcapstone/Femur-Regression-Project/pytorch_25.09-py3.sif python /home/bapttron/prjcapstone/Femur-Regression-Project/train_regression_random.py --model pointnet2_reg --batch_size 64 --epoch 500 --learning_rate 0.01 --num_point 6000 --log_dir /home/bapttron/prjcapstone/Femur-Regression-Project/Jobs/94 --process_data 
