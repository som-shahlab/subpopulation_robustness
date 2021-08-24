#!/bin/bash

cd ../

echo "Running experiment:" "${1}"

slurm_pre="--partition t4v2,t4v1 --exclude=gpu021,gpu115 --gres gpu:1 --mem 40gb -c 4 --job-name ${1} --output /scratch/ssd001/home/haoran/projects/group_robustness_fairness/group_robustness_fairness/logs/${1}_%A.log"

python sweep.py launch \
    --experiment ${1} \
    --output_dir "/scratch/hdd001/home/haoran/stanford_robustness/results/${1}/" \
    --slurm_pre "${slurm_pre}" \
    --command_launcher "slurm" 
