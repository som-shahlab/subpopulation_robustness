#!/bin/bash

cd ../

slurm_pre="--partition cpu --mem 40gb -c 4 --job-name bootstrap --qos nopreemption --output /scratch/ssd001/home/haoran/projects/group_robustness_fairness/group_robustness_fairness/logs/bootstrap_%A.log"

python sweep.py launch \
    --experiment Bootstrap \
    --slurm_pre "${slurm_pre}" \
    --command_launcher "slurm" \
    --no_output_dir 
