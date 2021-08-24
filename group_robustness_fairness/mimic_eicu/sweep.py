import argparse
import copy
import getpass
import hashlib
import json
import os
import random
import shutil
import time
import uuid

import numpy as np
import torch

import tqdm
import shlex
import group_robustness_fairness.mimic_eicu.experiments
import group_robustness_fairness.mimic_eicu.launchers


class Job:
    NOT_LAUNCHED = "Not launched"
    INCOMPLETE = "Incomplete"
    DONE = "Done"

    def __init__(
        self, train_args, sweep_output_dir, slurm_pre, script_name, no_output_dir=False
    ):
        args_str = json.dumps(train_args, sort_keys=True)
        args_hash = hashlib.md5(args_str.encode("utf-8")).hexdigest()
        self.no_output_dir = no_output_dir
        self.train_args = copy.deepcopy(train_args)
        if not no_output_dir:
            self.output_dir = os.path.join(sweep_output_dir, args_hash)
            self.train_args["output_dir"] = self.output_dir
        command = ["python", script_name]
        for k, v in sorted(self.train_args.items()):
            if isinstance(v, (list, tuple)):  # "arg": ["a","b"] ->  --arg a b c
                v = " ".join([str(v_) for v_ in v])
            elif isinstance(v, str):  # "arg": "a" ->  --arg a
                v = shlex.quote(v)

            if k:
                if not isinstance(v, bool):
                    command.append(f"--{k} {v}")
                else:
                    if v:  # "arg": True ->  --arg
                        command.append(f"--{k}")
                    else:  # "arg": False ->  pass
                        pass
            else:  # "": "use_pretrained" ->  --use_pretrained
                command.append(f"{v}")

        self.command_str = " ".join(command)
        self.command_str = f'sbatch {slurm_pre} --wrap "{self.command_str}"'

        print(self.command_str)

        if not no_output_dir and os.path.exists(os.path.join(self.output_dir, "done")):
            self.state = Job.DONE
        elif not no_output_dir and os.path.exists(self.output_dir):
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

    def __str__(self):
        job_info = {
            i: self.train_args[i]
            for i in self.train_args
            if i not in ["experiment", "output_dir"]
        }
        return "{}: {} {}".format(
            self.state, self.output_dir if not self.no_output_dir else "", job_info
        )

    @staticmethod
    def launch(jobs, launcher_fn, *args, **kwargs):
        print("Launching...")
        jobs = jobs.copy()
        print("Making job directories:")
        for job in tqdm.tqdm(jobs, leave=False):
            if not job.no_output_dir:
                os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands, *args, **kwargs)
        print(f"Launched {len(jobs)} jobs!")

    @staticmethod
    def delete(jobs):
        print("Deleting...")
        for job in jobs:
            if not job.no_output_dir:
                shutil.rmtree(job.output_dir)
        print(f"Deleted {len(jobs)} jobs!")


def ask_for_confirmation():
    response = input("Are you sure? (y/n) ")
    if not response.lower().strip()[:1] == "y":
        exit(0)


def make_args_list(experiment):
    return experiments.get_hparams(experiment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a sweep")
    parser.add_argument(
        "command", choices=["launch", "delete_incomplete", "delete_complete"]
    )
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument("--skip_confirmation", action="store_true")
    parser.add_argument("--slurm_pre", type=str, required=True)
    parser.add_argument("--command_launcher", type=str, required=True)
    parser.add_argument("--no_output_dir", action="store_true")
    parser.add_argument("--max_slurm_jobs", type=int, default=400)
    args = parser.parse_args()

    args_list = make_args_list(args.experiment)
    jobs = [
        Job(
            train_args,
            args.output_dir,
            args.slurm_pre,
            experiments.get_script_name(args.experiment),
            args.no_output_dir,
        )
        for train_args in args_list
    ]

    for job in jobs:
        print(job)
    print(
        "{} jobs: {} done, {} incomplete, {} not launched.".format(
            len(jobs),
            len([j for j in jobs if j.state == Job.DONE]),
            len([j for j in jobs if j.state == Job.INCOMPLETE]),
            len([j for j in jobs if j.state == Job.NOT_LAUNCHED]),
        )
    )

    if args.command == "launch":
        to_launch = [j for j in jobs if j.state in [Job.NOT_LAUNCHED, job.INCOMPLETE]]
        print(f"About to launch {len(to_launch)} jobs.")
        if not args.skip_confirmation:
            ask_for_confirmation()
        launcher_fn = launchers.REGISTRY[args.command_launcher]
        Job.launch(to_launch, launcher_fn, max_slurm_jobs=args.max_slurm_jobs)

    elif args.command == "delete_incomplete":
        to_delete = [j for j in jobs if j.state == Job.INCOMPLETE]
        print(f"About to delete {len(to_delete)} jobs.")
        if not args.skip_confirmation:
            ask_for_confirmation()
        Job.delete(to_delete)

    elif args.command == "delete_complete":
        to_delete = [j for j in jobs]
        print(f"About to delete {len(to_delete)} jobs.")
        if not args.skip_confirmation:
            ask_for_confirmation()
        Job.delete(to_delete)
