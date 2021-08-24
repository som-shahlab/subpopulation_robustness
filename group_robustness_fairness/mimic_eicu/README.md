## Step 1: Data Access and Preprocessing

### eICU

1. [Obtain access](https://eicu-crd.mit.edu/gettingstarted/access/) to the eICU Collaborative Research Database on PhysioNet and download the [dataset](https://physionet.org/content/eicu-crd/2.0/).

2. Clone the [eICU Benchmarks](https://github.com/mostafaalishahi/eICU_Benchmark) repository and follow the instructions under the "Data extraction" section.

3. Update the `eicu_dir` and `benchmark_dir` variables in `extraction_scripts/eicu_benchmarks/process_data.py` to point to the raw data and processed data folders.

### MIMIC
1. [Obtain access](https://mimic.physionet.org/gettingstarted/access/) to the MIMIC-III Database on PhysioNet and download the [dataset](https://physionet.org/content/mimiciii/1.4/).

2. Clone the [MIMIC Benchmarks](https://github.com/YerevaNN/mimic3-benchmarks) repository and follow the instructions under the "Building a benchmark" section.

3. Update the `mimic_benchmark_dir` variable in `extraction_scripts/mimic_benchmarks/process_data.py` to point to the processed data folder.


## Step 2: Cohort Creation
In each of the files below, update the `output_dir` variable and run the file from the folder that the file is located. In the final file, you will also have to update the MIMIC-III database login details.
```
* extraction_scripts/mimic_benchmarks/process_data.py
* extraction_scripts/eicu_benchmarks/process_data.py
```


## Step 3: Training Models

To train a single model, use `train_model.py`, passing in the appropriate arguments.

To run a grid of experiments varying hyperparameters and data splits, use `sweep.py`. In the paper, we use the experiments defined in `experiments.py`. For example:

```
python sweep.py launch \
    --experiment MIMICMortalityERM \
    --output_dir /my/sweep/output/path \
    --command_launcher "local" 
```

This command can also be ran easily using `scripts/run_exp.sh`. You will likely need to update the launcher to fit your compute environment.

After running the ERM experiments and prior to running the DRO experiments, you should run `notebooks/get_best_model_erm_fold_1_5.ipynb` in order to output optimal model hyperparameters. As such, the experiments should be ran roughly in the following order:

```
* MIMICMortalityERM
* eICUMortalityERM
* notebooks/get_best_model_erm_fold_1_5.ipynb
* MIMICMortalityDRO
* eICUMortalityDRO
```

## Step 4: Result Analysis

1. Run `notebooks/get_best_model_fold_1_5_scratch.ipynb`

2. Run the `Bootstrap` experiment in `experiments.py`, with `scripts/run_bootstrap.sh`

3. Use the `make_plots.ipynb` notebooks in the `omop` module, after moving the results to the appropriate subdirectory.
