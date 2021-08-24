**Code to accompany,
    "A comparison of approaches to improve worst-case predictive model performance over patient subpopulations", by
    Stephen R. Pfohl, Haoran Zhang, Yizhe Xu, Agata Foryciarz, Marzyeh Ghassemi, Nigam H. Shah**


## Installation
1. Pull this repository, and from within the project directory call `pip install .` or `pip install -e .` if you intend to modify the code


## Package structure
* The package is structured in three submodules. Further information on each submodule is provided README files within the respective directories. 
* `group_robustness_fairness.omop`
    * Contains code to reproduce experiments conducted on STARR. Also contains code to generate visualization and tables for the `mimic_eicu` module
* `group_robustness_fairness.mimic_eicu`
    * Contains code to reproduce experiments conducted on MIMIC-III and the eICU database.
* `group_robustness_fairness.prediction_utils`
    * Library code with pytorch models and evaluators