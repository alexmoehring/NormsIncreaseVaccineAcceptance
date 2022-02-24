# Surfacing norms to increase vaccine acceptance
This repository contains replication materials for Moehring et al (2021).

## Citation
Moehring, Alex, Avinash Collis, Kiran Garimella, M. Amin Rahimian, Sinan Aral, and Dean Eckles. "Surfacing norms to increase vaccine acceptance." Available at SSRN 3782082 (2021).

[![DOI](https://zenodo.org/badge/437312905.svg)](https://zenodo.org/badge/latestdoi/437312905)

## Data availability
Documentation of the COVID-19 Beliefs, Behaviors, and Norms survey instrument and aggregated data from the survey are publicly
available at https://covidsurvey.mit.edu. To be able to replicate the results, you must 
have access to the survey microdata. Researchers can request access to the 
microdata from Facebook and MIT at https://dataforgood.fb.com/docs/preventive-health-survey-request-for-data-access/. 

We have provided a demo dataset "random_demo_data.txt.gz" that matches the schema of the actual data but contains random entries. The code runs on this dataset in under 20 minutes on a typical machine.

## Replication process
To replicate the results from the paper, you must complete the following.
1) Update the <code>path</code> variable in 1_make_numeric_dataset.py to point to the 
survey microdata and then run this script.
2) Run 2_analyze_experiment.py to replicate the figures from the paper.
3) Run 3_multilevel_model.R to estimate the country-level model. We bootstrap standard errors, so it can a very long time without a server with many cores.
4) Run 4_getVaccineAnalysisData.py and then 5_experiment_baseline.Rmd. These scripts generate Figures 1 and 2 in the manuscript.

Running these scripts will produce a directory norms_experiment/current_date and populate it with
the figures and tables used in the paper.

## Requirements
The code has only been tested on machines with at least 32GiB of RAM. The code was run in the following python 3.8 environment
with the following packages. On the full dataset, the analysis code takes roughly an hour. On the demo data, the code runs 
in under 10 minutes.

- DateTime                           4.3
- json5                              0.9.6
- matplotlib                         3.3.4
- numpy                              1.21.2
- pandas                             1.3.0
- patsy                              0.5.1
- scipy                              1.6.2
- stargazer                          0.0.5
- statsmodels                        0.12.2
- tqdm                               4.61.2

The multilevel modeling script was run using R version 3.5.1 and the experiment_baseline script was run using R 4.0.2. 
