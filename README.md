# CVST-SII Study: Reproducible Analysis Repository

## Overview

This repository contains the analytic data and Python code used for the statistical analyses in the revised manuscript examining the association between the systemic immune-inflammation index (SII) and consciousness disturbance in patients with cerebral venous sinus thrombosis (CVST).

The repository is provided to facilitate reproducibility, data traceability, and independent review of the analytical workflow.

## Repository Contents

The repository includes:

- `data.csv` – de-identified analytic dataset
- `data_standardized.csv` – standardized dataset used for reproducibility and modelling-related analyses
- `data_dictionary.csv` – variable descriptions and coding information
- `cvst_complete_analysis.py` – complete analysis script
- `CVST_complete_run.ipynb` – Jupyter Notebook for reproducing the analysis workflow
- `requirements.txt` – required Python packages
- `README_CN.md` – supplementary Chinese-language instructions

## Data Preparation

The analytic dataset was prepared from the study database after data cleaning and de-identification.

Derived inflammatory indices, including SII, were generated during data preprocessing from the available hematologic measurements according to the definitions used in the study. The derived SII variable retained in the analytic dataset was used directly in the subsequent analyses.

Where appropriate, continuous variables were standardized for modelling and reproducibility purposes. A standardized version of the analytic dataset is provided separately.

## Analytical Framework

The prespecified adjusted analysis of the association between SII and consciousness disturbance is regarded as the primary analysis.

Variable-selection procedures, parsimonious multivariable modelling, machine-learning comparisons, and model-interpretation analyses are considered exploratory and are intended to complement the primary association analysis.

The exploratory modelling results should therefore not be interpreted as establishing causal relationships, biological mechanisms, or externally validated prognostic performance.

## Internal Performance Evaluation

Repeated stratified cross-validation was used to evaluate model performance for a fixed set of selected features.

Because feature selection was performed using the full analytic cohort before this evaluation, the repeated cross-validation should be interpreted as performance assessment conditional on the selected feature set rather than validation of the complete feature-selection and modelling pipeline.

Calibration and additional sensitivity analyses are also included in the analysis workflow.

## Reproducibility

All reported analyses can be reproduced from the files contained in this repository.

To install the required Python packages:

```bash
pip install -r requirements.txt
```

The analysis can then be run either through the Jupyter Notebook:

```text
CVST_complete_run.ipynb
```

or directly using:

```bash
python cvst_complete_analysis.py
```

The scripts generate the corresponding analytical tables, figures, and supporting output files.

## Data and Code Availability

The de-identified analytic dataset, standardized dataset, variable documentation, and complete analysis code are provided in this repository for reproducibility and editorial review.

The shared files contain the variables required to reproduce the reported analyses and do not include direct patient identifiers.

Use of the data should remain consistent with the ethical and institutional requirements described in the manuscript.

## Interpretation

This study is observational. The available data do not establish a clear temporal or causal relationship between the measured predictors and consciousness disturbance.

Accordingly, the results are presented as associations and exploratory predictive patterns within the study cohort and should not be interpreted as evidence of incident neurological deterioration, biological mechanism, or externally validated clinical prognosis.

## Contact

Questions regarding the analytical workflow or reproducibility of the reported results may be directed to the corresponding author.
