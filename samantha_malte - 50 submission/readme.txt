# Overview
- data.py allows to import the full imputed dataset or different subsets (based on feature selection)
- feature_selection.py contains performs feature selection
- impute.R is our imputation script
- inspect_features.py prints useful information about the feature masks created by feature_selection.py
- inspect_most_important.py allows to analyze the most important features as determined by a random forest
- predictions.py runs all classifiers and regressors on the different datasets
- predictor.py bundles functionality common to all models

# Setup
Use the requirements.txt to install all required packages with pip.
