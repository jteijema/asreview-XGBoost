# XGBoost implementation for ASReview
This repository contains an extension for ASReview based on [xgboost](https://github.com/dmlc/xgboost). The hyperparameters are not yet optimised.

## Installation
Install the new classifier with:

```bash
pip install .
```

or

```bash
python -m pip install git+https://github.com/JTeijema/asreview-XGBoost.git
```

## Usage
The new feature extractor can be used with `-m xgboost`:

```bash
asreview simulate benchmark:van_de_Schoot_2017 -m xgboost
```