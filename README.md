# XGBoost implementation for ASReview
This repository contains an extension for ASReview based on [xgboost](https://github.com/dmlc/xgboost)


Commands used for performance testing:
`asreview simulate benchmark:van_de_Schoot_2017 -m xgboost --n_queries min -s xgboost.h5 --seed 1 --n_prior_included 5 --n_instances 10`

`asreview simulate benchmark:van_de_Schoot_2017 -m rf --n_queries min -s nb.h5 --seed 1 --n_prior_included 5 --n_instances 10`

`asreview simulate benchmark:van_de_Schoot_2017 -m nb --n_queries min -s nb.h5 --seed 1 --n_prior_included 5 --n_instances 10`

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