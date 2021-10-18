# XGBoost implementation for ASReview
This repository contains an extension for ASReview based on [xgboost](https://github.com/dmlc/xgboost).

The hyperparameters are not yet optimised.

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

Note that `TF-IDF` works best as FE.

## To-Do
CV the model every iteration.

## WHy use XGBoost?
From https://www.kdnuggets.com/2017/10/xgboost-concise-technical-overview.html
 
XGBoost is scalable in distributed as well as memory-limited settings. This scalability is due to several algorithmic optimizations.

1. Split finding algorithms: approximate algorithm:

To find the best split over a continuous feature, data needs to be sorted and fit entirely into memory. This may be a problem in case of large datasets.

An approximate algorithm is used for this. Candidate split points are proposed based on the percentiles of feature distribution. The continuous features are binned into buckets that are split based on the candidate split points. The best solution for candidate split points is chosen from the aggregated statistics on the buckets.

2. Column block for parallel learning:

Sorting the data is the most time-consuming aspect of tree learning. To reduce sorting costs, data is stored in in-memory units called ‘blocks’. Each block has data columns sorted by the corresponding feature value. This computation needs to be done only once before training and can be reused later.

Sorting of blocks can be done independently and can be divided between parallel threads of the CPU. The split finding can be parallelized as the collection of statistics for each column is done in parallel.

3. Weighted quantile sketch for approximate tree learning:

To propose candidate split points among weighted datasets, the Weighted Quantile Sketch algorithm is used. It carries out merge and prune operations on quantile summaries over the data.

4. Sparsity-aware algorithm:

Input may be sparse due to reasons such as one-hot encoding, missing values and zero entries. XGBoost is aware of the sparsity pattern in the data and visits only the default direction (non-missing entries) in each node.

5. Cache-aware access:

To prevent cache miss during split finding and ensure parallelization, choose 2^16 examples per block.

6. Out-of-core computation:

For data that does not fit into main memory, divide the data into multiple blocks, and store each block on the disk. Compress each block by columns and decompress on the fly by an independent thread while disk reading.

7. Regularized Learning Objective:

To measure the performance of a model given a certain set of parameters, we need to define an objective function. An objective function must always contain two parts: training loss and regularization. The regularization term penalizes the complexity of the model.
