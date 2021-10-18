from asreview.models.classifiers.base import BaseTrainClassifier

import xgboost as xgb

import numpy as np

class XGBoost(BaseTrainClassifier):
    """XGBoost classifier

    """

    name = "xgboost"

    def __init__(self,
                 objective='multi:softmax',
                 num_class = 2,
                 eval_metric = "logloss",
                 colsample_bytree = 0.3, 
                 learning_rate = 0.1,
                 max_depth = 5, 
                 alpha = 10, 
                 n_estimators = 10):

        super().__init__()

        self.objective=objective
        self.num_class = num_class
        self.eval_metric = eval_metric
        self.colsample_bytree = colsample_bytree
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.alpha = alpha
        self.n_estimators = n_estimators
                
        self._model = xgb.XGBClassifier(eval_metric = eval_metric)

