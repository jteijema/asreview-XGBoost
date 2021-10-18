from asreview.models.classifiers.base import BaseTrainClassifier

import xgboost as xgb

from sklearn.model_selection import GridSearchCV

class XGBoost(BaseTrainClassifier):
    """XGBoost classifier

    This classifier is based on the XGBClassifier.

    """

    name = "xgboost"

    def __init__(self,
                 use_GridSearchCV = True,):           #Error messaging

        super().__init__()
        self.use_GridSearchCV = use_GridSearchCV

    def fit(self, X, y):
        """Fit the model to the data.

        Arguments
        ---------
        X: numpy.ndarray
            Feature matrix to fit.
        y: numpy.ndarray
            Labels for supervised learning.
        """
        if y.size > 6 and self.use_GridSearchCV:
            self._model = self._tune(X, y)
        else:
            self._model = self._createClassifier()

        return self._model.fit(X, y)

    def _createClassifier(self):
        return xgb.XGBClassifier(
            eval_metric = "logloss", 
            use_label_encoder=False, 
            verbosity=1)

            # base_score=0.5, booster='gbtree', colsample_bylevel=1,
            # colsample_bynode=1, colsample_bytree=1, gamma=0,
            # importance_type='gain', interaction_constraints='',
            # learning_rate=0.300000012, max_delta_step=0, max_depth=6,
            # min_child_weight=1, missing=nan, monotone_constraints='()',
            # n_estimators=n_estimators, n_jobs=4, num_parallel_tree=1,
            # random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
            # subsample=1, tree_method='exact', validate_parameters=1

    def _tune(self, X, y):

        clf = self._createClassifier()
        parameters = {
                    "n_estimators" : [10, 100, 1000],
                    "max_depth": [3, 4, 5, 7],
                    "learning_rate": [0.1, 0.01, 0.05],
                    "gamma": [0, 0.25, 1],
                    "reg_lambda": [0, 1, 10],
                    "scale_pos_weight": [1, 3, 5],
                    "subsample": [0.8],
                    "colsample_bytree": [0.5],
            }

        grid = GridSearchCV(clf,
                            parameters, n_jobs=-1,
                            scoring="neg_log_loss",
                            cv=2)

        grid.fit(X, y)

        print(grid.best_params_)

        return grid