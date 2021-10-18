from asreview.models.classifiers.base import BaseTrainClassifier

import xgboost as xgb

class XGBoost(BaseTrainClassifier):
    """XGBoost classifier

    """

    name = "xgboost"

    def __init__(self,
                 objective='multi:softmax',
                 num_class = 2,
                 colsample_bytree = 0.3, 
                 learning_rate = 0.1,
                 max_depth = 5, 
                 alpha = 10, 
                 n_estimators = 10):

        super().__init__()

        self.objective=objective
        self.num_class = num_class
        self.colsample_bytree = colsample_bytree
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.alpha = alpha
        self.n_estimators = n_estimators
                
        self.xg_reg = xgb.XGBRegressor(
            objective = objective, 
            num_class = num_class, 
            colsample_bytree = colsample_bytree, 
            learning_rate = learning_rate,
            max_depth = max_depth, 
            alpha = alpha, 
            n_estimators = n_estimators
            )

    def fit(self, X, y):
        """Fit the model to the data.

        Arguments
        ---------
        X: numpy.ndarray
            Feature matrix to fit.
        y: numpy.ndarray
            Labels for supervised learning.
        """

        self.dmatrix = xgb.DMatrix(data=X,label=y)

        return self.xg_reg.fit(X, y)

    def predict_proba(self, X):
        """Get the inclusion probability for each sample.

        Arguments
        ---------
        X: numpy.ndarray
            Feature matrix to predict.

        Returns
        -------
        numpy.ndarray
            Array with the probabilities for each class, with two
            columns (class 0, and class 1) and the number of samples rows.
        """

        return self.xg_reg.predict(X)