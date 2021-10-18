from asreview.models.classifiers.base import BaseTrainClassifier

import xgboost as xgb

class XGBoost(BaseTrainClassifier):
    """XGBoost classifier

    This classifier is based on the XGBClassifier.

    """

    name = "xgboost"

    def __init__(self,
                 eval_metric = "logloss",):

        super().__init__()

        self.eval_metric = eval_metric

        self._model = xgb.XGBClassifier(eval_metric = eval_metric, use_label_encoder=False)

