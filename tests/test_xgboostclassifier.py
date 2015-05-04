import numpy as np
import XGBoostClassifier
from sklearn.metrics import log_loss
from sklearn.datasets import make_classification

def test_xgboost_classifier():
    X, y = make_classification(random_state=1337)

    xgb = XGBoostClassifier.XGBoostClassifier(num_classes=2, n_iter=10)
    xgb.fit(X, y)
    np.testing.assert_almost_equal(log_loss(y, xgb.predict_proba(X)), 0.12696089, decimal = 6) 
