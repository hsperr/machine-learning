"""
    A wrapper around XGBoost trying to maintain the Scikit Learn API

    Authors: Henning Sperr

    License: BSD-3 clause
"""

import random
import xgboost as xgb
import numpy as np

class XGBoostClassifier():
    """
        A simple wrapper around XGBoost

        more details:
        https://github.com/dmlc/xgboost/wiki/Parameters

        Parameters
        ----------

        base_estimator : can be 'gbtree' or 'gblinear'
        gamma : minimum loss reduction required to make a partition, higher values
                mean more conservative boosting
        max_depth : maximum depth of a tree
        min_child_weight : larger values mean more conservative partitioning

        objective : 'reg:linear' - linear regression
                    'reg:logistic' - logistic regression
                    'binary:logistic' - binary logistic regression
                    'binary:logitraw' - binary logistic regression before logistic transformation
                    'multi:softmax' - multiclass classification
                    'multi:softprob' - multiclass classification with class probability output
                    'rank:pairwise' - pairwise minimize loss

        metric : 'rmse' - root mean square error
                 'logloss' - negative log likelihood
                 'error' - binary classification error rate
                 'merror' - multiclass error rate
                 'mlogloss' - multiclass logloss
                 'auc' - area under the curve for ranking evaluation
                 'ndcg' - normalized discounted cumulative gain ndcg@n for top n eval
                 'map' - mean average precision map@n for top n eval
    """
    def __init__(self,
                 base_estimator='gbtree',
                 objective='multi:softprob',
                 metric='mlogloss',
                 num_classes=9,
                 learning_rate=0.25,
                 max_depth=10,
                 max_samples=1.0,
                 max_features=1.0,
                 max_delta_step=0,
                 min_child_weight=4,
                 min_loss_reduction=1,
                 l1_weight=0.0,
                 l2_weight=0.0,
                 l2_on_bias=False,
                 gamma=0.02,
                 inital_bias=0.5,
                 random_state=None,
                 watchlist=None,
                 n_jobs=4,
                 n_iter=150):

        self.booster = None

        if random_state is None:
            random_state = random.randint(0, 1000000)

        param ={
          'silent':1,
          'use_buffer': True,
          'base_score': inital_bias,
          'nthread': n_jobs,
          'booster': base_estimator,
          'eta': learning_rate,
          'gamma': gamma,
          'max_depth': max_depth,
          'max_delta_step' : max_delta_step,
          'min_child_weight': min_child_weight,
          'min_loss_reduction':min_loss_reduction,
          'subsample': max_samples,
          'colsample_bytree': max_features,
          'alpha': l1_weight,
          'lambda':l2_weight,
          'lambda_bias': l2_on_bias,
          'objective': objective,
          'eval_metric': metric,
          'seed': random_state,
          'num_class': num_classes
        }
        self.param = param
        if not watchlist:
            self.wl=[]
        else:
            self.wl = watchlist
        self.n_iter=n_iter

    def fit(self, X, y=None):
        X=self.convert(X, y)
        if self.wl:
            wl = [(X, 'train')]
            for i, ent in enumerate(self.wl):
                ent, lbl = ent
                wl.append((self.convert(ent, lbl), 'test-'+str(i)))
            self.booster = xgb.train(self.param, X, self.n_iter, wl)
        else:
            self.booster = xgb.train(self.param, X, self.n_iter, [(X,'train')])


    def predict_proba(self, X):
        X = xgb.DMatrix(X)
        return self.booster.predict(X)


    def convert(self, X, y=None):
        if y is None:
            if isinstance(X, xgb.DMatrix):
                return X
            if hasattr(X,'values'):
                X = xgb.DMatrix(X.values)
                return X
            return xgb.DMatrix(X)
        else:
            if hasattr(X,'values'):
                X = xgb.DMatrix(X.values, y.values, missing=np.nan)
                return X
            return xgb.DMatrix(X, y, missing=np.nan)

    def predict(self, X):
        X = self.convert(X)
        probs = self.booster.predict(X)
        return np.argmax(probs, axis=1)
