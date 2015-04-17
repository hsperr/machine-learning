import sys
if not 'xgboost' in sys.path:
    sys.path.append('~/Applications/xgboost/wrapper/')

import xgboost as xgb
import numpy as np

class XGBoostClassifier():

    def __init__(self,
                 learning_rate=0.25,
                 max_trees=10,
                 max_samples=1.0,
                 param=None,
                 watchlist=None,
                 n_iter=150):
        self.booster = None
        if param is None:
            param ={
          'silent':1,
          'use_buffer': True,
          'num_round': 250,
          'ntree_limit': 0,
          'nthread': 4,
          'booster': 'gbtree',
          'eta': learning_rate,
          'gamma': 0.02,
          'max_depth': max_trees,
          'min_child_weight': 4,
          'min_loss_reduction':1,
          'row_subsample' : 0.9,
          'subsample': max_samples,
          'colsample_bytree': 1.0,
          'l': 0,
          'alpha': 0,
          'lambda':0.0,
          'lambda_bias': 0,
          'objective': 'multi:softprob',
          'eval_metric': 'mlogloss',
          'seed': 1,
          'num_class': 9
        }
        self.param = param
        if not watchlist:
            self.wl=[]
        else:
            self.wl = watchlist
        self.n_iter=n_iter

    def fit(self, X, y=None):
        X=self.convert(X, y)
        print X
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
