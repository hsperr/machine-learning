import numpy as np

from sklearn.base import TransformerMixin

class FunctionTransformer(TransformerMixin):
    def __init__(self, func):
        self.func=func

    def fit(self, X, y):
        return self

    def transform(self, X):
        return self.func(X.astype(np.float32))

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

class LogTransformer(FunctionTransformer):
    def __init__(self):
        super(LogTransformer, self).__init__(np.log1p)
