from itertools import combinations

from sklearn.base import ClassifierMixin
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import log_loss

import numpy as np

class LinearModelCombination(ClassifierMixin):
    def __init__(self, model1, model2, weight=None, metric=log_loss):
        self.model1 = model1
        self.model2 = model2
        self.weight = weight
        self.metric = metric

    def fit(self, X, y):
        scores = []
        pred1 = self.model1.predict_proba(X)
        pred2 = self.model2.predict_proba(X)

        for i in xrange(0, 101):
            weight=i/100.
            scores.append(self.metric(y, weight*pred1+(1-weight)*pred2))
            #linear surface so if the score gets worse we can stop
            if len(scores)>1 and scores[-1]>scores[-2]:
                break

        best_weight = np.argmin(scores)

        self.best_score = scores[best_weight]
        self.weight=best_weight/100.

        return self

    def predict(self, X):
        if self.weight == None:
            raise Exception("Classifier seems to be not yet fitted")

        pred1 = self.model1.predict_proba(X)*self.weight
        pred2 = self.model2.predict_proba(X)*(1-self.weight)
        return np.argmax(pred1+pred2)

    def predict_proba(self, X):
        if self.weight == None:
            raise Exception("Classifier seems to be not yet fitted")

        pred1 = self.model1.predict_proba(X)*self.weight
        pred2 = self.model2.predict_proba(X)*(1-self.weight)
        return pred1+pred2

    def __str__(self):
        return ' '.join(["LM: ", str(self.model1), ' - ', str(self.model2), ' W: ', str(self.weight)])

class LogisticModelCombination(ClassifierMixin):
    """
        Combine multiple models using a Logistic Regression
    """
    def __init__(self, classifiers, cv_folds=1, verbose=0):
        self.classifiers = classifiers
        self.logistic = LogisticRegressionCV(Cs=[10, 1, 0.1, 0.01, 0.001])

    def fit(self, X, y):
        sss= StratifiedShuffleSplit(y, n_iter=self.cv_folds)
        for train_index, test_index in sss:
            train_x = X[train_index]
            train_y = y[train_index]

            test_x = X[test_index]
            test_y = y[test_index]

            self._fit_logistic(train_x, train_y)

    def _fit_logitstic(self, X, y):
        preds = []
        for clf in self.classifiers:
            preds.append(clf.predict(X))

        pred_X = np.concatenate(preds, axis=1)
        print pred_X.shape

        self.logistic.fit(pred_X,y)
        print self.logistic._coef



class EnsambleClassifier(ClassifierMixin):
    '''
        Instanitate with a bunch of classifiers and a metric
        and find the best weights for combining them.
    '''
    def __init__(self, classifiers, metric=log_loss, cv_folds=1, verbose=0):
        self.classifiers = classifiers
        self.names = dict((i, str(clf)[:10]) for i, clf in enumerate(classifiers))
        self.metric = metric
        self.cv_folds = cv_folds
        self.verbose = verbose
        self.best_model = None

    def fit(self, X, y):
        sss= StratifiedShuffleSplit(y, n_iter=self.cv_folds)
        for train_index, test_index in sss:
            train_x = X[train_index]
            train_y = y[train_index]

            test_x = X[test_index]
            test_y = y[test_index]

            self._find_simple_weighting(train_x, train_y)

    def predict(self, x):
        if self.best_model == None:
            raise exception("first call .fit on the classifier")

        return self.best_model.predict(x)

    def predict_proba(self, x):
        if self.best_model == None:
            raise exception("first call .fit on the classifier")

        return self.best_model.predict_proba(x)


    def _find_simple_weighting(self, X, y):
        preds = []

        for i, clf in enumerate(self.classifiers):
            preds.append((i, clf.predict_proba(X)))

        best_score = 1000
        best_models = []

        for clf1, clf2 in combinations(preds, 2):
            name1, pred1 = clf1
            name2, pred2 = clf2

            scores = []
            for i in xrange(0, 101):
                weight=i/100.
                scores.append(self.metric(y, weight*pred1+(1-weight)*pred2))
                if len(scores)>1 and scores[-1]>scores[-2]:
                    break

            c_best_weight = np.argmin(scores)
            c_best_score = scores[c_best_weight]
            c_best_weight = c_best_weight/100.

            print self.names[name1], '/', self.names[name2], c_best_score, c_best_weight
            if best_score>c_best_score:
                best_score = c_best_score
                best_models = (best_score, name1, name2, c_best_weight)

        score, index1, index2, weight =  best_models

        used_models = set()
        used_models.add(index1)
        used_models.add(index2)

        print 'combining', score, index1, index2, weight
        model = LinearModelCombination(self.classifiers[index1], self.classifiers[index2], weight)
        while True:
            self.best_model = model
            best_score = model.predict_proba(X)
            scores = []
            for i, clf in enumerate(self.classifiers):
                if i in used_models:
                    continue

                test_model = LinearModelCombination(model, clf)
                test_model.fit(X, y)
                if test_model.weight < 0.01 or test_model.weight >= 1.0:
                    continue

                score = log_loss(y, test_model.predict_proba(X))
                scores.append([score, test_model, i, test_model.weight])
            print scores

            if not scores:
                break

            scores.sort()
            model = scores[0][1]
            used_models.add(scores[0][2])
        print self.best_model
