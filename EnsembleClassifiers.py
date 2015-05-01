"""
    A wrapper for different ways of combining models

    Authors: Henning Sperr

    License: BSD-3 clause
"""
from __future__ import print_function
from itertools import combinations, izip
import random

from sklearn.base import ClassifierMixin
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

import numpy as np
from scipy.optimize import minimize


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
            weight = i / 100.
            scores.append(
                self.metric(y, weight * pred1 + (1 - weight) * pred2))
            # linear surface so if the score gets worse we can stop
            if len(scores) > 1 and scores[-1] > scores[-2]:
                break

        best_weight = np.argmin(scores)

        self.best_score = scores[best_weight]
        self.weight = best_weight / 100.

        return self

    def predict(self, X):
        if self.weight == None:
            raise Exception("Classifier seems to be not yet fitted")

        pred1 = self.model1.predict_proba(X) * self.weight
        pred2 = self.model2.predict_proba(X) * (1 - self.weight)
        return np.argmax(pred1 + pred2)

    def predict_proba(self, X):
        if self.weight == None:
            raise Exception("Classifier seems to be not yet fitted")

        pred1 = self.model1.predict_proba(X) * self.weight
        pred2 = self.model2.predict_proba(X) * (1 - self.weight)
        return pred1 + pred2

    def __str__(self):
        return ' '.join(["LM: ", str(self.model1), ' - ', str(self.model2), ' W: ', str(self.weight)])


class BestEnsembleWeights(ClassifierMixin):

    """
        Use scipys optimize package to find best weights for classifier combination.

        classifiers : list of classifiers
        prefit : if True classifiers will be assumed to be fit already and the data passed to
                 fit method will be fully used for finding best weights
        random_state : random seed
        verbose : print verbose output

    """

    def __init__(self, classifiers, num_iter=50, prefit=False, random_state=None, verbose=0):
        self.classifiers = classifiers
        self.prefit = prefit
        if random_state is None:
            self.random_state = random.randint(0, 10000)
        else:
            self.random_state = random_state
        self.verbose = verbose
        self.num_iter = num_iter

    def fit(self, X, y):
        if self.prefit:
            test_x, test_y = X, y
        else:
            sss = StratifiedShuffleSplit(
                y, n_iter=1, random_state=self.random_state)
            for train_index, test_index in sss:
                break

            train_x = X[train_index]
            train_y = y[train_index]

            test_x = X[test_index]
            test_y = y[test_index]

            for clf in self.classifiers:
                clf.fit(train_x, train_y)

        self._find_best_weights(test_x, test_y)

    def _find_best_weights(self, X, y):
        predictions = []
        for clf in self.classifiers:
            predictions.append(clf.predict_proba(X))

        def log_loss_func(weights):
            ''' scipy minimize will pass the weights as a numpy array '''
            final_prediction = 0
            for weight, prediction in izip(weights, predictions):
                final_prediction += weight * prediction

            return log_loss(y, final_prediction)

        # the algorithms need a starting value, right not we chose 0.5 for all weights
        # its better to choose many random starting points and run minimize a
        # few times
        starting_values = np.ones(len(predictions)) / (len(predictions))
        # This sets the bounds on the weights, between 0 and 1
        bounds = tuple((0, 1) for w in starting_values)

        # adding constraints  and a different solver as suggested by user 16universes
        # https://kaggle2.blob.core.windows.net/forum-message-attachments/75655/2393/otto%20model%20weights.pdf?sv=2012-02-12&se=2015-05-03T21%3A22%3A17Z&sr=b&sp=r&sig=rkeA7EJC%2BiQ%2FJ%2BcMpcA4lYQLFh6ubNqs2XAkGtFsAv0%3D
        cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})

        res = minimize(log_loss_func, starting_values,
                       method='SLSQP', bounds=bounds, constraints=cons)

        self.best_score = res['fun']
        self.best_weights = res['x']

        for i in xrange(self.num_iter):
            starting_values = np.random.uniform(0,1,size=len(predictions))

            res = minimize(log_loss_func, starting_values,
                           method='SLSQP', bounds=bounds, constraints=cons)
            print('%s' % (res['fun']))

            if res['fun']<self.best_score:
                self.best_score = res['fun']
                self.best_weights = res['x']

                if self.verbose:
                    print('')
                    print('Update Ensamble Score: {best_score}'.format(best_score=res['fun']))
                    print('Update Best Weights: {weights}'.format(weights=self.best_weights))

        if self.verbose:
            print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
            print('Best Weights: {weights}'.format(weights=self.best_weights))

    def predict_proba(self, X):
        prediction = 0
        for weight, clf in izip(self.best_weights, self.classifiers):
            prediction += weight * clf.predict_proba(X)
        return prediction


class LogisticModelCombination(ClassifierMixin):

    """
        Combine multiple models using a Logistic Regression
    """

    def __init__(self, classifiers, cv_folds=1, use_original_features=False, random_state=None, verbose=0):
        self.classifiers = classifiers
        self.cv_folds = cv_folds
        self.use_original_features = use_original_features
        self.logistic = LogisticRegressionCV(
            Cs=[10, 1, 0.1, 0.01, 0.001], refit=True)

        if random_state is None:
            self.random_state = random.randint(0, 10000)
        else:
            self.random_state = random_state

    def fit(self, X, y):
        sss = StratifiedShuffleSplit(
            y, n_iter=self.cv_folds, random_state=self.random_state)
        for train_index, test_index in sss:
            train_x = X[train_index]
            train_y = y[train_index]

            test_x = X[test_index]
            test_y = y[test_index]

            self._fit_logistic(train_x, train_y)

    def _fit_logistic(self, X, y):
        pred_X = self.convert_data(X)
        self.logistic.fit(pred_X, y)
        return self

    def convert_data(self, X):
        preds = []
        for i, clf in enumerate(self.classifiers):
            class_proba = clf.predict(X)
            preds.append(class_proba)
        pred_X = np.vstack(preds).T

        if self.use_original_features:
            pred_X = np.concatenate([X, pred_X], axis=1)
        return pred_X

    def predict_proba(self, X):
        pred_X = self.convert_data(X)
        return self.logistic.predict_proba(pred_X)


class EnsambleClassifier(ClassifierMixin):

    '''
        Instanitate with a bunch of classifiers and a metric
        and find the best weights for combining them.
    '''

    def __init__(self, classifiers, metric=log_loss, cv_folds=1, verbose=0):
        self.classifiers = classifiers
        self.names = dict((i, str(clf)[:10])
                          for i, clf in enumerate(classifiers))
        self.metric = metric
        self.cv_folds = cv_folds
        self.verbose = verbose
        self.best_model = None

    def fit(self, X, y):
        sss = StratifiedShuffleSplit(y, n_iter=self.cv_folds)
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
                weight = i / 100.
                scores.append(
                    self.metric(y, weight * pred1 + (1 - weight) * pred2))
                if len(scores) > 1 and scores[-1] > scores[-2]:
                    break

            c_best_weight = np.argmin(scores)
            c_best_score = scores[c_best_weight]
            c_best_weight = c_best_weight / 100.

            print('%s/%s %s %s' %
                  (self.names[name1], self.names[name2], c_best_score, c_best_weight))
            if best_score > c_best_score:
                best_score = c_best_score
                best_models = (best_score, name1, name2, c_best_weight)

        score, index1, index2, weight = best_models

        used_models = set()
        used_models.add(index1)
        used_models.add(index2)

        print('combining %s %s %s %s' % (score, index1, index2, weight))
        model = LinearModelCombination(
            self.classifiers[index1], self.classifiers[index2], weight)
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
            print('%s' % (scores))

            if not scores:
                break

            scores.sort()
            model = scores[0][1]
            used_models.add(scores[0][2])
