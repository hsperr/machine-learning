# Machine Learning Helpers

This repository contains several helper classes for ML, it tries to maintain rudimentary scikit learn compatibility

## Helpers

### FunctionTransformer / LasagneUtils

- applies a function to all elements

Example creates a simple network, with linear decrease in learning rate and 
linear increase in momentum, it will save the best iterations after the first 10 and
stop if either there was no improvement in the last 10 iterations or the train/test ratio is below 0.8:


```Python
from lasagne.layers import DenseLayer, InputLayer, DropoutLayer
from lasagne.nonlinearities import rectify, softmax, tanh, linear
from lasagne.updates import nesterov_momentum, rmsprop, momentum
from nolearn.lasagne import NeuralNet
import theano

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from FunctionTransformer import LogTransformer
from LasagneUtils import EarlyStopper, LinearAdjustVariable, TrainRatioStopper, BestIteratorSaver



layers=[('input', InputLayer),
        ('dense0', DenseLayer),
        ('output', DenseLayer),
        ]
#DenseLayer()
net =  NeuralNet(layers=layers,
                 input_shape=(None, train_x.shape[1]),
                 dense0_num_units=512,
                 dense0_nonlinearity=rectify,
                 output_num_units=9,
                 output_nonlinearity=softmax,
                 update=momentum,
                 update_learning_rate=theano.shared(np.float32(0.05)),
                 update_momentum=theano.shared(np.float32(0.9)),
                 on_epoch_finished = [LinearAdjustVariable('update_learning_rate', start=0.05, stop=0.0001),
                                      LinearAdjustVariable('update_momentum', start=0.9, stop=0.999),
                                      TrainRatioStopper(0.8),
                                      EarlyStopper(),
                                      BestIterationSaver(verbose=1)
                                      ],
                 eval_size=0.1,
                 verbose=1,
                 max_epochs=501)

net_ppl2 = Pipeline([('LogTrans', LogTransformer()),('StandartScale', StandardScaler()), ('nn',net)])
net_ppl2.fit(train_x.astype(np.float32), train_y.astype(np.int32))
```

### XGBoostClassifier

- wrapper for xgboost. needs xgboost installed and have <xgboostmaindir>/wrapper in PYTHON_PATH

Example:

```Python
from XGBoostClassifier import XGBoostClassifier
xgb = XGBoostClassifier(watchlist=[(test_x, test_y)],
                        max_samples=0.9,
                        n_iter=105,
                        random_state=1335)

xgb.fit(train_x, train_y)
xgb.predict_proba(train_x)
```

### EnsambleClassifiers

- pass a list of classifiers and find the best weights for combining them

Example:

```Python
from EnsembleClassifier import BestEnsambleWeights

rfc = RandomForestClassifier(...)
xgb = XGBoostClassifier(...)
logreg = LogisticRegression(...)

bew = BestEnsembleWeights([rfc, xgb, logreg], prefit=False, random_state=1337, verbose=1)
bew.fit(train_y, train_x)
bew.predict_proba(test_x)

```
