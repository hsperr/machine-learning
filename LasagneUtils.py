from __future__ import print_function
import numpy as np
import abc


class LinearAdjustVariable(object):

    """
        Adjusts a variable after each epoch, e.g. learning_rate or momentum

        name : name of the variable to update
        start : start value for update
        stop : stop value for update
    """

    def __init__(self, name, start=0.1, stop=0.001):
        self.name = name
        self.start = start
        self.stop = stop

        # TODO: make custom stepfuction, exponential etc.
        self.step_function = None

    def __call__(self, nn, train_history):
        epoch = train_history[-1]['epoch']

        new_value = self.start
        stepsize = 0.4 * (self.start - self.stop) / nn.max_epochs
        if self.start < 0.1:
            new_value -= stepsize * epoch  # np.float32(self.ls[epoch-1])
            new_value = max(new_value, self.stop)
        else:
            new_value += stepsize * epoch
            new_value = min(new_value, self.stop)

        new_value = np.float32(new_value)
        getattr(nn, self.name).set_value(new_value)


class EarlyStopper(object):

    """
        Stops learning if there was no improvement for N iterations

        max_iterations : number of iterations to have no improvement in a row
                         before stopping
    """

    def __init__(self, max_iterations=10):
        self.max_iterations = max_iterations
        self.best_iteration_score = 10000
        self.best_iteration = 0

    def __call__(self, nn, train_history):
        if self.best_iteration_score > train_history[-1]['valid_loss']:
            self.best_iteration_score = train_history[-1]['valid_loss']
            self.best_iteration = len(train_history)

        if len(train_history) - self.best_iteration >= self.max_iterations:
            nn.max_epochs = train_history[-1]['epoch']


class TrainRatioStopper(object):

    """
        Stops learning if train_loss/validation_loss falls below a certain ratio

        stop_ratio : the train_loss/validation_loss ratio to stop training
    """

    def __init__(self, stop_ratio=0.8):
        self.stop_ratio = stop_ratio

    def __call__(self, nn, train_history):
        ratio = train_history[-1]['train_loss'] / \
            train_history[-1]['valid_loss']
        if ratio < self.stop_ratio:
            nn.max_epochs = train_history[-1]['epoch']


class BestIterationSaver(object):

    """
        Saves the weights for the best iteration

        name : name of the best iteration weights file
        delayed_start : number of iterations to wait before starting to save
        verbose : print a logmessage when saving

    """

    def __init__(self, name='best_iteration.weights', delayed_start=10, verbose=0):
        self.best_score = None
        self.best_weights = None
        self.delayed_start = delayed_start
        self.filename = name
        self.verbose=verbose

    def __call__(self, nn, train_history):
        if len(train_history) < self.delayed_start:
            return

        if self.best_score is None or train_history[-1]['valid_loss'] < self.best_score:
            if self.verbose:
                print('Saving to {filename}'.format(filename=self.filename))

            self.best_score = train_history[-1]['valid_loss']
            nn.save_weights_to(self.filename)
