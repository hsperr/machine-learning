import numpy as np

class AdjustVariable(object):
    """
        Adjusts a variable after each epoch, e.g. learning_rate or momentum
    """
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start=start
        self.stop=stop

        self.ls=None

    def __call__(self, nn, train_history):

        epoch =  train_history[-1]['epoch']
        new_value = self.start
        stepsize = 0.4*(self.start-self.stop)/nn.max_epochs
        if self.start<0.1:
            new_value-=stepsize*epoch# np.float32(self.ls[epoch-1])
            new_value=max(new_value, self.stop)
        else:
            new_value+=stepsize*epoch
            new_value=min(new_value, self.stop)

        new_value=np.float32(new_value)
        getattr(nn, self.name).set_value(new_value)

class OverFitProtector(object):
    """
        Stops learning if train_loss/validation_loss falls below a certain ratio
    """
    def __init__(self, stop_ratio=0.8):
        self.stop_ratio=stop_ratio

    def __call__(self, nn, train_history):
        ratio = train_history[-1]['train_loss']/train_history[-1]['valid_loss']
        if ratio<self.stop_ratio:
            nn.max_epochs = train_history[-1]['epoch']
