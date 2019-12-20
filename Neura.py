import numpy as np


def sigmoid(a):
  return 1 / (1 + np.exp(-a))
  #end

def hardlim(a):
  return np.where(a > 0, 1, 0)
  #end


class Network:
  """
  A toy implemention of neural network framework
  - Author: Comcx
  - Email:  comcx@outlook.com
  """
  def __init__(self, shape, actors = None):
    """
    - shape: the size of every layer.
    - actors: actor function of each layer.
    """
    self.shape  = shape
    self.bias   = [np.random.randn(y, 1) for y in shape[1:]]
    self.weight = [np.random.randn(y, x) for x, y in zip(shape[:-1], shape[1:])]
    self.actors = actors if actors else [sigmoid] * (len(shape) - 1)
    #end

  def forward(self, a):
    """Return the output of the network if ``a`` is input."""
    for b, w, f in zip(self.bias, self.weight, self.actors):
      a = f(np.dot(w, a) + b)
    return a
    #end

  def back(self, x, y):
    return 0



