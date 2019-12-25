import numpy as np
import numpy.random as random

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
    self.num_layers = len(shape)
    #end

  def forward(self, a):
    """Return the output of the network if ``a`` is input."""
    for b, w, f in zip(self.bias, self.weight, self.actors):
      a = f(np.dot(w, a) + b)
    return a
    #end

  def back(self, x, y):
    """Backpropagation"""
    nabla_b = [np.zeros(b.shape) for b in self.bias]
    nabla_w = [np.zeros(w.shape) for w in self.weight]
    # feedforward
    activation = x
    activations = [x] # list to store all the activations, layer by layer
    zs = [] # list to store all the z vectors, layer by layer
    for b, w, f in zip(self.bias, self.weight, self.actors):
      z = np.dot(w, activation) + b
      zs.append(z)
      activation = f(z)
      activations.append(activation)
      #end for
    
    # backward pass
    delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())

    for i in range(2, self.num_layers):
      z = zs[-i]
      sp = sigmoid_prime(z)
      delta = np.dot(self.weight[-i+1].transpose(), delta) * sp
      nabla_b[-i] = delta
      nabla_w[-i] = np.dot(delta, activations[-i-1].transpose())
      #end for
    return (nabla_b, nabla_w)
    #end Backpropagation

  def cost_derivative(self, output_activations, y):
    """Return the vector of partial derivatives \partial C_x /
       \partial a for the output activations."""
    return (output_activations - y)

  def train(self, data, epochs, batch_size, eta):
    """Training..."""
    n = len(data)
    rate_ = 0
    for j in range(epochs):
      
      random.shuffle(data)
      batches = [data[k:k+batch_size] for k in range(0, n, batch_size)]
      #print(len(batches))
      ii = 0
      for batch in batches:
        #print(ii)
        ii = ii + 1
        self.updateBatch(batch, eta)
      size_ok = self.evaluate(data)
      rate_ok = size_ok / n
      delta_rate = rate_ok - rate_
      rate_ = rate_ok
      
      print("Epoch {0}:\t {1} / {2}\t ==> {3}\t {4}".format(
        j, size_ok, n, rate_ok, ":)" if delta_rate > 0 else ":("))

        #end for batch
      #end for j
    #end train

  def updateBatch(self, batch, eta):

    nabla_b = [np.zeros(b.shape) for b in self.bias]
    nabla_w = [np.zeros(w.shape) for w in self.weight]
    for x, y in batch:
      delta_nabla_b, delta_nabla_w = self.back(x, y)
      nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
      nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    self.weight = [w-(eta/len(batch))*nw for w, nw in zip(self.weight, nabla_w)]
    self.bias   = [b-(eta/len(batch))*nb for b, nb in zip(self.bias, nabla_b)]

  def evaluate(self, data):

    results = [(np.argmax(self.forward(x)), np.argmax(y)) for (x, y) in data]
    return sum(int(x == y) for (x, y) in results)



def sigmoid_prime(z):
  """Derivative of the sigmoid function."""
  return sigmoid(z) * (1 - sigmoid(z))






