import numpy as np
import typing

from lib.graph import GraphOperation

class Tanh(GraphOperation):
  """
  Stateful hyperbolic tangent activation.
  """

  def __call__(self, z):
    return self.forward(z)

  def forward(self, z):
    """
    Compute the hyperbolic tangent activation.
    :param z: pre activation value
    :return: neuron activation
    """
    assert type(z) == np.ndarray, "delta is not a np.ndarray"

    self.z = z
    self.a = np.tanh(self.z)
    return self.a

  def backward(self, delta):
    """
    Compute the backprop delta error after a forward pass.
    :param delta: higher level error delta
    :return: delta processed by the prime of this activation
    """
    assert type(delta) == np.ndarray, "delta is not a np.ndarray"
    assert delta.shape == self.z.shape, "delta and pre-activation are not the same shape"

    dz = 1.0 - np.tanh(self.z)**2
    return delta * dz


class Sigmoid(GraphOperation):
  """
  Stateful sigmoid activation.
  """

  def __call__(self, z):
    return self.forward(z)

  def forward(self, z):
    """
    Compute the sigmoid activation.
    :param z: pre activation value
    :return: neuron activation
    """
    assert type(z) == np.ndarray

    self.z = z
    self.a = np.nan_to_num([1 / (1 + np.exp(-self.z))])
    return self.a

  def backward(self, delta):
    """
    Compute the backprop delta error after a forward pass.
    :param delta: higher level error delta
    :return: delta processed by the prime of this activation
    """
    assert delta.shape == self.z.shape

    dz = self.z * (1 - self.z)
    return delta * dz


class SoftmaxCrossEntropy(GraphOperation):
  """
  The softmax cross entropy function.
  """

  def __call__(self, z, y):
    return self.forward(z, y)

  def softmax(self, z):
    """
    Computes the softmax activation.
    :param z: pre activation value (2 dimensional i.e. sample per row)
    :return: quasi probability output / activation of this neuron
    """
    assert type(z) == np.ndarray, "input is not an ndarray"
    assert len(z.shape) == 2, "z input is not 2 dimensional"

    s = np.max(z, axis=1) # should I not max over the cols as in here (???) http://cs231n.github.io/linear-classify/#softmax
    s = s[:, np.newaxis] # necessary step to do broadcasting in this situation with python3
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div


  def cross_entropy_cost(self, y_pred, y):
    """
    Computes the cross_entropy loss per sample.
    :param y_pred: prediction of y
    :param y: real y from our data
    :return: cross entropy cost of y_pred per sample
    """
    assert type(y_pred) == np.ndarray, "y_pred is not an ndarray"
    assert type(y) == np.ndarray, "y is not an ndarray"
    assert y.shape == y_pred.shape, "y and y_pred don't have the same shape"
    assert len(y.shape) == 2

    return -np.sum(np.nan_to_num(y * np.log(y_pred)), axis=1)
    #return -np.nan_to_num(y * np.log(y_pred))

  def forward(self, z, y):
    """
    Computes the softmax activation and the cross_entropy loss.
    :param z: softmax input i.e. pre-activation
    :param y: target y
    :return: cost for every sample
    """
    assert type(z) == np.ndarray, "z is not an ndarray"
    assert type(y) == np.ndarray, "y is not an ndarray"
    assert z.shape == y.shape, "y and z don't have the same shape"

    self.z = z
    self.y = y
    self.y_pred = self.softmax(self.z)
    return self.cross_entropy_cost(self.y_pred, self.y)

  def backward(self):
    """
    Computes the backprop delta error based on the forward input.
    :return: delta processed by the derivative of the softmax cross entropy function
    """

    #return (self.y_pred - self.y)[range(self.y_pred.shape[0]), np.argmax(self.y, axis=1)]
    return self.y_pred - self.y

