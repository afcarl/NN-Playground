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

