import numpy as np
import typing

from lib.graph import GraphOperation


class DenseCell(GraphOperation):
  """
  Fully connected layer without activation.
  """

  def __init__(self, size_in, size_out, stddev=0.01):
    self.W = np.random.normal(scale=stddev, size=size_in*size_out).reshape((size_in, size_out))
    self.z = np.zeros(size_out)
    self.delta = np.zeros(size_out)

  def __call__(self, z):
    return self.forward(z)

  def forward(self, x):
    """
    Compute the output of the fully connected layer (without activation).
    :param x: input tensor
    :return: output tensor z (linear combination)
    """
    assert type(x) == np.ndarray, "x is not a np.ndarray"
    assert x.shape[1] == self.W.shape[0], "x "+x.shape+" doesn't match W " + self.W.shape

    self.z = np.dot(x, self.W)
    return self.z
 
  def backward(self, delta):
    """
    Compute the backprop delta error after a forward pass.
    :param x: higher level error delta
    :return: delta multiplied with the transpose of W
    """
    self.delta = delta
    return np.dot(self.delta, self.W.transpose())

  def get_gradient(self):
    """
    Compute the gradient based on the prior forward and backward pass.
    :return: returns the gradient of the
    """
    return np.outer(self.z, self.delta)

  def update_weights(self, delta_update):
    """
    Substracts the delta_update from the given weight matrix W.
    :param delta_update: the update which will be subtracted from the weights.
    :return: None
    """
    assert type(delta_update) == np.ndarray
    assert delta_update.shape == self.W.shape
    self.W -= delta_update

