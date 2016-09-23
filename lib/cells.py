import numpy as np
import typing

from lib.graph import GraphOperation


class DenseCell(GraphOperation):
  """
  Fully connected layer without activation.
  """

  def __init__(self, size_in, size_out, stddev=0.01, name="empty-name"):
    self.W = np.random.normal(scale=stddev, size=size_in*size_out).reshape((size_in, size_out))
    self.z = np.zeros(size_out)
    self.name = name
    self.delta = np.zeros(size_out)

  def __call__(self, z):
    return self.forward(z)

  def get_name(self):
    return self.name

  def forward(self, x):
    """
    Compute the output of the fully connected layer (without activation).
    :param x: input tensor
    :return: output tensor z (linear combination)
    """
    assert type(x) == np.ndarray, "x is not a np.ndarray"
    assert x.shape[1] == self.W.shape[0], "x "+x.shape+" doesn't match W " + self.W.shape
    self.z = x
    return np.dot(x, self.W)
 
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
    output = []
    for i in range(self.z.shape[0]):
      output.append(np.outer(self.z[i], self.delta[i]))
    return output

  def get_weights(self):
    """
    Getter method for the weights of this cell.
    :return: weight matrix
    """
    return self.W

  def update_weights(self, delta_update):
    """
    Substracts the delta_update from the given weight matrix W.
    :param delta_update: the update which will be subtracted from the weights.
    :return: None
    """
    assert type(delta_update) == np.ndarray
    assert delta_update.shape == self.W.shape
    self.W -= delta_update

