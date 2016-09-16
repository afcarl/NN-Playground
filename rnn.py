
import numpy as np
from lib.cells import DenseCell
from lib.activations import *

def forward(batch, target):
  """
  Performs a forward pass through a fully connected network.
  :param batch_sequence: numpy ndarray with rows as samples and cols input dimensions.
  :return:
  """
  assert type(batch) == np.ndarray, "batch input must be an numpy ndarray"
  assert batch.shape[1] != 0, "batch must have at least 2 dimensions with sample rows " \
                              "and data dimensions as columns."
  assert target.shape[1] != 0, "target must have at least 2 dimensions with sample rows " \
                               "and one-hot class predicitons as columns."
  assert batch.shape[0] == target.shape[0], "batch and targets don't have the same number " \
                                            "of samples"

  # number of samples
  N = batch.shape[0]

  # input data dimensions
  D = batch.shape[0]

  fc1 = DenseCell(2,2)
  tanh1 = Tanh()

  z1 = fc1.forward(batch)
  a1 = tanh1.forward(z1)

  return a1
