import numpy as np

from lib.activations import Tanh
from lib.cells import DenseCell

import rnn

def npcast(x):
  if not type(x) == list:
    return np.asarray([x])
  else:
    return np.asarray(x)

batch = [
  [0,2],
  [0,2],
  [0,2]
]

target = [
  [1,0],
  [1,0],
  [1,0]
]

batch = npcast(batch)
target = npcast(target)

pred = rnn.forward(batch, target)

print(pred)