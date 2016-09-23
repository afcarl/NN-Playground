from lib.cells import DenseCell
from lib.activations import *

class FullyConnectedFeedForward:

  fc1 = DenseCell(20, 20, name="dense1")
  tanh1 = Tanh()

  fc2 = DenseCell(20, 20, name="dense2")
  tanh2 = Tanh()

  fc3 = DenseCell(20, 20, name="dense3")
  tanh3 = Tanh()

  smce = SoftmaxCrossEntropy()

  train_objects = [fc1, fc2, fc3]

  def forward(self, batch, target):
    """
    Performs a forward pass through a fully connected network and prepares it for the backward pass.
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

    z1 = self.fc1(batch)
    a1 = self.tanh1(z1)

    z2 = self.fc2(a1)
    a2 = self.tanh2(z2)

    z3 = self.fc3(a2)
    a3 = self.tanh3(z3)

    loss = self.smce.forward(a3, target)
    prediction = self.smce.softmax(a3)

    return loss, prediction

  def backward(self):
    """
    Performs a backward pass. No inputs are needed since every object saves the
    necessary variables during the forward pass.
    :return:
    """
    curr_delta = self.smce.backward()

    curr_delta = self.tanh3.backward(curr_delta)
    curr_delta = self.fc3.backward(curr_delta)

    curr_delta = self.tanh2.backward(curr_delta)
    curr_delta = self.fc2.backward(curr_delta)

    curr_delta = self.tanh1.backward(curr_delta)
    curr_delta = self.fc1.backward(curr_delta)

