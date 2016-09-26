from lib.cells import DenseCell
from lib.activations import *

class SimpleLinearNet:

  fc1 = DenseCell(2,2, name="dense1")

  smce = SoftmaxCrossEntropy()

  train_objects = [fc1]


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

    loss = self.smce.forward(z1, target)
    prediction = self.smce.softmax(z1)

    return loss, prediction


  def backward(self):
    """
    Performs a backward pass. No inputs are needed since every object saves the
    necessary variables during the forward pass.
    :return:
    """
    curr_delta = self.smce.backward()
    curr_delta = self.fc1.backward(curr_delta)

