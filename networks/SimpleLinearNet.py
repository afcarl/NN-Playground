from lib.cells import DenseCell
from lib.activations import *
from plot.plotNeuron import plotLayer

class SimpleLinearNet:
  visualize_mode = False

  import matplotlib.pyplot as plt
  fig = plt.figure("SimpleLinearNet")
  fig.layers = 2
  fig.max_per_layer = 2

  fc1 = DenseCell(2,2, name="dense1")
  plot1 = plotLayer(fig, figure_count=2, layer=1, name="layer1")

  smce = SoftmaxCrossEntropy()
  plot2 = plotLayer(fig, figure_count=2, layer=2, name="layer2")

  train_objects = [fc1]


  def forward(self, batch, target, visualize=False):
    """
    Performs a forward pass through a f§ully connected network and prepares it for the backward pass.
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
    self.plot1.update(z1, visualize)

    prediction = self.smce.softmax(z1)
    self.plot2.update(np.round(prediction), visualize)

    loss = self.smce.forward(z1, target)

    return loss, prediction


  def backward(self):
    """
    Performs a backward pass. No inputs are needed since every object saves the
    necessary variables during the forward pass.
    :return:
    """
    curr_delta = self.smce.backward()
    curr_delta = self.fc1.backward(curr_delta)

