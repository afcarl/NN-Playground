"""
Train the FC MNIST network to classify digits.
"""

import numpy as np
import misc.utils
import trainers.sgd

from data import mnist
from networks.fcMnist import FullyConnectedFeedForward


def load_data(mnist_path):
  number_of_labels = 10

  train_images, train_labels = mnist.load_mnist(dataset="training", path=mnist_path)
  tl = np.zeros((train_labels.shape[0], number_of_labels))
  for i in range(train_labels.shape[0]):
    tl[i][train_labels[i]] = 1
  train_labels = tl
  train_images = train_images.flatten().reshape(60000, 784)
  return train_images, train_labels

X, Y = load_data("/home/schlag/MyStuff/Data/MNIST/")
misc.utils.shuffle_in_unison(X, Y)

net = FullyConnectedFeedForward()
sgd = trainers.sgd.SGD()

sgd.train(X, Y, epochs=5, batch_size=200, learning_rate=0.01, network=net)


