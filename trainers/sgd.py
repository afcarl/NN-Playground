import numpy as np
import math

class SGD:

  def __gradient_descent_batch(self, batch, targets, learning_rate, net):
    """
    Performs one training iteration and updates the weights of all trainable objects in the network.
    :param batch: mini-batch containing all inputs
    :param targets: one-hot encoded target classes
    :param learning_rate: the learning rate
    :param net: the network to train
    :return: None
    """
    _, pred = net.forward(batch, targets)
    print(" batch accuracy: {:0.4f} ".format((1.0/batch.shape[0]) * np.sum(np.argmax(targets, axis=1) == np.argmax(pred, axis=1))))
    net.backward()

    for cell in net.train_objects:
      dW_list = cell.get_gradient()
      dW_avg = np.mean(dW_list, axis=0)
      dW_step = learning_rate * dW_avg
      cell.update_weights(dW_step)


  def train(self, X, Y, epochs, batch_size, learning_rate, network):
    assert batch_size <= X.shape[0], "batch size cannot be larger than the number of samples"

    print("pre evaluation:")
    self.evaluate(X, Y, batch_size=2000, network=network)

    batches_per_epoch = math.ceil(X.shape[0] / batch_size)
    print("batches per epoch: ", batches_per_epoch)
    for i in range(epochs):
      print("Epoch {}:".format(i))

      for j in range(batches_per_epoch):

        start_idx = j * batch_size
        end_idx = start_idx + batch_size

        batch = X[start_idx:end_idx]
        targets = Y[start_idx:end_idx]
        print("epoch {:0.3f}".format(i + j/batches_per_epoch), end="")
        self.__gradient_descent_batch(batch, targets, learning_rate, network)

      self.evaluate(X, Y, batch_size=2000, network=network)

  def evaluate(self, X, Y, batch_size, network):
    assert batch_size <= X.shape[0], "batch size cannot be larger the the number of samples"

    correct = 0
    batches_per_epoch = math.ceil(X.shape[0] / batch_size)

    for j in range(batches_per_epoch):
      start_idx = j * batch_size
      end_idx = start_idx + batch_size

      batch = X[start_idx:end_idx]
      targets = Y[start_idx:end_idx]

      prediction = network.forward(batch, targets)[1]
      correct += np.sum(np.argmax(targets, axis=1) == np.argmax(prediction, axis=1))

    print("Correct: ", correct)
    print("Samples: ", X.shape[0])
    print("Accuracy: ", (1.0 / X.shape[0]) * correct)









