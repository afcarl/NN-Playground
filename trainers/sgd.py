import numpy as np
import math

class SGD:

  def __init__(self, X, Y, epochs, learning_rate, train_batch_size, eval_batch_size, network):
    assert train_batch_size <= X.shape[0], "batch size cannot be larger than the number of samples"
    assert eval_batch_size <= X.shape[0], "batch size cannot be larger the the number of samples"

    self.X = X
    self.Y = Y
    self.epochs = epochs
    self.learning_rate = learning_rate
    self.train_batch_size = train_batch_size
    self.eval_batch_size = eval_batch_size
    self.network = network

  def gradient_descent_batch(self, batch, targets):
    _, pred = self.network.forward(batch, targets)
    print(" batch accuracy: {:0.4f} ".format((1.0/batch.shape[0]) * np.sum(np.argmax(targets, axis=1) == np.argmax(pred, axis=1))))
    self.network.backward()

    for cell in self.network.train_objects:
      dW_list = cell.get_gradient()
      dW_avg = np.mean(dW_list, axis=0)
      dW_step = self.learning_rate * dW_avg
      cell.update_weights(dW_step)

  def step(self, batch, targets):
    self.gradient_descent_batch(batch, targets)

  def train(self):
    print("pre evaluation:")
    self.evaluate()

    batches_per_epoch = math.ceil(self.X.shape[0] / self.train_batch_size)
    print("batches per epoch: ", batches_per_epoch)
    for i in range(self.epochs):
      print("Epoch {}:".format(i))
      for j in range(batches_per_epoch):
        start_idx = j * self.train_batch_size
        end_idx = start_idx + self.train_batch_size

        batch = self.X[start_idx:end_idx]
        targets = self.Y[start_idx:end_idx]
        print("epoch {:0.3f}".format(i + j/batches_per_epoch), end="")
        self.step(batch, targets)

      self.evaluate()

  def evaluate(self):
    correct = 0
    batches_per_epoch = math.ceil(self.X.shape[0] / self.eval_batch_size)

    for j in range(batches_per_epoch):
      start_idx = j * self.eval_batch_size
      end_idx = start_idx + self.eval_batch_size

      batch = self.X[start_idx:end_idx]
      targets = self.Y[start_idx:end_idx]

      prediction = self.network.forward(batch, targets)[1]
      correct += np.sum(np.argmax(targets, axis=1) == np.argmax(prediction, axis=1))

    print("Correct: ", correct)
    print("Samples: ", self.X.shape[0])
    print("Accuracy: ", (1.0 / self.X.shape[0]) * correct)









