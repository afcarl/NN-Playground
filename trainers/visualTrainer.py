import numpy as np
from time import sleep

class VisualTrainer:

  def __init__(self, trainer):
    self.trainer = trainer
    self.trainer.step = self.decorator_step
    self.generate_plot_data()

  def generate_plot_data(self):
    # generate X data for red and blue class (two gaussian which can separated with one hyperplane)
    x1 = np.linspace(-1, 1, 20)
    x2 = np.linspace(-1, 1, 20)

    X1, X2 = np.meshgrid(x1, x2)
    self.Xplot = np.dstack((np.ravel(X1), np.ravel(X2)))[0]
    self.Yplot = np.zeros((self.Xplot.shape[0],self.trainer.Y.shape[1])) # targets don't matter for visualization


  def decorator_step(self, batch, targets):
    #print("perform visualization ...")
    self.trainer.network.forward(self.Xplot, self.Yplot, visualize=True)

    #print("perform training ...")
    self.trainer.gradient_descent_batch(batch, targets)

    sleep(0.05)
    #input("press key for next batch ...")

  def train(self):
    self.trainer.train()



