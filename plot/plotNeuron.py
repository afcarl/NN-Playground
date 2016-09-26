import numpy as np
import matplotlib.pyplot as plt

#%matplotlib


class plotNeuronPlane:
  """
  Plots the activation of a neuron
  """
  def __init__(self, x1, y1, color1, x2, y2, color2, name="unnamed"):

    plt.ion()

    self.fig, self.ax = plt.subplots(figsize=(5.5, 5.5))
    self.fig.canvas.set_window_title(name)
    self.fig.show()
    self.fig.canvas.draw()

    # the scatter plot:
    self.scat1 = self.ax.scatter(x1, y1, c=color1)
    self.scat2 = self.ax.scatter(x2, y2, c=color2)

    # background
    self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    # adjust range of axes
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    # adjust ticks
    plt.xticks(np.arange(-1, 1, 0.2))
    plt.yticks(np.arange(-1, 1, 0.2))

    # enable grid
    plt.grid()

  def update(self):
    self.fig.canvas.restore_region(self.bg)

    # redraw dots
    self.ax.draw_artist(self.scat1)
    self.ax.draw_artist(self.scat2)

    # draw activation
    # TODO

    self.fig.canvas.blit(self.ax.bbox)
