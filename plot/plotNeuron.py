import numpy as np
import matplotlib.pyplot as plt

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

class plotLayer:

  n = 20

  def __init__(self, fig, figure_count, layer, name="unnamed"):
    plt.ion()
    self.fig = fig
    self.axes = [self.fig.add_subplot(self.fig.max_per_layer, self.fig.layers, (self.fig.layers*i)+layer, aspect=1.0) for i in range(figure_count)]
    for ax in self.axes:
      ax.set_axis_off()
    init_data = np.zeros((self.n,self.n))
    self.qm = [self.axes[i].pcolormesh(init_data, cmap='RdYlBu_r') for i in range(figure_count)]
    self.cbar = self.fig.colorbar(self.qm[-1], ax=self.axes)

    #self.bg = self.fig.canvas.copy_from_bbox(self.ax1.bbox) # saving background doesn't makes sense

    self.fig.show()
    self.fig.canvas.draw()

  def update(self, data, visualize):
    if not visualize:
      return

    #plots = [data[:, i] for i in range(data.shape[1])]
    for i in range(data.shape[1]):
      self.qm[i].set_array(data[:, i])
      self.qm[i].autoscale()

    #self.ax1.draw_artist(self.qm1)
    #self.fig.canvas.update()       # doesn't work with TkAgg
    #self.fig.canvas.blit(self.ax1.bbox)
    #self.fig.canvas.flush_events()

    self.fig.canvas.draw() # slightly slower than above

  def calc_pos(self, layer, row_size, curr_pos):
    return curr_pos*row_size + layer