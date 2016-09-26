import numpy as np
import plot.plotNeuron
from plot.plotNeuron import plotNeuronPlane

x1 = np.random.normal(-0.5, 0.2, 100)
y1 = np.random.normal(-0.5, 0.2, 100)
color1 = np.array([1,0,0])

x2 = np.random.normal(0.5, 0.2, 100)
y2 = np.random.normal(0.5, 0.2, 100)
color2 = np.array([0,0,1])

pnp = plotNeuronPlane(x1, y1, color1, x2, y2, color2)




# visualization batch generation

x1 = np.linspace(-1, 1, 3)
x2 = np.linspace(-1, 1, 3)

X1, X2 = np.meshgrid(x1,x2)
batch = np.dstack((np.ravel(X1), np.ravel(X2)))
