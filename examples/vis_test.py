"""
.. versionadded:: 1.1.0
   This demo depends on new features added to contourf3d.
"""

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)



def fun(x, y):
  return x**2 + y**2

x = y = np.arange(-3.0, 3.0, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)


ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)

cset = ax.contourf(X, Y, Z, zdir='z', offset=-30, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, Z, zdir='x', offset=-5, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, Z, zdir='y', offset=5, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(-5, 5)
ax.set_ylabel('Y')
ax.set_ylim(-5, 5)
ax.set_zlabel('Z')
ax.set_zlim(-30, 30)



plt.show()
