import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# the random data
color1 = np.array([1.0, 0.0, 0.0]) # red
x1 = np.random.randn(3)
y1 = np.random.randn(3)

fig, axScatter = plt.subplots(figsize=(5.5, 5.5))

# the scatter plot:
axScatter.scatter(x1, y1, c=color1)
axScatter.set_aspect(1.)

# create new axes on the right and on the top of the current axes
# The first argument of the new_vertical(new_horizontal) method is
# the height (width) of the axes to be created in inches.
#divider = make_axes_locatable(axScatter)
#axHistx = divider.append_axes("top", 1.2, pad=0.1, sharex=axScatter)
#axHisty = divider.append_axes("right", 1.2, pad=0.1, sharey=axScatter)

# make some labels invisible
#plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),
#         visible=False)

# now determine nice limits by hand:
#binwidth = 0.25
#xymax = np.max([np.max(np.fabs(x1)), np.max(np.fabs(y1))])
#lim = (int(xymax/binwidth) + 1)*binwidth

#bins = np.arange(-lim, lim + binwidth, binwidth)
#axHistx.hist(x1, bins=bins)
#axHisty.hist(y1, bins=bins, orientation='horizontal')

# the xaxis of axHistx and yaxis of axHisty are shared with axScatter,
# thus there is no need to manually adjust the xlim and ylim of these
# axis.

#axHistx.axis["bottom"].major_ticklabels.set_visible(False)
#for tl in axHistx.get_xticklabels():
#    tl.set_visible(False)
#axHistx.set_yticks([0, 50, 100])

#axHisty.axis["left"].major_ticklabels.set_visible(False)
#for tl in axHisty.get_yticklabels():
#    tl.set_visible(False)
#axHisty.set_xticks([0, 50, 100])

plt.draw()
plt.show()