import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import sys
sys.path.append('/Users/dennis/customPyModules')
from pubplots import *

data = np.load('contour_sim_data_doublet.npz')

x = data['x']
y = data['y']

landa = data['landa_aPA']
print(landa)

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

latexify_palatino_euler()
fig = plt.figure()
# fig.subplots_adjust(wspace = 0.3)#, hspace = 0.3
ax1 = fig.add_subplot(1,1,1)
# Create a continuous norm to map from data points to colors
norm = plt.Normalize(100, 250)
lc = LineCollection(segments, cmap='turbo', norm=norm)
# Set the values used for colormapping
lc.set_array(landa)
lc.set_linewidth(2)
line = ax1.add_collection(lc)
cbar = fig.colorbar(line, ax=ax1)
cbar.set_label(r'line tension [nN]', rotation=270)
cbar.ax.get_yaxis().labelpad = 15

ax1.set_title(r"Contour")
ax1.set_ylabel(r"$y(x)$ [$\mu \textrm{m}$]")
ax1.set_xlabel(r"$x$ [$\mu \textrm{m}$]")

ax1.set_xlim(x.min()-5, x.max()+5)
ax1.set_ylim(-15, 10)
# ax1.set_xlim(0,350)
# ax1.grid()
ax1.legend(loc='best')

format_axes_palatino_euler(ax1)

# fig.tight_layout()
ax1.ticklabel_format(useOffset=False)
fig.savefig("contour_with_line_tension.pdf", bbox_inches='tight')

