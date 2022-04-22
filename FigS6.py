"""

@author: Artur Ruppel

"""
import os
import pickle
import pandas as pd
from scipy.stats import zscore
from plot_and_filter_functions import *
from scipy.interpolate import interp1d

pixelsize = 0.864  # in µm
# %% load data for plotting
folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"

# define some colors for the plots
colors_parent = ['#026473', '#E3CC69', '#77C8A6', '#D96248']
colors_parent_dark = ['#01353D', '#564910', '#235741', '#A93B23']

colors = [colors_parent[0], colors_parent[1], colors_parent[3], colors_parent_dark[0], colors_parent_dark[1]]

figfolder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/_FigureS6/"
if not os.path.exists(figfolder):
    os.mkdir(figfolder)

summarized_data = pd.read_csv(figfolder + "active_coupling_vs_anisotropies.csv")
# df2 = summarized_data.stack()
# df2 = df2.reset_index()
# df2.columns = ['1to2', '1to1', '2to1', 'tissue_lefthalf', 'tissue_tophalf']

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2, 2))
sns.scatterplot(data=summarized_data, x="active coupling", y="junction length", hue="condition", style="condition", palette=colors, legend=False)
fig.savefig(figfolder + 'A.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'A.svg', dpi=300, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2, 2))
sns.scatterplot(data=summarized_data, x="active coupling", y="stress anisotropy", hue="condition", style="condition", palette=colors, legend=False)
fig.savefig(figfolder + 'B.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'B.svg', dpi=300, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2, 2))
sns.scatterplot(data=summarized_data, x="active coupling", y="actin anisotropy", hue="condition", style="condition", palette=colors, legend=False)
fig.savefig(figfolder + 'C.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'C.svg', dpi=300, bbox_inches="tight")
plt.show()

