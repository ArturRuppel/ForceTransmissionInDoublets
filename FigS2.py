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
AR1to1d_fullstim_long = pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_long.dat", "rb"))
AR1to1d_fullstim_short = pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_short.dat", "rb"))
AR1to1d_halfstim = pickle.load(open(folder + "analysed_data/AR1to1d_halfstim.dat", "rb"))

AR1to1s_fullstim_long = pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_long.dat", "rb"))
AR1to1s_fullstim_short = pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_short.dat", "rb"))
AR1to1s_halfstim = pickle.load(open(folder + "analysed_data/AR1to1s_halfstim.dat", "rb"))

# define some colors for the plots
colors_parent = ['#026473', '#E3CC69', '#77C8A6', '#D96248']

figfolder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/_FigureS2/"
if not os.path.exists(figfolder):
    os.mkdir(figfolder)
# %% set up pandas data frame to use with seaborn for box- and swarmplots

# initialize empty dictionaries
concatenated_data_1to1d = {}
concatenated_data_1to1s = {}
concatenated_data = {}

# loop over all keys
for key1 in AR1to1d_fullstim_long:
    for key2 in AR1to1d_fullstim_long[key1]:
        if AR1to1d_fullstim_long[key1][key2].ndim == 1:  # only 1D data can be stored in the data frame

            # concatenate values from different experiments
            concatenated_data_1to1d[key2] = np.concatenate(
                (AR1to1d_halfstim[key1][key2], AR1to1d_fullstim_short[key1][key2], AR1to1d_fullstim_long[key1][key2]))
            concatenated_data_1to1s[key2] = np.concatenate(
                (AR1to1s_halfstim[key1][key2], AR1to1s_fullstim_short[key1][key2], AR1to1s_fullstim_long[key1][key2]))

            # concatenate doublet and singlet data to create pandas dataframe
            concatenated_data[key2] = np.concatenate((concatenated_data_1to1d[key2], concatenated_data_1to1s[key2]))

# get number of elements for both conditions
n_doublets = concatenated_data_1to1d[key2].shape[0]
n_singlets = concatenated_data_1to1s[key2].shape[0]

# create a list of keys with the same dimensions as the data
keys1to1d = ['AR1to1d' for i in range(n_doublets)]
keys1to1s = ['AR1to1s' for i in range(n_singlets)]
keys = np.concatenate((keys1to1d, keys1to1s))

# add keys to dictionary with concatenated data
concatenated_data['keys'] = keys

# create DataFrame
df = pd.DataFrame(concatenated_data)

# convert to more convenient units for plotting
df['Es_baseline'] *= 1e12  # convert to fJ
df['spreadingsize_baseline'] *= 1e12  # convert to µm²
df['sigma_xx_baseline'] *= 1e3  # convert to mN/m
df['sigma_yy_baseline'] *= 1e3  # convert to mN/m

# %% plot figure S1B correlation plot of stress anisotropy and actin anisotropy

# set up global plot parameters
# ******************************************************************************************************************************************
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 2.5))  # create figure and axes
plt.subplots_adjust(wspace=0.45, hspace=0.45)  # adjust space in between plots


ylabeloffset = -7
xlabeloffset = 0
colors = [colors_parent[1], colors_parent[2]]  # defines colors for scatterplot

y = 'actin_anisotropy_coefficient'
x = 'AIC_baseline'
hue = 'keys'
ymin = -1
ymax = 1
xmin = -1
xmax = 1
yticks = np.arange(-1, 1.1, 0.5)
xticks = np.arange(-1, 1.1, 0.5)
ylabel = "Structural polarization (actin)"  # "'$\mathrm{\sigma_{x, MSM}}$'
xlabel = "Mechanical polarization"  # '$\mathrm{\sigma_{x, CM}}$'

corr, p = make_correlationplotsplots(x, y, hue, df, ax, xmin, xmax, ymin, ymax, xticks, yticks, xlabel, ylabel, colors)

# add line with slope 1 for visualisation
# ax.plot([ymin, ymax], [0, 0], linewidth=0.5, linestyle=':', color='grey')
# ax.plot([45, 45], [xmin, xmax], linewidth=0.5, linestyle=':', color='grey')

plt.text(0.21 * xmax + xmin, 1.05 * ymax, 'R = ' + str(corr))
# plt.text(0.52 * xmax, 1.1 * ymax, 'p = ' + '{:0.2e}'.format(p))

plt.savefig(figfolder + 'B.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'B.svg', dpi=300, bbox_inches="tight")
plt.show()
