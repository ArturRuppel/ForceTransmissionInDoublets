# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 21:56:01 2021

@author: Artur Ruppel

"""
import os
import pickle
import pandas as pd
from plot_and_filter_functions import *

mpl.rcParams['font.size'] = 8

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

figfolder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/_Figure4/"
if not os.path.exists(figfolder):
    os.mkdir(figfolder)

# %% prepare dataframe for boxplots
# initialize empty dictionaries
concatenated_data_fs = {}
concatenated_data_hs = {}
concatenated_data_doublet = {}
concatenated_data_singlet = {}

# loop over all keys
for key1 in AR1to1d_fullstim_long:  # keys are the same for all dictionaries so I'm just taking one example here
    for key2 in AR1to1d_fullstim_long[key1]:
        if AR1to1d_fullstim_long[key1][key2].ndim == 1:  # only 1D data can be stored in the data frame
            concatenated_data_doublet[key2] = np.concatenate(
                (AR1to1d_fullstim_long[key1][key2], AR1to1d_halfstim[key1][key2]))
            concatenated_data_singlet[key2] = np.concatenate(
                (AR1to1s_fullstim_long[key1][key2], AR1to1s_halfstim[key1][key2]))
key1 = "TFM_data"
key2 = "Es_baseline"

# get number of elements for both condition
n_d_fullstim = AR1to1d_fullstim_long[key1][key2].shape[0]
n_d_halfstim = AR1to1d_halfstim[key1][key2].shape[0]
n_s_fullstim = AR1to1s_fullstim_long[key1][key2].shape[0]
n_s_halfstim = AR1to1s_halfstim[key1][key2].shape[0]

# create a list of keys with the same dimensions as the data
keys1to1d_fs = ['AR1to1d_fs' for i in range(n_d_fullstim)]
keys1to1s_fs = ['AR1to1s_fs' for i in range(n_s_fullstim)]
keys1to1d_hs = ['AR1to1d_hs' for i in range(n_d_halfstim)]
keys1to1s_hs = ['AR1to1s_hs' for i in range(n_s_halfstim)]

keys_doublet = np.concatenate((keys1to1d_fs, keys1to1d_hs))
keys_singlet = np.concatenate((keys1to1s_fs, keys1to1s_hs))

# add keys to dictionary with concatenated data
concatenated_data_doublet['keys'] = keys_doublet
concatenated_data_singlet['keys'] = keys_singlet

# Creates DataFrame
df_doublet = pd.DataFrame(concatenated_data_doublet)
df_singlet = pd.DataFrame(concatenated_data_singlet)

# %% plot figure B, contour strain after photoactivation

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.linspace(-17.5, 17.5, 50)
x = x[::2]  # downsample data for nicer plotting
ymin = -0.06
ymax = 0
xticks = np.arange(-15, 15.1, 15)  # define where the major ticks are gonna be
yticks = np.arange(ymin, ymax + 0.001, 0.02)
xlabel = 'position [µm]'
xticklabels = ['global \n act.', 'local \n act.']  # which labels to put on x-axis
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(5, 3))  # create figure and axes
plt.subplots_adjust(wspace=0.4, hspace=0.35)  # adjust space in between plots
# ******************************************************************************************************************************************

# Set up plot parameters for first panel
#######################################################################################################
ax = axes[0, 0]
color = colors_parent[1]
ylabel = 'doublet'
title = 'global activation'
y = AR1to1d_fullstim_long["shape_data"]["contour_strain"]
y = y[::2, :]
# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False)

# Set up plot parameters for second panel
#######################################################################################################
ax = axes[0, 1]
color = colors_parent[1]
ylabel = None
title = 'local activation'
y = AR1to1d_halfstim["shape_data"]["contour_strain"]
y = y[::2, :]
# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False)

# Set up plot parameters for third panel
#######################################################################################################
ax = axes[1, 0]
color = colors_parent[2]
ylabel = 'singlet'
title = None
y = AR1to1s_fullstim_long["shape_data"]["contour_strain"]
y = y[::2, :]
# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False)

# Set up plot parameters for fourth panel
#######################################################################################################
ax = axes[1, 1]
color = colors_parent[2]
ylabel = None
title = None
y = AR1to1s_halfstim["shape_data"]["contour_strain"]
y = y[::2, :]
# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False)

# Set up plot parameters for fifth panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'ASC'  # variable that goes on the y-axis
ax = axes[0, 2]  # define on which axis the plot goes
colors = [colors_parent[1], colors_parent[1]]  # defines colors
ymin = -1  # minimum value on y-axis
ymax = 1  # maximum value on y-axis
yticks = np.arange(-1, 1.1, 0.5)  # define where to put major ticks on y-axis
stat_annotation_offset = -0.15  # vertical offset of statistical annotation
ylabel = None  # which label to put on y-axis
title = None  # title of plot
box_pairs = [('AR1to1d_fs', 'AR1to1d_hs')]  # which groups to perform statistical test on

# make plots
make_box_and_swarmplots_with_test(x, y, df_doublet, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title,
                                  colors)

# Set up plot parameters for sixth panel
# #######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'ASC'  # variable that goes on the y-axis
ax = axes[1, 2]  # define on which axis the plot goes
colors = [colors_parent[2], colors_parent[2]]  # defines colors
stat_annotation_offset = 0.15  # vertical offset of statistical annotation
ylabel = None  # which label to put on y-axis
title = None  # title of plot
ylabeloffset = -1
box_pairs = [('AR1to1s_fs', 'AR1to1s_hs')]  # which groups to perform statistical test on

# make plots
make_box_and_swarmplots_with_test(x, y, df_singlet, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title,
                                  colors)

plt.savefig(figfolder + 'B.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'B.svg', dpi=300, bbox_inches="tight")
plt.show()
# %% filter data to make sure that the baselines are stable
def filter_data_main(data, title):
    # concatenate data on which it will be determined which cells will be filtered
    filterdata = data["shape_data"]["relcell_width_center"][0:20, :]

    # maximal allowed slope for linear fit of baseline
    threshold = 0.001
    baselinefilter = create_filter(filterdata, threshold)

    # remove cells with unstable baselines
    data["TFM_data"] = apply_filter(data["TFM_data"], baselinefilter)
    data["MSM_data"] = apply_filter(data["MSM_data"], baselinefilter)
    data["shape_data"] = apply_filter(data["shape_data"], baselinefilter)

    new_N = np.sum(baselinefilter)
    print(title + ": " + str(baselinefilter.shape[0] - new_N) + " cells were filtered out")

    return data


AR1to1d_fullstim_long = filter_data_main(AR1to1d_fullstim_long, "AR1to1d_fullstim_long")
AR1to1d_fullstim_short = filter_data_main(AR1to1d_fullstim_short, "AR1to1d_fullstim_short")
AR1to1d_halfstim = filter_data_main(AR1to1d_halfstim, "AR1to1d_halfstim")

AR1to1s_fullstim_long = filter_data_main(AR1to1s_fullstim_long, "AR1to1s_fullstim_long")
AR1to1s_fullstim_short = filter_data_main(AR1to1s_fullstim_short, "AR1to1s_fullstim_short")
AR1to1s_halfstim = filter_data_main(AR1to1s_halfstim, "AR1to1s_halfstim")

# %% prepare dataframe for boxplots
# initialize empty dictionaries
concatenated_data_fsl = {}
concatenated_data_fss = {}
concatenated_data_doublet = {}
concatenated_data_singlet = {}

# loop over all keys
for key1 in AR1to1d_fullstim_long:  # keys are the same for all dictionaries so I'm just taking one example here
    for key2 in AR1to1d_fullstim_long[key1]:
        if AR1to1d_fullstim_long[key1][key2].ndim == 1:  # only 1D data can be stored in the data frame

            concatenated_data_doublet[key2] = np.concatenate(
                (AR1to1d_fullstim_long[key1][key2], AR1to1d_fullstim_short[key1][key2]))
            concatenated_data_singlet[key2] = np.concatenate(
                (AR1to1s_fullstim_long[key1][key2], AR1to1s_fullstim_short[key1][key2]))
key1 = "TFM_data"
key2 = "Es_baseline"
# get number of elements for both condition
n_d_fullstim_long = AR1to1d_fullstim_long[key1][key2].shape[0]
n_d_fullstim_short = AR1to1d_fullstim_short[key1][key2].shape[0]
n_s_fullstim_long = AR1to1s_fullstim_long[key1][key2].shape[0]
n_s_fullstim_short = AR1to1s_fullstim_short[key1][key2].shape[0]

# create a list of keys with the same dimensions as the data
keys1to1d_fsl = ['AR1to1d_fsl' for i in range(n_d_fullstim_long)]
keys1to1s_fsl = ['AR1to1s_fsl' for i in range(n_s_fullstim_long)]
keys1to1d_fss = ['AR1to1d_fss' for i in range(n_d_fullstim_short)]
keys1to1s_fss = ['AR1to1s_fss' for i in range(n_s_fullstim_short)]

keys_fsl = np.concatenate((keys1to1d_fsl, keys1to1s_fsl))
keys_fss = np.concatenate((keys1to1d_fss, keys1to1s_fss))
keys_doublet = np.concatenate((keys1to1d_fsl, keys1to1d_fss))
keys_singlet = np.concatenate((keys1to1s_fsl, keys1to1s_fss))

# add keys to dictionary with concatenated data
concatenated_data_fsl['keys'] = keys_fsl
concatenated_data_fss['keys'] = keys_fss
concatenated_data_doublet['keys'] = keys_doublet
concatenated_data_singlet['keys'] = keys_singlet

# Creates DataFrame
df_doublet_afterfilter = pd.DataFrame(concatenated_data_doublet)
df_singlet_afterfilter = pd.DataFrame(concatenated_data_singlet)

# %% plot figure SA, cell width in the center over time

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.arange(60)
x = x[::2]  # downsample data for nicer plotting
ymin = -0.06
ymax = 0.02
xticks = np.arange(0, 61, 20)  # define where the major ticks are gonna be
yticks = np.arange(ymin, ymax + 0.001, 0.02)
xlabel = 'time [min]'
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(5, 3))  # create figure and axes
plt.subplots_adjust(wspace=0.4, hspace=0.35)  # adjust space in between plots
# ******************************************************************************************************************************************


# Set up plot parameters for first panel
#######################################################################################################
ax = axes[0, 0]
color = colors_parent[1]
ylabel = 'doublet'
title = 'long activation'
y = AR1to1d_fullstim_long["shape_data"]["relcell_width_center"]
y = y[::2, :]
x = np.arange(60)
x = x[::2]  # downsample data for nicer plotting
# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color)


# Set up plot parameters for second panel
#######################################################################################################
ax = axes[0, 1]
color = colors_parent[1]
ylabel = None
title = 'short activation'
y = AR1to1d_fullstim_short["shape_data"]["relcell_width_center"]
y = y[::2, :]
x = np.arange(50)
x = x[::2]  # downsample data for nicer plotting
# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False, xmax=60)
for i in np.arange(3):
    ax.axline((20 + i, ymin), (20 + i, ymax), linewidth=0.1, color="cyan")


# Set up plot parameters for third panel
#######################################################################################################
ax = axes[1, 0]
color = colors_parent[2]
ylabel = 'singlet'
title = None
y = AR1to1s_fullstim_long["shape_data"]["relcell_width_center"]
y = y[::2, :]
x = np.arange(60)
x = x[::2]  # downsample data for nicer plotting
# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color)


# Set up plot parameters for fourth panel
#######################################################################################################
ax = axes[1, 1]
color = colors_parent[2]
ylabel = None
y = AR1to1s_fullstim_short["shape_data"]["relcell_width_center"]

y = y[::2, :]
x = np.arange(50)
x = x[::2]  # downsample data for nicer plotting

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False, xmax=60)
for i in np.arange(3):
    ax.axline((20 + i, ymin), (20 + i, ymax), linewidth=0.1, color="cyan")


# Set up plot parameters for fifth panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'relcell_width_center_end'  # variable that goes on the y-axis
ax = axes[0, 2]  # define on which axis the plot goes
colors = [colors_parent[1], colors_parent[1]]  # defines colors
ymin = -0.1  # minimum value on y-axis
ymax = 0.1  # maximum value on y-axis
yticks = np.arange(-0.1, 0.11, 0.1)  # define where to put major ticks on y-axis
xticklabels = ['long \n act.', 'short \n act.']  # which labels to put on x-axis
stat_annotation_offset = 0  # vertical offset of statistical annotation
ylabel = None  # which label to put on y-axis
title = None  # title of plot
box_pairs = [('AR1to1d_fsl', 'AR1to1d_fss')]  # which groups to perform statistical test on

# make plots
make_box_and_swarmplots_with_test(x, y, df_doublet_afterfilter, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels,
                                  ylabel, title, colors)

# Set up plot parameters for sixth panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'relcell_width_center_end'  # variable that goes on the y-axis
ax = axes[1, 2]  # define on which axis the plot goes
colors = [colors_parent[2], colors_parent[2]]  # defines colors
ymin = -0.1  # minimum value on y-axis
ymax = 0.1  # maximum value on y-axis
yticks = np.arange(-0.1, 0.11, 0.1)  # define where to put major ticks on y-axis
xticklabels = ['long \n act.', 'short \n act.']  # which labels to put on x-axis
stat_annotation_offset = 0  # vertical offset of statistical annotation
ylabel = None  # which label to put on y-axis
title = None  # title of plot
box_pairs = [('AR1to1s_fsl', 'AR1to1s_fss')]  # which groups to perform statistical test on

# make plots
make_box_and_swarmplots_with_test(x, y, df_singlet_afterfilter, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels,
                                  ylabel, title, colors)

# write title for panels 1 to 4
plt.text(-5.2, 0.415, 'Cell width at x=0 $\mathrm{\mu}$m', fontsize=10)
# write title for panels 5 to 6
plt.text(-1.15, 0.378, 'Cell width at x=0 $\mathrm{\mu}$m \n        after recovery', fontsize=10)

plt.savefig(figfolder + 'SA.png', dpi=300, bbox_inches="tight")
plt.show()
