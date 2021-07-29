# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 21:56:01 2021

@author: Artur Ruppel

"""
import os
import pickle
import pandas as pd

from plot_functions import *

# mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 8

# %% load data for plotting
folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"
AR1to1d_fullstim_long = pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_long.dat", "rb"))
AR1to1d_fullstim_short = pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_short.dat", "rb"))
AR1to1d_halfstim = pickle.load(open(folder + "analysed_data/AR1to1d_halfstim.dat", "rb"))

AR1to1s_fullstim_long = pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_long.dat", "rb"))
AR1to1s_fullstim_short = pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_short.dat", "rb"))
AR1to1s_halfstim = pickle.load(open(folder + "analysed_data/AR1to1s_halfstim.dat", "rb"))

# load contour model analysis data
AR1to1d_fullstim_long_CM = pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_long_CM.dat", "rb"))
AR1to1d_fullstim_short_CM = pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_short_CM.dat", "rb"))
AR1to1d_halfstim_CM = pickle.load(open(folder + "analysed_data/AR1to1d_halfstim_CM.dat", "rb"))

AR1to1s_fullstim_long_CM = pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_long_CM.dat", "rb"))
AR1to1s_fullstim_short_CM = pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_short_CM.dat", "rb"))
AR1to1s_halfstim_CM = pickle.load(open(folder + "analysed_data/AR1to1s_halfstim_CM.dat", "rb"))

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

            # concatenate values from different experiments
            concatenated_data_fs[key2] = np.concatenate(
                (AR1to1d_fullstim_long[key1][key2], AR1to1s_fullstim_long[key1][key2]))
            concatenated_data_hs[key2] = np.concatenate((AR1to1d_halfstim[key1][key2], AR1to1s_halfstim[key1][key2]))
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

keys_fs = np.concatenate((keys1to1d_fs, keys1to1s_fs))
keys_hs = np.concatenate((keys1to1d_hs, keys1to1s_hs))
keys_doublet = np.concatenate((keys1to1d_fs, keys1to1d_hs))
keys_singlet = np.concatenate((keys1to1s_fs, keys1to1s_hs))

# add keys to dictionary with concatenated data
concatenated_data_fs['keys'] = keys_fs
concatenated_data_hs['keys'] = keys_hs
concatenated_data_doublet['keys'] = keys_doublet
concatenated_data_singlet['keys'] = keys_singlet

# Creates DataFrame
df_fs = pd.DataFrame(concatenated_data_fs)
df_hs = pd.DataFrame(concatenated_data_hs)
df_doublet = pd.DataFrame(concatenated_data_doublet)
df_singlet = pd.DataFrame(concatenated_data_singlet)

df_d_fs = df_fs[df_fs["keys"] == "AR1to1d_fs"]
df_d_hs = df_hs[df_hs["keys"] == "AR1to1d_hs"]

df_s_fs = df_fs[df_fs["keys"] == "AR1to1s_fs"]
df_s_hs = df_hs[df_hs["keys"] == "AR1to1s_hs"]
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
xlabeloffset = 1  # adjusts distance of xlabel to the plot
ylabeloffset = 1  # adjusts distance of ylabel to the plot
titleoffset = 5  # adjusts distance of title to the plot
optolinewidth = 0.1  # adjusts the linewidth of the annotations that represent the optogenetic activation
linewidth_bp = 0.7  # linewidth of boxplot borders
width_bp = 0.3  # width of boxplots
dotsize = 1.8  # size of datapoints in swarmplot
linewidth_sw = 0.3  # linewidth of boxplot borders
alpha_sw = 1  # transparency of dots in swarmplot
alpha_bp = 0.8  # transparency of boxplots
test = 'Mann-Whitney'  # which statistical test to compare different conditions
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
plot_one_value_over_time(x, y, xticks, yticks, xlabeloffset, ylabeloffset, titleoffset,
                         False, ymin, ymax, xlabel, ylabel, title, ax, color)

# Set up plot parameters for second panel
#######################################################################################################
ax = axes[0, 1]
color = colors_parent[1]
ylabel = None
title = 'local activation'
y = AR1to1d_halfstim["shape_data"]["contour_strain"]
y = y[::2, :]
# make plots
plot_one_value_over_time(x, y, xticks, yticks, xlabeloffset, ylabeloffset, titleoffset,
                         False, ymin, ymax, xlabel, ylabel, title, ax, color)

# Set up plot parameters for third panel
#######################################################################################################
ax = axes[1, 0]
color = colors_parent[1]
ylabel = 'singlet'
title = None
y = AR1to1s_fullstim_long["shape_data"]["contour_strain"]
y = y[::2, :]
# make plots
plot_one_value_over_time(x, y, xticks, yticks, xlabeloffset, ylabeloffset, titleoffset,
                         False, ymin, ymax, xlabel, ylabel, title, ax, color)

# Set up plot parameters for fourth panel
#######################################################################################################
ax = axes[1, 1]
color = colors_parent[2]
ylabel = None
title = None
y = AR1to1s_halfstim["shape_data"]["contour_strain"]
y = y[::2, :]
# make plots
plot_one_value_over_time(x, y, xticks, yticks, xlabeloffset, ylabeloffset, titleoffset,
                         False, ymin, ymax, xlabel, ylabel, title, ax, color)



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
ylabeloffset = -1
box_pairs = [('AR1to1d_fs', 'AR1to1d_hs')]  # which groups to perform statistical test on

# make plots
make_two_box_and_swarmplots(linewidth_bp, width_bp, dotsize, linewidth_sw, alpha_sw, alpha_bp, ylabeloffset, titleoffset, test,
                            x, y, df_doublet, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title, colors)

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
make_two_box_and_swarmplots(linewidth_bp, width_bp, dotsize, linewidth_sw, alpha_sw, alpha_bp, ylabeloffset, titleoffset, test,
                            x, y, df_singlet, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title, colors)

plt.savefig(figfolder + 'B.png', dpi=300, bbox_inches="tight")
plt.show()
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
xlabeloffset = 1  # adjusts distance of xlabel to the plot
ylabeloffset = 1  # adjusts distance of ylabel to the plot
titleoffset = 5  # adjusts distance of title to the plot
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(5, 3))  # create figure and axes
plt.subplots_adjust(wspace=0.35, hspace=0.35)  # adjust space in between plots
# ******************************************************************************************************************************************


# Set up plot parameters for first panel
#######################################################################################################
ax = axes[0, 0]
color = colors_parent[1]
ylabel = 'doublet'
title = 'long activation'
y = AR1to1d_fullstim_long["shape_data"]["relcell_width_center"]

y = y[::2, :]
# make plots
plot_one_value_over_time(x, y, xticks, yticks, xlabeloffset, ylabeloffset, titleoffset,
                         optolinewidth, ymin, ymax, xlabel, ylabel, title, ax, color)

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
plot_one_value_over_time(x, y, xticks, yticks, xlabeloffset, ylabeloffset, titleoffset,
                         optolinewidth, ymin, ymax, xlabel, ylabel, title, ax, color)

# Set up plot parameters for third panel
#######################################################################################################
ax = axes[0, 2]
color = colors_parent[1]
ylabel = None
title = 'half activation'
y = AR1to1d_halfstim["shape_data"]["relcell_width_center"]

y = y[::2, :]
x = np.arange(60)
x = x[::2]  # downsample data for nicer plotting
# make plots
plot_one_value_over_time(x, y, xticks, yticks, xlabeloffset, ylabeloffset, titleoffset,
                         optolinewidth, ymin, ymax, xlabel, ylabel, title, ax, color)

# Set up plot parameters for fourth panel
#######################################################################################################
ax = axes[1, 0]
color = colors_parent[2]
ylabel = 'singlet'
title = None
y = AR1to1s_fullstim_long["shape_data"]["relcell_width_center"]

y = y[::2, :]
# make plots
plot_one_value_over_time(x, y, xticks, yticks, xlabeloffset, ylabeloffset, titleoffset,
                         optolinewidth, ymin, ymax, xlabel, ylabel, title, ax, color)

# Set up plot parameters for fifth panel
#######################################################################################################
ax = axes[1, 1]
color = colors_parent[2]
ylabel = None
y = AR1to1s_fullstim_short["shape_data"]["relcell_width_center"]

y = y[::2, :]
x = np.arange(50)
x = x[::2]  # downsample data for nicer plotting
# make plots
plot_one_value_over_time(x, y, xticks, yticks, xlabeloffset, ylabeloffset, titleoffset,
                         optolinewidth, ymin, ymax, xlabel, ylabel, title, ax, color)

# Set up plot parameters for sixth panel
#######################################################################################################
ax = axes[1, 2]
color = colors_parent[2]
ylabel = None
y = AR1to1s_halfstim["shape_data"]["relcell_width_center"]

y = y[::2, :]
x = np.arange(60)
x = x[::2]  # downsample data for nicer plotting
# make plots
plot_one_value_over_time(x, y, xticks, yticks, xlabeloffset, ylabeloffset, titleoffset,
                         optolinewidth, ymin, ymax, xlabel, ylabel, title, ax, color)

plt.savefig(figfolder + 'SA.png', dpi=300, bbox_inches="tight")

plt.show()

# %% plot figure SA detrend, detrend cell width in the center over time

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.arange(60)
x = x[::2]  # downsample data for nicer plotting
ymin = -0.06
ymax = 0.02
xticks = np.arange(0, 61, 20)  # define where the major ticks are gonna be
yticks = np.arange(ymin, ymax + 0.001, 0.02)
xlabel = 'time [min]'
xlabeloffset = 1  # adjusts distance of xlabel to the plot
ylabeloffset = 1  # adjusts distance of ylabel to the plot
titleoffset = 5  # adjusts distance of title to the plot
optolinewidth = 0.1  # adjusts the linewidth of the annotations that represent the optogenetic activation
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(5, 3))  # create figure and axes
plt.subplots_adjust(wspace=0.35, hspace=0.35)  # adjust space in between plots
# ******************************************************************************************************************************************


# Set up plot parameters for first panel
#######################################################################################################
ax = axes[0, 0]
color = colors_parent[1]
ylabel = 'doublet'
title = 'long activation'
y = AR1to1d_fullstim_long["shape_data"]["relcell_width_center_detrend"]

y = y[::2, :]
# make plots
plot_one_value_over_time(x, y, xticks, yticks, xlabeloffset, ylabeloffset, titleoffset,
                         optolinewidth, ymin, ymax, xlabel, ylabel, title, ax, color)

# Set up plot parameters for second panel
#######################################################################################################
ax = axes[0, 1]
color = colors_parent[1]
ylabel = None
title = 'short activation'
y = AR1to1d_fullstim_short["shape_data"]["relcell_width_center_detrend"]

y = y[::2, :]
x = np.arange(50)
x = x[::2]  # downsample data for nicer plotting
# make plots
plot_one_value_over_time(x, y, xticks, yticks, xlabeloffset, ylabeloffset, titleoffset,
                         optolinewidth, ymin, ymax, xlabel, ylabel, title, ax, color)

# Set up plot parameters for third panel
#######################################################################################################
ax = axes[0, 2]
color = colors_parent[1]
ylabel = None
title = 'half activation'
y = AR1to1d_halfstim["shape_data"]["relcell_width_center_detrend"]

y = y[::2, :]
x = np.arange(60)
x = x[::2]  # downsample data for nicer plotting
# make plots
plot_one_value_over_time(x, y, xticks, yticks, xlabeloffset, ylabeloffset, titleoffset,
                         optolinewidth, ymin, ymax, xlabel, ylabel, title, ax, color)

# Set up plot parameters for fourth panel
#######################################################################################################
ax = axes[1, 0]
color = colors_parent[2]
ylabel = 'singlet'
title = None
y = AR1to1s_fullstim_long["shape_data"]["relcell_width_center_detrend"]

y = y[::2, :]
# make plots
plot_one_value_over_time(x, y, xticks, yticks, xlabeloffset, ylabeloffset, titleoffset,
                         optolinewidth, ymin, ymax, xlabel, ylabel, title, ax, color)

# Set up plot parameters for fifth panel
#######################################################################################################
ax = axes[1, 1]
color = colors_parent[2]
ylabel = None
y = AR1to1s_fullstim_short["shape_data"]["relcell_width_center_detrend"]

y = y[::2, :]
x = np.arange(50)
x = x[::2]  # downsample data for nicer plotting
# make plots
plot_one_value_over_time(x, y, xticks, yticks, xlabeloffset, ylabeloffset, titleoffset,
                         optolinewidth, ymin, ymax, xlabel, ylabel, title, ax, color)

# Set up plot parameters for sixth panel
#######################################################################################################
ax = axes[1, 2]
color = colors_parent[2]
ylabel = None
y = AR1to1s_halfstim["shape_data"]["relcell_width_center_detrend"]

y = y[::2, :]
x = np.arange(60)
x = x[::2]  # downsample data for nicer plotting
# make plots
plot_one_value_over_time(x, y, xticks, yticks, xlabeloffset, ylabeloffset, titleoffset,
                         optolinewidth, ymin, ymax, xlabel, ylabel, title, ax, color)

plt.savefig(figfolder + 'SA_detrend.png', dpi=300, bbox_inches="tight")

plt.show()
