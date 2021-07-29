# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 21:56:01 2021

@author: Artur Ruppel

"""
import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statannot
from scipy.stats import pearsonr
import matplotlib.image as mpimg

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


# %% plot figure 3C, Relative strain energy over time
# set up global plot parameters
# ******************************************************************************************************************************************
x = np.arange(60)
x = x[::2]  # downsample data for nicer plotting
ymin = -0.05
ymax = 0.05
xticks = np.arange(0, 61, 20)  # define where the major ticks are gonna be
yticks = np.arange(ymin, ymax + 0.001, 0.02)
xlabel = 'time [min]'
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
plt.subplots_adjust(wspace=0.35, hspace=0.35)  # adjust space in between plots


# ******************************************************************************************************************************************

def plot_one_value_over_time(x, y, xticks, yticks, xlabeloffset, ylabeloffset, titleoffset, optolinewidth,
                             ymin, ymax, xlabel, ylabel, title, ax, colors):
    y_mean = np.nanmean(y, axis=1)
    y_std = np.nanstd(y, axis=1)
    y_sem = y_std / np.sqrt(np.shape(y)[1])
    # create box- and swarmplots
    ax.errorbar(x, y_mean, yerr=y_sem, mfc='w', color=colors, marker='o', ms=2, linewidth=0.5, ls='none',
                markeredgewidth=0.5)

    # set labels
    ax.set_xlabel(xlabel=xlabel, labelpad=xlabeloffset)
    ax.set_ylabel(ylabel=ylabel, labelpad=ylabeloffset)
    ax.set_title(label=title, pad=titleoffset)

    # add anotations for opto pulses
    for i in np.arange(10):
        ax.axline((20 + i, ymin), (20 + i, ymax), linewidth=optolinewidth, color="cyan")

    # set ticks
    ax.xaxis.set_ticks(xticks)
    ax.yaxis.set_ticks(yticks)

    # provide info on tick parameters
    ax.minorticks_on()
    ax.tick_params(direction='in', which='minor', length=3, bottom=False, top=False, left=True, right=True)
    ax.tick_params(direction='in', which='major', length=6, bottom=False, top=False, left=True, right=True)

    # set limits
    ax.set_ylim(ymin=ymin, ymax=ymax)


def make_two_box_and_swarmplots(linewidth_bp, width_bp, dotsize, linewidth_sw, alpha_sw, alpha_bp, ylabeloffset, titleoffset, test,
                                x, y, df, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title, colors):
    sns.set_palette(sns.color_palette(colors))  # sets colors
    # create box- and swarmplots
    sns.swarmplot(x=x, y=y, data=df, ax=ax, alpha=alpha_sw, linewidth=linewidth_sw, zorder=0, size=dotsize)
    bp = sns.boxplot(x=x, y=y, data=df, ax=ax, linewidth=linewidth_bp, notch=True, showfliers=False, width=width_bp, showmeans=True,
                     meanprops={"marker": "o",
                                "markerfacecolor": "white",
                                "markeredgecolor": "black",
                                "markersize": "3", "markeredgewidth": "0.5"})

    statannot.add_stat_annotation(bp, data=df, x=x, y=y, box_pairs=box_pairs,
                                  line_offset_to_box=stat_annotation_offset, test=test, text_format='star', loc='inside', verbose=2)

    # make boxplots transparent
    for patch in bp.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, alpha_bp))

    plt.setp(bp.artists, edgecolor='k')
    plt.setp(bp.lines, color='k')

    # set labels
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel(xlabel=None)
    ax.set_ylabel(ylabel=ylabel, labelpad=ylabeloffset)
    ax.set_title(label=title, pad=titleoffset)

    # set yaxis ticks
    ax.yaxis.set_ticks(yticks)

    # provide info on tick parameters
    ax.minorticks_on()
    ax.tick_params(direction='in', which='minor', length=3, bottom=False, top=False, left=True, right=True)
    ax.tick_params(direction='in', which='major', length=6, bottom=False, top=False, left=True, right=True)

    # set limits
    ax.set_ylim(ymin=ymin)
    ax.set_ylim(ymax=ymax)


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

# # Set up plot parameters for second panel
# #######################################################################################################
# ax = axes[0, 1]
# color = colors_parent[1]
# ylabel = None
# title = 'local activation'
# y = AR1to1d_halfstim["TFM_data"]["relEs"]
# y = y[::2, :]
#
# # make plots
# plot_one_value_over_time(x, y, xticks, yticks, xlabeloffset, ylabeloffset, titleoffset,
#                          optolinewidth, ymin, ymax, xlabel, ylabel, title, ax, color)
#
# # Set up plot parameters for third panel
# #######################################################################################################
# ax = axes[1, 0]
# color = colors_parent[2]
# ylabel = 'singlet'
# title = None
# y = AR1to1s_fullstim_long["TFM_data"]["relEs"]
# y = y[::2, :]
#
# # make plots
# plot_one_value_over_time(x, y, xticks, yticks, xlabeloffset, ylabeloffset, titleoffset,
#                          optolinewidth, ymin, ymax, xlabel, ylabel, title, ax, color)
#
# # Set up plot parameters for fourth panel
# #######################################################################################################
# ax = axes[1, 1]
# color = colors_parent[2]
# ylabel = None
# title = None
# y = AR1to1s_halfstim["TFM_data"]["relEs"]
# y = y[::2, :]
#
# # make plots
# plot_one_value_over_time(x, y, xticks, yticks, xlabeloffset, ylabeloffset, titleoffset,
#                          optolinewidth, ymin, ymax, xlabel, ylabel, title, ax, color)

# # Set up plot parameters for fifth panel
# #######################################################################################################
# x = 'keys'  # variable by which to group the data
# y = 'REI'  # variable that goes on the y-axis
# ax = axes[0, 2]  # define on which axis the plot goes
# colors = [colors_parent[1], colors_parent[1]]  # defines colors
# ymin = -0.2  # minimum value on y-axis
# ymax = 0.8  # maximum value on y-axis
# yticks = np.arange(-0.2, 0.81, 0.2)  # define where to put major ticks on y-axis
# stat_annotation_offset = 0.1  # vertical offset of statistical annotation
# ylabel = None  # which label to put on y-axis
# title = 'REI'  # title of plot
# ylabeloffset = -1
# box_pairs = [('AR1to1d_fs', 'AR1to1d_hs')]  # which groups to perform statistical test on
#
# # make plots
# make_two_box_and_swarmplots(linewidth_bp, width_bp, dotsize, linewidth_sw, alpha_sw, alpha_bp, ylabeloffset, titleoffset, test,
#                             x, y, df_doublet, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title, colors)
#
# # Set up plot parameters for sixth panel
# #######################################################################################################
# x = 'keys'  # variable by which to group the data
# y = 'REI'  # variable that goes on the y-axis
# ax = axes[1, 2]  # define on which axis the plot goes
# colors = [colors_parent[2], colors_parent[2]]  # defines colors
# ymin = -0.2  # minimum value on y-axis
# ymax = 0.8  # maximum value on y-axis
# yticks = np.arange(-0.2, 0.81, 0.2)  # define where to put major ticks on y-axis
# stat_annotation_offset = 0.3  # vertical offset of statistical annotation
# ylabel = None  # which label to put on y-axis
# title = 'REI'  # title of plot
# ylabeloffset = -1
# box_pairs = [('AR1to1s_fs', 'AR1to1s_hs')]  # which groups to perform statistical test on

# # make plots
# make_two_box_and_swarmplots(linewidth_bp, width_bp, dotsize, linewidth_sw, alpha_sw, alpha_bp, ylabeloffset, titleoffset, test,
#                             x, y, df_singlet, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title, colors)

# # write title for panels 1 to 4
# plt.text(-5.1, 2.55, 'Relative strain energy', fontsize=10)
# # write title for panels 5 to 6
# plt.text(-0.61, 2.4, 'Relative energy \n     increase', fontsize=10)
# # save plot to file
plt.savefig(figfolder + 'C.png', dpi=300, bbox_inches="tight")

plt.show()

