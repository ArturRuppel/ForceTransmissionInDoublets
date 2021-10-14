# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 21:56:01 2021

@author: Artur Ruppel

"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from plot_and_filter_functions import *
from scipy.interpolate import interp1d

mpl.rcParams['font.size'] = 8


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


# %% load data for plotting
folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"
AR1to1d_fullstim_long = pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_long.dat", "rb"))
AR1to1d_fullstim_short = pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_short.dat", "rb"))
AR1to1d_halfstim = pickle.load(open(folder + "analysed_data/AR1to1d_halfstim.dat", "rb"))

AR1to1s_fullstim_long = pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_long.dat", "rb"))
AR1to1s_fullstim_short = pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_short.dat", "rb"))
AR1to1s_halfstim = pickle.load(open(folder + "analysed_data/AR1to1s_halfstim.dat", "rb"))

doublet_simulation = pickle.load(open(folder + "_contour_simulations/CM_doublet_simulation.dat", "rb"))
singlet_simulation = pickle.load(open(folder + "_contour_simulations/CM_singlet_simulation.dat", "rb"))

doublet_FEM_simulation = pickle.load(open(folder + "_FEM_simulations/FEM_doublets.dat", "rb"))
singlet_FEM_simulation = pickle.load(open(folder + "_FEM_simulations/FEM_singlets.dat", "rb"))

feedbacks = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# define some colors for the plots
colors_parent = ['#026473', '#E3CC69', '#77C8A6', '#D96248']

figfolder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/_Figure4/"
if not os.path.exists(figfolder):
    os.mkdir(figfolder)
# %% filter data to make sure that the baselines are stable
def filter_data_main(data, threshold, title):
    # concatenate data on which it will be determined which cells will be filtered
    filterdata = np.stack(
        (data["TFM_data"]["relEs"][0:20, :], data["MSM_data"]["relsigma_xx"][0:20, :], data["MSM_data"]["relsigma_yy"][0:20, :],
         data["MSM_data"]["relsigma_xx_left"][0:20, :], data["MSM_data"]["relsigma_xx_right"][0:20, :],
         data["MSM_data"]["relsigma_yy_left"][0:20, :], data["MSM_data"]["relsigma_yy_right"][0:20, :]))

    # move axis of variable to the last position for consistency
    filterdata = np.moveaxis(filterdata, 0, -1)

    # maximal allowed slope for linear fit of baseline

    baselinefilter = create_baseline_filter(filterdata, threshold)

    # remove cells with unstable baselines
    data["TFM_data"] = apply_filter(data["TFM_data"], baselinefilter)
    data["MSM_data"] = apply_filter(data["MSM_data"], baselinefilter)
    data["shape_data"] = apply_filter(data["shape_data"], baselinefilter)

    new_N = np.sum(baselinefilter)
    print(title + ": " + str(baselinefilter.shape[0] - new_N) + " cells were filtered out")

    return data

threshold = 0.005
# threshold = 0.05
AR1to1d_fullstim_long = filter_data_main(AR1to1d_fullstim_long, threshold, "AR1to1d_fullstim_long")
AR1to1d_fullstim_short = filter_data_main(AR1to1d_fullstim_short, threshold, "AR1to1d_fullstim_short")
AR1to1d_halfstim = filter_data_main(AR1to1d_halfstim, threshold, "AR1to1d_halfstim")

AR1to1s_fullstim_long = filter_data_main(AR1to1s_fullstim_long, threshold, "AR1to1s_fullstim_long")
AR1to1s_fullstim_short = filter_data_main(AR1to1s_fullstim_short, threshold, "AR1to1s_fullstim_short")
AR1to1s_halfstim = filter_data_main(AR1to1s_halfstim, threshold, "AR1to1s_halfstim")
# %% plot figure X, lp
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 2.5))  # create figure and axes
color = colors_parent[1]
c = 0
# for key in doublet_FEM_simulation:
#     c += 1
#     y_sim = doublet_FEM_simulation[key]["sigma_normal_x_profile_increase"] * 1e3  # convert to nN
#     x_sim = np.linspace(-22.5, 22.5, y_sim.shape[0])
#     ax.plot(x_sim, y_sim, color=adjust_lightness(color, c / 10), alpha=0.5, linewidth=0.7)
#     plt.axvline(x=-10, ymin=0.0, ymax=1)
#     plt.axvline(x=-22.5 + 5, ymin=0.0, ymax=1, color="black")
#     plt.axvline(x=22.5 - 5, ymin=0.0, ymax=1, color="black")

x = np.linspace(-40, 40, 92)
x = x[::2]  # downsample data for nicer plotting
xticks = np.arange(-40, 40.1, 5)  # define where the major ticks are gonna be
xlabel = 'position [µm]'
ymin = -0.1
ymax = 1.5
yticks = np.arange(ymin, ymax + 0.001, 0.1)
ylabel = None
title = '$\mathrm{\Delta \sigma _{avg. normal}(x)}$ [mN/m]'
y = np.sqrt(AR1to1d_halfstim["TFM_data"]["Dx"] ** 2 + AR1to1d_halfstim["TFM_data"]["Dy"] ** 2)  # convert to nN
y = np.nanmean(y[:, :, 0:20, :], axis=(1, 2))
y = y / max(np.nanmean(y, axis=1))
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False)

plt.show()
fig.savefig(figfolder + 'X.png', dpi=300, bbox_inches="tight")
# %% plot figure X, feedback of FEM
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(1.5, 1.5))  # create figure and axes
color = colors_parent[1]
c = 0
for key in doublet_FEM_simulation:
    c += 1
    y_sim = doublet_FEM_simulation[key]["sigma_normal_x_profile_increase"] * 1e3  # convert to nN
    # y_sim = np.nancumsum(y_sim) / y_sim.shape[0]
    x_sim = np.linspace(-22.5, 22.5, y_sim.shape[0])
    ax.plot(x_sim, y_sim, color=adjust_lightness(color, c / 10), alpha=0.5, linewidth=0.7)
    # plt.axvline(x=-10, ymin=0.0, ymax=1)
    # plt.axvline(x=-22.5 + 5, ymin=0.0, ymax=1, color="black")
    # plt.axvline(x=22.5 - 5, ymin=0.0, ymax=1, color="black")

x = np.linspace(-40, 40, 92)
x = x[::2]  # downsample data for nicer plotting
xticks = np.arange(-40, 40.1, 20)  # define where the major ticks are gonna be
xlabel = 'position [µm]'
ymin = -0.1
ymax = 0.51
# ymax = 20
yticks = np.arange(ymin, ymax + 0.001, 0.1)
ylabel = None
title = '$\mathrm{\Delta \sigma _{avg. normal}(x)}$ [mN/m]'
y = AR1to1d_halfstim["MSM_data"]["sigma_avg_normal_x_profile_increase"] * 1e3  # convert to nN

y = y[::2, :]
# y = np.nancumsum(y, axis=0) / y.shape[0]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False)
# plt.plot(x, y, '*', markersize=4)

# add line at y=0 for visualisation
ax.plot([x[0], x[-1]], [0, 0], linewidth=0.5, linestyle=':', color='grey')
plt.show()
fig.savefig(figfolder + 'X.svg', dpi=300, bbox_inches="tight")
# %% plot figure X, feedback of FEM singlets
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(1.5, 1.5))  # create figure and axes
color = colors_parent[2]
c = 0
for key in singlet_FEM_simulation:
    c += 1
    y_sim = singlet_FEM_simulation[key]["sigma_normal_x_profile_increase"] * 1e3  # convert to nN
    # y_sim = np.nancumsum(y_sim) / y_sim.shape[0]
    x_sim = np.linspace(-22.5, 22.5, y_sim.shape[0])
    ax.plot(x_sim, y_sim, color=adjust_lightness(color, c / 10), alpha=0.5, linewidth=0.7)
    # plt.axvline(x=-10, ymin=0.0, ymax=1)
    # plt.axvline(x=-22.5 + 5, ymin=0.0, ymax=1, color="black")
    # plt.axvline(x=22.5 - 5, ymin=0.0, ymax=1, color="black")

x = np.linspace(-40, 40, 92)
x = x[::2]  # downsample data for nicer plotting
xticks = np.arange(-40, 40.1, 20)  # define where the major ticks are gonna be
xlabel = 'position [µm]'
ymin = -0.1
ymax = 0.51
# ymax = 20
yticks = np.arange(ymin, ymax + 0.001, 0.1)
ylabel = None
title = '$\mathrm{\Delta \sigma _{avg. normal}(x)}$ [mN/m]'
y = AR1to1s_halfstim["MSM_data"]["sigma_avg_normal_x_profile_increase"] * 1e3  # convert to nN

y = y[::2, :]
# y = np.nancumsum(y, axis=0) / y.shape[0]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False)
# plt.plot(x, y, '*', markersize=4)

# add line at y=0 for visualisation
ax.plot([x[0], x[-1]], [0, 0], linewidth=0.5, linestyle=':', color='grey')
plt.show()
fig.savefig(figfolder + 'X1.svg', dpi=300, bbox_inches="tight")

# %% plot figure 3X1, normal-stress maps difference simulation

# prepare data first
key = "feedback0.1"
sigma_avg_normal_doublet_sim = (doublet_FEM_simulation[key]["sigma_avg_norm"][:, :, 32] - \
                                doublet_FEM_simulation[key]["sigma_avg_norm"][:, :, 20]) * 1e3 # convert to mN/m

# sigma_avg_normal_doublet_sim = (doublet_FEM_simulation[key]["sigma_avg_norm"][:, :, 32]) * 1e2 # convert to mN/m
# set up plot parameters
# ******************************************************************************************************************************************
pixelsize = 0.864  # in µm
pmin = -1
pmax = 1  # in mN/m
axtitle = 'mN/m'  # unit of colorbar
suptitle = '$\mathrm{\Delta \sigma _{avg. normal}(x,y)}$'  # title of plot
x_end = np.shape(sigma_avg_normal_doublet_sim)[1]  # create x- and y-axis for plotting maps
y_end = np.shape(sigma_avg_normal_doublet_sim)[0]
extent = [0, x_end * pixelsize, 0, y_end * pixelsize]
xq, yq = np.meshgrid(np.linspace(0, extent[1], x_end), np.linspace(0, extent[3], y_end))  # create mesh for vectorplot

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(1.7, 1.7))  # create figure and axes
plt.subplots_adjust(wspace=0.02, hspace=-0.06)  # adjust space in between plots
# ******************************************************************************************************************************************


im = ax.imshow(sigma_avg_normal_doublet_sim, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent,
                    vmin=pmin, vmax=pmax, aspect='auto')


# remove axes
ax.axis('off')
aspectratio = 1.0
ratio_default = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
ax.set_aspect(ratio_default * aspectratio)

# add colorbar
cbar = fig.colorbar(im, ax=ax)
cbar.ax.set_title(axtitle)

# add title
plt.suptitle(suptitle, y=0.94, x=0.44)


# save figure
fig.savefig(figfolder + 'X2.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'X2.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure A, contour strain measurement
# prepare data first
pixelsize = 0.864  # in µm
initial_pixelsize = 0.108  # in µm
# concatenate TFM maps from different experiments and calculate average maps over first 20 frames and all cells to get average maps
doublet_example = 2

# get data for one example
actin_image_path = folder + "AR1to1_doublets_full_stim_long/actin_images/cell" + str(doublet_example) + "frame32.png"
actin_image = rgb2gray(mpimg.imread(actin_image_path))

# tracking data
x_tracking_top_bs = AR1to1d_fullstim_long["shape_data"]["Xtop"][:, 20, doublet_example] * initial_pixelsize
y_tracking_top_bs = AR1to1d_fullstim_long["shape_data"]["Ytop"][:, 20, doublet_example] * initial_pixelsize
x_tracking_bottom_bs = AR1to1d_fullstim_long["shape_data"]["Xbottom"][:, 20, doublet_example] * initial_pixelsize
y_tracking_bottom_bs = AR1to1d_fullstim_long["shape_data"]["Ybottom"][:, 20, doublet_example] * initial_pixelsize

x_tracking_top_as = AR1to1d_fullstim_long["shape_data"]["Xtop"][:, 30, doublet_example] * initial_pixelsize
y_tracking_top_as = AR1to1d_fullstim_long["shape_data"]["Ytop"][:, 30, doublet_example] * initial_pixelsize
x_tracking_bottom_as = AR1to1d_fullstim_long["shape_data"]["Xbottom"][:, 30, doublet_example] * initial_pixelsize
y_tracking_bottom_as = AR1to1d_fullstim_long["shape_data"]["Ybottom"][:, 30, doublet_example] * initial_pixelsize

# crop force maps and actin images
crop_start = 14
crop_end = 78

# actin_image_bs_crop = actin_image_bs[crop_start * 8:crop_end * 8, crop_start * 8:crop_end * 8]
actin_image_crop = actin_image[crop_start * 8:crop_end * 8, crop_start * 8:crop_end * 8]

# remove 0 values from tracking data
x_tracking_top_bs = x_tracking_top_bs[x_tracking_top_bs != 0]
y_tracking_top_bs = y_tracking_top_bs[y_tracking_top_bs != 0]
x_tracking_bottom_bs = x_tracking_bottom_bs[x_tracking_bottom_bs != 0]
y_tracking_bottom_bs = y_tracking_bottom_bs[y_tracking_bottom_bs != 0]

x_tracking_top_as = x_tracking_top_as[x_tracking_top_as != 0]
y_tracking_top_as = y_tracking_top_as[y_tracking_top_as != 0]
x_tracking_bottom_as = x_tracking_bottom_as[x_tracking_bottom_as != 0]
y_tracking_bottom_as = y_tracking_bottom_as[y_tracking_bottom_as != 0]

# set up plot parameters
# ******************************************************************************************************************************************
x_end = crop_end - crop_start
y_end = crop_end - crop_start
extent = [0, x_end * pixelsize, 0, y_end * pixelsize]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 2.5))  # create figure and axes
# ******************************************************************************************************************************************

# tracking data doesn't live in the same coordinate system as the actin image, so I transform it
x_tracking_top_bs = x_tracking_top_bs - (125 - x_end) / 2 * pixelsize
y_tracking_top_bs = -y_tracking_top_bs + extent[1] + (125 - y_end) / 2 * pixelsize
x_tracking_bottom_bs = x_tracking_bottom_bs - (125 - x_end) / 2 * pixelsize
y_tracking_bottom_bs = -y_tracking_bottom_bs + extent[1] + (125 - y_end) / 2 * pixelsize

x_tracking_top_as = x_tracking_top_as - (125 - x_end) / 2 * pixelsize
y_tracking_top_as = -y_tracking_top_as + extent[1] + (125 - y_end) / 2 * pixelsize
x_tracking_bottom_as = x_tracking_bottom_as - (125 - x_end) / 2 * pixelsize
y_tracking_bottom_as = -y_tracking_bottom_as + extent[1] + (125 - y_end) / 2 * pixelsize

# plot actin image
ax.imshow(actin_image_crop, cmap=plt.get_cmap("Greys"), interpolation="bilinear", extent=extent, aspect=(1), alpha=0.5)

# plot tracking data
ax.plot(x_tracking_top_bs[::2], y_tracking_top_bs[::2],
        color=colors_parent[0], marker='o', markerfacecolor='none', markersize=3, markeredgewidth=0.75, linestyle='none', alpha=0.3)
ax.plot(x_tracking_bottom_bs[::2], y_tracking_bottom_bs[::2],
        color=colors_parent[0], marker='o', markerfacecolor='none', markersize=3, markeredgewidth=0.75, linestyle='none', alpha=0.3)

ax.plot(x_tracking_top_as[::2], y_tracking_top_as[::2],
        color=colors_parent[0], marker='o', markerfacecolor='none', markersize=3, markeredgewidth=0.75, linestyle='none', alpha=1)
ax.plot(x_tracking_bottom_as[::2], y_tracking_bottom_as[::2],
        color=colors_parent[0], marker='o', markerfacecolor='none', markersize=3, markeredgewidth=0.75, linestyle='none', alpha=1)

ax.set_xlim([-0.1 * extent[1], 1.1 * extent[1]])
ax.set_ylim([-0.1 * extent[3], 1.1 * extent[3]])

ax.axis('off')

fig.savefig(figfolder + 'A.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'A.svg', dpi=300, bbox_inches="tight")
plt.show()



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

# %% plot figure D, contour strain after photoactivation

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
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(3.2, 3))  # create figure and axes
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

for fb in feedbacks:
    y_sim = doublet_simulation["EA300"]["FB" + str(fb)]["epsilon_yy"]
    x_sim = np.linspace(-17.5, 17.5, y_sim.shape[0])
    ax.plot(x_sim, y_sim, color=adjust_lightness(color, fb), alpha=0.5, linewidth=0.7)

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

for fb in feedbacks:
    y_sim = doublet_simulation["EA300"]["FB" + str(fb)]["epsilon_yy"]
    x_sim = np.linspace(-17.5, 17.5, y_sim.shape[0])
    ax.plot(x_sim, y_sim, color=adjust_lightness(color, fb), alpha=0.5, linewidth=0.7)

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False)

# Set up plot parameters for third panel
#######################################################################################################
ax = axes[1, 0]
color = colors_parent[2]
ylabel = 'singlet'
title = None
ymin = -0.03
y = AR1to1s_fullstim_long["shape_data"]["contour_strain"]
y = y[::2, :]

for fb in feedbacks:
    y_sim = singlet_simulation["EA300"]["FB" + str(fb)]["epsilon_yy"]
    x_sim = np.linspace(-17.5, 17.5, y_sim.shape[0])
    ax.plot(x_sim, y_sim, color=adjust_lightness(color, fb), alpha=0.5, linewidth=0.7)

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

for fb in feedbacks:
    y_sim = singlet_simulation["EA300"]["FB" + str(fb)]["epsilon_yy"]
    x_sim = np.linspace(-17.5, 17.5, y_sim.shape[0])
    ax.plot(x_sim, y_sim, color=adjust_lightness(color, fb), alpha=0.5, linewidth=0.7)

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False)

plt.savefig(figfolder + 'D.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'D.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure E, contour strain after photoactivation

# make some calculations on the simulated data first
epsilon_asymmetry_coefficient_all = []
# singlet_epsilon_asymmetry_coefficient = []

for fb in feedbacks:
    epsilon = doublet_simulation["EA50"]["FB" + str(fb)]["epsilon_yy"]
    rel_epsilon = epsilon / np.nansum(np.abs(epsilon), axis=0)
    epsilon_asymmetry_curve = (rel_epsilon - np.flipud(rel_epsilon))
    epsilon_asymmetry_coefficient = -np.nansum(epsilon_asymmetry_curve[0:int(epsilon_asymmetry_curve.shape[0] / 2)], axis=0)
    epsilon_asymmetry_coefficient_all.append(epsilon_asymmetry_coefficient)


# for fb in feedbacks:
#     epsilon = singlet_simulation["EA50"]["FB" + str(fb)]["epsilon_yy"]
#     # rel_epsilon = epsilon / np.nansum(epsilon)
#     # epsilon_asymmetry_curve = (rel_epsilon - np.flipud(rel_epsilon)) / 2
#     # epsilon_asymmetry_coefficient = np.nansum(np.abs(epsilon_asymmetry_curve), axis=0)
#     # singlet_epsilon_asymmetry_coefficient.append(epsilon_asymmetry_coefficient)
#     rel_epsilon = epsilon / np.nansum(np.abs(epsilon), axis=0)
#     epsilon_asymmetry_curve = (rel_epsilon - np.flipud(rel_epsilon))
#     epsilon_asymmetry_coefficient = -np.nansum(epsilon_asymmetry_curve[0:int(epsilon_asymmetry_curve.shape[0] / 2)], axis=0)
#     singlet_epsilon_asymmetry_coefficient.append(epsilon_asymmetry_coefficient)

# def find_nearest(array, value):
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return array[idx]

def find_x_position_of_point_on_array(x_list, y_list, y_point):
    f = interp1d(y_list, x_list, kind='cubic')
    return f(y_point)


stats_doublet = calculate_median_and_CI(df_doublet, 'ASC')
stats_singlet = calculate_median_and_CI(df_singlet, 'ASC')

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))

ax.plot(feedbacks, epsilon_asymmetry_coefficient_all, color='grey')
# ax.plot(feedbacks, singlet_epsilon_asymmetry_coefficient, color=colors_parent[2])

# add data points
color = colors_parent[1]
y_median = stats_doublet["median"]["AR1to1d_hs"]
y_CI = stats_doublet["CI"]["AR1to1d_hs"]
x = find_x_position_of_point_on_array(feedbacks, epsilon_asymmetry_coefficient_all, y_median)
# x = 0.45
# x_CI = find_x_position_of_point_on_line(feedbacks, doublet_epsilon_asymmetry_coefficient, y_median - y_CI) - x
ax.errorbar(x, y_median, yerr=y_CI, mfc='w', color=color, marker='s', ms=4, linewidth=0.5, ls='none',
            markeredgewidth=0.5)

color = colors_parent[2]
y_median = stats_singlet["median"]["AR1to1s_hs"]
y_CI = stats_singlet["CI"]["AR1to1s_hs"]
x = find_x_position_of_point_on_array(feedbacks, epsilon_asymmetry_coefficient_all, y_median)
# x = 0.35
# x_CI = find_x_position_of_point_on_line(feedbacks, singlet_epsilon_asymmetry_coefficient, y_median - y_CI) - x
ax.errorbar(x, y_median, yerr=y_CI, mfc='w', color=color, marker='s', ms=4, linewidth=0.5, ls='none',
            markeredgewidth=0.5)

# provide info on tick parameters
ax.minorticks_on()
ax.tick_params(direction='in', which='minor', length=3, bottom=True, top=False, left=True, right=True)
ax.tick_params(direction='in', which='major', length=6, bottom=True, top=False, left=True, right=True)

plt.xlabel('feedback')
plt.ylabel('ASC')
plt.title('Asymmetry coefficient')
plt.savefig(figfolder + 'E.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'E.svg', dpi=300, bbox_inches="tight")
plt.show()
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
