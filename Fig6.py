# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 21:56:01 2021

@author: Artur Ruppel

"""
import os
import pickle
import pandas as pd
from scipy.stats import zscore
from plot_and_filter_functions import *


mpl.rcParams['font.size'] = 8

# %% load data for plotting
folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"
tissues_20micron_full_stim = pickle.load(open(folder + "analysed_data/tissues_20micron_full_stim.dat", "rb"))
tissues_20micron_lefthalf_stim = pickle.load(open(folder + "analysed_data/tissues_20micron_lefthalf_stim.dat", "rb"))
tissues_20micron_tophalf_stim = pickle.load(open(folder + "analysed_data/tissues_20micron_tophalf_stim.dat", "rb"))
tissues_40micron_full_stim = pickle.load(open(folder + "analysed_data/tissues_40micron_full_stim.dat", "rb"))
tissues_40micron_lefthalf_stim = pickle.load(open(folder + "analysed_data/tissues_40micron_lefthalf_stim.dat", "rb"))
tissues_40micron_tophalf_stim = pickle.load(open(folder + "analysed_data/tissues_40micron_tophalf_stim.dat", "rb"))

# define some colors for the plots
colors_parent = ['#026473', '#E3CC69', '#77C8A6', '#D96248']
colors_parent_dark = ['#01353D', '#564910', '#235741', '#A93B23']

figfolder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/_Figure6/"
if not os.path.exists(figfolder):
    os.mkdir(figfolder)
# %% prepare dataframe for boxplots

# initialize empty dictionaries
concatenated_data_20micron_full_stim = {}
concatenated_data_20micron_lefthalf_stim = {}
concatenated_data_20micron_tophalf_stim = {}
concatenated_data_40micron_full_stim = {}
concatenated_data_40micron_lefthalf_stim = {}
concatenated_data_40micron_tophalf_stim = {}
concatenated_data = {}

# loop over all keys
for key1 in tissues_20micron_full_stim:  # keys are the same for all dictionaries so I'm just taking one example here
    for key2 in tissues_20micron_full_stim[key1]:
        if tissues_20micron_full_stim[key1][key2].ndim == 1:  # only 1D data can be stored in the data frame
            # concatenate values from different experiments
            concatenated_data_20micron_full_stim[key2] = tissues_20micron_full_stim[key1][key2]
            concatenated_data_20micron_lefthalf_stim[key2] = tissues_20micron_lefthalf_stim[key1][key2]
            concatenated_data_20micron_tophalf_stim[key2] = tissues_20micron_tophalf_stim[key1][key2]
            concatenated_data_40micron_full_stim[key2] = tissues_40micron_full_stim[key1][key2]
            concatenated_data_40micron_lefthalf_stim[key2] = tissues_40micron_lefthalf_stim[key1][key2]
            concatenated_data_40micron_tophalf_stim[key2] = tissues_40micron_tophalf_stim[key1][key2]

            # concatenated_data[key2] = np.concatenate(
            #     (concatenated_data_20micron_full_stim[key2], concatenated_data_20micron_lefthalf_stim[key2], concatenated_data_20micron_tophalf_stim[key2],
            #      concatenated_data_40micron_full_stim[key2], concatenated_data_40micron_lefthalf_stim[key2], concatenated_data_40micron_tophalf_stim[key2]))
            concatenated_data[key2] = np.concatenate(
                (concatenated_data_20micron_full_stim[key2], concatenated_data_20micron_lefthalf_stim[key2],
                 concatenated_data_40micron_full_stim[key2], concatenated_data_40micron_lefthalf_stim[key2]))
key2 = 'Es_baseline'
# get number of elements for both condition
n_20micron_fs = concatenated_data_20micron_full_stim[key2].shape[0]
n_20micron_ls = concatenated_data_20micron_lefthalf_stim[key2].shape[0]
n_20micron_ts = concatenated_data_20micron_tophalf_stim[key2].shape[0]
n_40micron_fs = concatenated_data_40micron_full_stim[key2].shape[0]
n_40micron_ls = concatenated_data_40micron_lefthalf_stim[key2].shape[0]
n_40micron_ts = concatenated_data_40micron_tophalf_stim[key2].shape[0]

# create a list of keys with the same dimensions as the data
keys20micron_fs = ['20micron_fs' for i in range(n_20micron_fs)]
keys20micron_ls = ['20micron_fs' for i in range(n_20micron_ls)]
# keys20micron_ts = ['20micron_fs' for i in range(n_20micron_ts)]
keys40micron_fs = ['40micron_fs' for i in range(n_40micron_fs)]
keys40micron_ls = ['40micron_fs' for i in range(n_40micron_ls)]
# keys40micron_ts = ['40micron_fs' for i in range(n_40micron_ts)]

# keys = np.concatenate((keys20micron_fs, keys20micron_ls, keys20micron_ts, keys40micron_fs, keys40micron_ls, keys40micron_ts))
keys = np.concatenate((keys20micron_fs, keys20micron_ls, keys40micron_fs, keys40micron_ls))

# add keys to dictionary with concatenated data
concatenated_data['keys'] = keys

# Creates DataFrame
df = pd.DataFrame(concatenated_data)

# convert to more convenient units for plotting
df_plot_units = df  # all units here are in SI units
df_plot_units['Es_baseline'] *= 1e12  # convert to fJ
df_plot_units['sigma_xx_baseline'] *= 1e3  # convert to mN/m
df_plot_units['sigma_yy_baseline'] *= 1e3  # convert to mN/m

# %% plot figure 5A, force maps

# prepare data first

# Calculate average maps over first 20 frames and all cells to get average maps
Tx_20micron_average = np.nanmean(tissues_20micron_full_stim["TFM_data"]["Tx"][:, :, 0:20, :], axis=(2, 3)) * 1e-3  # convert to kPa
Ty_20micron_average = np.nanmean(tissues_20micron_full_stim["TFM_data"]["Ty"][:, :, 0:20, :], axis=(2, 3)) * 1e-3

Tx_40micron_average = np.nanmean(tissues_40micron_full_stim["TFM_data"]["Tx"][:, :, 0:20, :], axis=(2, 3)) * 1e-3
Ty_40micron_average = np.nanmean(tissues_40micron_full_stim["TFM_data"]["Ty"][:, :, 0:20, :], axis=(2, 3)) * 1e-3

# calculate amplitudes
T_20micron_average = np.sqrt(Tx_20micron_average ** 2 + Ty_20micron_average ** 2)
T_40micron_average = np.sqrt(Tx_40micron_average ** 2 + Ty_40micron_average ** 2)


# set up plot parameters
# *****************************************************************************
n = 4  # every nth arrow will be plotted
pixelsize = 1.296  # in µm
pmax = 2  # kPa

# create x- and y-axis for plotting maps
x_end = np.shape(T_20micron_average)[1]
y_end = np.shape(T_20micron_average)[0]
extent = [0, x_end * pixelsize, 0, y_end * pixelsize]

# create mesh for vectorplot
xq, yq = np.meshgrid(np.linspace(0, extent[1], x_end), np.linspace(0, extent[3], y_end))

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(2.5, 3))

im = axes[0].imshow(T_20micron_average, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0,
                    vmax=pmax, aspect='auto')
axes[0].quiver(xq[::n, ::n], yq[::n, ::n], Tx_20micron_average[::n, ::n], Ty_20micron_average[::n, ::n],
               angles='xy', scale=10, units='width', color="r")
# axes[0,0].set_title('n=1', pad=-400, color='r')

axes[1].imshow(T_40micron_average, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0,
               vmax=pmax, aspect='auto')
axes[1].quiver(xq[::n, ::n], yq[::n, ::n], Tx_40micron_average[::n, ::n], Ty_40micron_average[::n, ::n],
               angles='xy', scale=10, units='width', color="r")


# adjust space in between plots
plt.subplots_adjust(wspace=0, hspace=0)

# remove axes
for ax in axes.flat:
    ax.axis('off')
    aspectratio = 1.0
    ratio_default = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.set_aspect(ratio_default * aspectratio)

# add colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist())
cbar.ax.set_title('kPa')

# add title
plt.suptitle('Traction forces', y=0.95, x=0.54)

# add annotations
plt.text(0.3, 0.8, 'n=' + str(n_20micron_fs), transform=plt.figure(1).transFigure, color='w')
plt.text(0.3, 0.42, 'n=' + str(n_40micron_fs), transform=plt.figure(1).transFigure, color='w')

fig.savefig(figfolder + 'A.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'A.svg', dpi=300, bbox_inches="tight")

plt.show()
# %% plot figure 5B, stress maps

# prepare data first

# Calculate average maps over first 20 frames and all cells to get average maps
sigma_xx_20micron_average = np.nanmean(tissues_20micron_full_stim["MSM_data"]["sigma_xx"][:, :, 0:20, :], axis=(2, 3)) * 1e3  # convert to mN/m
sigma_yy_20micron_average = np.nanmean(tissues_20micron_full_stim["MSM_data"]["sigma_yy"][:, :, 0:20, :], axis=(2, 3)) * 1e3

sigma_xx_40micron_average = np.nanmean(tissues_40micron_full_stim["MSM_data"]["sigma_xx"][:, :, 0:20, :], axis=(2, 3)) * 1e3
sigma_yy_40micron_average = np.nanmean(tissues_40micron_full_stim["MSM_data"]["sigma_yy"][:, :, 0:20, :], axis=(2, 3)) * 1e3


# convert NaN to 0 to have black background
sigma_xx_20micron_average[np.isnan(sigma_xx_20micron_average)] = 0
sigma_yy_20micron_average[np.isnan(sigma_yy_20micron_average)] = 0

sigma_xx_40micron_average[np.isnan(sigma_xx_40micron_average)] = 0
sigma_yy_40micron_average[np.isnan(sigma_yy_40micron_average)] = 0


# set up plot parameters
# *****************************************************************************
n = 4  # every nth arrow will be plotted
pixelsize = 1.296  # in µm
pmax = 10  # mN/m

# create x- and y-axis for plotting maps
x_end = np.shape(sigma_xx_20micron_average)[1]
y_end = np.shape(sigma_xx_20micron_average)[0]
extent = [0, x_end * pixelsize, 0, y_end * pixelsize]

# create mesh for vectorplot
xq, yq = np.meshgrid(np.linspace(0, extent[1], x_end), np.linspace(0, extent[3], y_end))

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4, 3))

im = axes[0, 0].imshow(sigma_xx_20micron_average, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent,
                       vmin=0, vmax=pmax, aspect='auto')
axes[1, 0].imshow(sigma_xx_40micron_average, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent,
                  vmin=0, vmax=pmax, aspect='auto')

axes[0, 1].imshow(sigma_yy_20micron_average, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent,
                  vmin=0, vmax=pmax, aspect='auto')
axes[1, 1].imshow(sigma_yy_40micron_average, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent,
                  vmin=0, vmax=pmax, aspect='auto')


# adjust space in between plots
plt.subplots_adjust(wspace=0, hspace=0)

# remove axes
for ax in axes.flat:
    ax.axis('off')
    aspectratio = 1.0
    ratio_default = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.set_aspect(ratio_default * aspectratio)

# add colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist())
cbar.ax.set_title('mN/m')

# add title
plt.suptitle('Cell stresses', y=0.97, x=0.42)
plt.text(-80, 210, 'xx-Stress')
plt.text(25, 210, 'yy-Stress')

# add annotations
plt.text(0.18, 0.8, 'n=' + str(n_20micron_fs), transform=plt.figure(1).transFigure, color='w')
plt.text(0.18, 0.42, 'n=' + str(n_40micron_fs), transform=plt.figure(1).transFigure, color='w')

plt.text(0.48, 0.8, 'n=' + str(n_20micron_fs), transform=plt.figure(1).transFigure, color='w')
plt.text(0.48, 0.42, 'n=' + str(n_40micron_fs), transform=plt.figure(1).transFigure, color='w')

fig.savefig(figfolder + 'B.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'B.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 5C boxplots of strain energy and spreading sizes

# set up global plot parameters
# ******************************************************************************************************************************************
xticklabels = ['20 $\mathrm{\mu}$m', '40 $\mathrm{\mu}$m']  # which labels to put on x-axis
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(3, 3))  # create figure and axes
plt.subplots_adjust(wspace=0.45, hspace=0.45)  # adjust space in between plots

# Set up plot parameters for first panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'sigma_xx_baseline'  # variable that goes on the y-axis
ax = axes[0, 0]  # define on which axis the plot goes
colors = [colors_parent[0], colors_parent[1]]  # defines colors
ymin = 0  # minimum value on y-axis
ymax = 5  # maximum value on y-axis
yticks = np.arange(0, 5.1, 2.5)  # define where to put major ticks on y-axis
ylabel = None  # which label to put on y-axis
title = 'xx-Stress'  # title of plot

# make plots
make_box_and_swarmplots(x, y, df, ax, ymin, ymax, yticks, xticklabels, ylabel, title, colors)

# Set up plot parameters for second panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'sigma_yy_baseline'  # variable that goes on the y-axis
ax = axes[0, 1]  # define on which axis the plot goes
colors = [colors_parent[0], colors_parent[1]]  # defines colors
ymin = 0  # minimum value on y-axis
ymax = 20  # maximum value on y-axis
yticks = np.arange(0, 20.1, 2.5)  # define where to put major ticks on y-axis
ylabel = None  # which label to put on y-axis
title = 'yy-Stress'  # title of plot

# make plots
make_box_and_swarmplots(x, y, df, ax, ymin, ymax, yticks, xticklabels, ylabel, title, colors)

# Set up plot parameters for third panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'AIC_baseline'  # variable that goes on the y-axis
ax = axes[1, 0]  # define on which axis the plot goes
colors = [colors_parent[0], colors_parent[1], colors_parent[3]]  # defines colors
ymin = -1  # minimum value on y-axis
ymax = 1  # maximum value on y-axis
yticks = np.arange(-1, 1.1, 0.5)  # define where to put major ticks on y-axis
ylabel = None  # which label to put on y-axis
title = 'AIC'  # title of plot

# make plots
make_box_and_swarmplots(x, y, df, ax, ymin, ymax, yticks, xticklabels, ylabel, title, colors)


# Set up plot parameters for fourth panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'Es_baseline'  # variable that goes on the y-axis
ax = axes[1, 1]  # define on which axis the plot goes
colors = [colors_parent[0], colors_parent[1], colors_parent[3]]  # defines colors
ymin = 0  # minimum value on y-axis
ymax = 2  # maximum value on y-axis
yticks = np.arange(0, 2.1, 0.25)  # define where to put major ticks on y-axis
ylabel = None  # which label to put on y-axis
title = 'Strain energy'  # title of plot

# make plots
make_box_and_swarmplots(x, y, df, ax, ymin, ymax, yticks, xticklabels, ylabel, title, colors)

# Set up plot parameters for fifth panel
#######################################################################################################


# Set up plot parameters for sixth panel
#######################################################################################################
# ylabeloffset = -7
# xlabeloffset = 0
# colors = [colors_parent[0], colors_parent[1], colors_parent[3]]  # defines colors for scatterplot
#
# dotsize = 1.8  # size of datapoints in scatterplot
# linewidth_sw = 0.3  # linewidth of dots in scatterplot
# alpha_sw = 1  # transparency of dots in scatterplot
#
# ax = axes[2, 1]
# x = 'actin_angles'
# y = 'AIC_baseline'
# hue = 'keys'
# xmin = 15
# xmax = 75
# ymin = -1
# ymax = 1
# xticks = np.arange(15, 75.1, 15)
# yticks = np.arange(-1, 1.1, 0.5)
# xlabel = "Actin angle"  # "'$\mathrm{\sigma_{x, MSM}}$'
# ylabel = "AIC"  # '$\mathrm{\sigma_{x, CM}}$'
#
# corr, p = make_correlationplotsplots(x, y, hue, df, ax, xmin, xmax, ymin, ymax, xticks, yticks, xlabel, ylabel, colors, ylabeloffset=-8)
#
# # add line with slope 1 for visualisation
# ax.plot([xmin, xmax], [0, 0], linewidth=0.5, linestyle=':', color='grey')
# ax.plot([45, 45], [ymin, ymax], linewidth=0.5, linestyle=':', color='grey')
#
# plt.text(0.21 * xmax, 1.1 * ymax, 'R = ' + str(corr))
# # plt.text(0.52 * xmax, 1.1 * ymax, 'p = ' + '{:0.2e}'.format(p))
#
plt.savefig(figfolder + 'C.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'C.svg', dpi=300, bbox_inches="tight")
plt.show()

# # %% filter data to remove cells that don't react very much to opto stimulation
# def filter_data_main(data, title):
#     # concatenate data on which it will be determined which cells will be filtered
#     filterdata = data["MSM_data"]["RSI_normal_left"]
#
#     # move axis of variable to the last position for consistency
#     filterdata = np.moveaxis(filterdata, 0, -1)
#
#     # maximal allowed slope for linear fit of baseline
#     threshold = 0.05
#     opto_increase_filter = create_opto_increase_filter(filterdata, threshold)
#
#     # remove cells with unstable baselines
#     data["TFM_data"] = apply_filter(data["TFM_data"], opto_increase_filter)
#     data["MSM_data"] = apply_filter(data["MSM_data"], opto_increase_filter)
#     data["shape_data"] = apply_filter(data["shape_data"], opto_increase_filter)
#
#     new_N = np.sum(opto_increase_filter)
#     print(title + ": " + str(opto_increase_filter.shape[0] - new_N) + " cells were filtered out")
#
#     return data
#
# AR1to2d_halfstim = filter_data_main(AR1to2d_halfstim, "AR1to2d_halfstim")
# AR1to1d_halfstim = filter_data_main(AR1to1d_halfstim, "AR1to1d_halfstim")
# AR2to1d_halfstim = filter_data_main(AR2to1d_halfstim, "AR2to1d_halfstim")
#
# #%% prepare dataframe again after filtering
#
# # initialize empty dictionaries
# concatenated_data_1to2d = {}
# concatenated_data_1to1d = {}
# concatenated_data_2to1d = {}
# concatenated_data = {}
#
# # loop over all keys
# for key1 in AR1to1d_halfstim:  # keys are the same for all dictionaries so I'm just taking one example here
#     for key2 in AR1to1d_halfstim[key1]:
#         if AR1to1d_halfstim[key1][key2].ndim == 1:  # only 1D data can be stored in the data frame
#             # concatenate values from different experiments
#             concatenated_data_1to2d[key2] = AR1to2d_halfstim[key1][key2]
#             concatenated_data_1to1d[key2] = AR1to1d_halfstim[key1][key2]
#             concatenated_data_2to1d[key2] = AR2to1d_halfstim[key1][key2]
#
#             concatenated_data[key2] = np.concatenate(
#                 (concatenated_data_1to2d[key2], concatenated_data_1to1d[key2], concatenated_data_2to1d[key2]))
# key2 = 'Es_baseline'
# # get number of elements for both condition
# n_1to2d = concatenated_data_1to2d[key2].shape[0]
# n_1to1d = concatenated_data_1to1d[key2].shape[0]
# n_2to1d = concatenated_data_2to1d[key2].shape[0]
#
# # create a list of keys with the same dimensions as the data
# keys1to2d = ['AR1to2d' for i in range(n_1to2d)]
# keys1to1d = ['AR1to1d' for i in range(n_1to1d)]
# keys2to1d = ['AR2to1d' for i in range(n_2to1d)]
# keys = np.concatenate((keys1to2d, keys1to1d, keys2to1d))
#
# # add keys to dictionary with concatenated data
# concatenated_data['keys'] = keys
#
# # Creates DataFrame
# df = pd.DataFrame(concatenated_data)
#
# # # remove outliers
# filter_outlier1 = np.abs(zscore(df["attenuation_position"])) < 1
# filter_outlier2 = np.abs(zscore(df["attenuation_length"])) < 1
# filter_outlier = np.all((filter_outlier1, filter_outlier2), axis=0)
# df = df[filter_outlier]
#
# stressmappixelsize = 0.864   # in µm
# # convert to more convenient units for plotting
# df['Es_baseline'] *= 1e12  # convert to fJ
# df['spreadingsize_baseline'] *= 1e12  # convert to µm²
# df['sigma_xx_baseline'] *= 1e3  # convert to mN/m
# df['sigma_yy_baseline'] *= 1e3  # convert to mN/m
# df["attenuation_position"] *= stressmappixelsize   # convert to µm
# df["attenuation_length"] *= stressmappixelsize   # convert to µm

# %% plot figure 5D, normal stress maps

# prepare data first

# Calculate average maps over first 20 frames and all cells to get average maps
sigma_normal_20micron_lefthalf_stim_average = \
    np.nanmean(tissues_20micron_lefthalf_stim["MSM_data"]["sigma_normal"][:, :, 0:20, :], axis=(2, 3)) * 1e3  # convert to mN/m
sigma_normal_20micron_tophalf_stim_average = \
    np.nanmean(tissues_20micron_tophalf_stim["MSM_data"]["sigma_normal"][:, :, 0:20, :], axis=(2, 3)) * 1e3
sigma_normal_40micron_lefthalf_stim_average = \
    np.nanmean(tissues_40micron_lefthalf_stim["MSM_data"]["sigma_normal"][:, :, 0:20, :], axis=(2, 3)) * 1e3
sigma_normal_40micron_tophalf_stim_average = \
    np.nanmean(tissues_40micron_tophalf_stim["MSM_data"]["sigma_normal"][:, :, 0:20, :], axis=(2, 3)) * 1e3

# convert NaN to 0 to have black background
sigma_normal_20micron_lefthalf_stim_average[np.isnan(sigma_normal_20micron_lefthalf_stim_average)] = 0
sigma_normal_20micron_tophalf_stim_average[np.isnan(sigma_normal_20micron_tophalf_stim_average)] = 0
sigma_normal_40micron_lefthalf_stim_average[np.isnan(sigma_normal_40micron_lefthalf_stim_average)] = 0
sigma_normal_40micron_tophalf_stim_average[np.isnan(sigma_normal_40micron_tophalf_stim_average)] = 0

# set up plot parameters
# *****************************************************************************
n = 4  # every nth arrow will be plotted
pixelsize = 1.296  # in µm
pmax = 10  # mN/m

# create x- and y-axis for plotting maps
x_end = np.shape(sigma_normal_20micron_lefthalf_stim_average)[1]
y_end = np.shape(sigma_normal_20micron_lefthalf_stim_average)[0]
extent = [0, x_end * pixelsize, 0, y_end * pixelsize]

# create mesh for vectorplot
xq, yq = np.meshgrid(np.linspace(0, extent[1], x_end), np.linspace(0, extent[3], y_end))

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(2, 5.5))

im = axes[0].imshow(sigma_normal_20micron_lefthalf_stim_average, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent,
                       vmin=0, vmax=pmax, aspect='auto')
axes[1].imshow(sigma_normal_20micron_tophalf_stim_average, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent,
                  vmin=0, vmax=pmax, aspect='auto')
axes[2].imshow(sigma_normal_40micron_lefthalf_stim_average, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent,
                  vmin=0, vmax=pmax, aspect='auto')
axes[3].imshow(sigma_normal_40micron_tophalf_stim_average, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent,
                  vmin=0, vmax=pmax, aspect='auto')


# adjust space in between plots
plt.subplots_adjust(wspace=0, hspace=0)

# remove axes
for ax in axes.flat:
    ax.axis('off')
    aspectratio = 1.0
    ratio_default = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.set_aspect(ratio_default * aspectratio)

# add colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist())
cbar.ax.set_title('mN/m')

# add title
plt.suptitle('Normal stresses', y=0.95, x=0.52)
# plt.text(-20, 230, '$\mathrm{\Delta}$ Normal stresses')

# add annotations
plt.text(0.36, 0.85, 'n=' + str(n_20micron_ls), transform=plt.figure(1).transFigure, color='w')
plt.text(0.36, 0.64, 'n=' + str(n_20micron_ts), transform=plt.figure(1).transFigure, color='w')
plt.text(0.36, 0.45, 'n=' + str(n_40micron_ls), transform=plt.figure(1).transFigure, color='w')
plt.text(0.36, 0.28, 'n=' + str(n_20micron_fs), transform=plt.figure(1).transFigure, color='w')

fig.savefig(figfolder + 'D.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'D.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 5E

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.linspace(-78, 78, 120)
x = x[::2]  # downsample data for nicer plotting
xticks = np.arange(-60, 60.1, 30)  # define where the major ticks are gonna be
xlabel = 'position [µm]'
ymin = 0
ymax = 5
yticks = np.arange(ymin, ymax + 0.001, 1)
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(1.2, 5.5))  # create figure and axes
plt.subplots_adjust(wspace=0.4, hspace=0.35)  # adjust space in between plots
# ******************************************************************************************************************************************

# Set up plot parameters for first panel
#######################################################################################################
ax = axes[0]
color = colors_parent[0]
ylabel = None
title = '$\mathrm{\sigma _{normal}(x)}$  [nN]\n baseline'
y = tissues_20micron_lefthalf_stim["MSM_data"]["sigma_normal_x_profile_baseline"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False)

# # Set up plot parameters for second panel
# #######################################################################################################
ax = axes[1]
color = colors_parent[1]
ylabel = None
title = None
y = tissues_20micron_tophalf_stim["MSM_data"]["sigma_normal_x_profile_baseline"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False)

# # Set up plot parameters for third panel
# #######################################################################################################
ax = axes[2]
color = colors_parent[2]
ylabel = None
title = None
y = tissues_40micron_lefthalf_stim["MSM_data"]["sigma_normal_x_profile_baseline"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False)

# # Set up plot parameters for third panel
# #######################################################################################################
ax = axes[3]
color = colors_parent[3]
ylabel = None
title = None
y = tissues_40micron_tophalf_stim["MSM_data"]["sigma_normal_x_profile_baseline"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False)


plt.savefig(figfolder + 'E.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'E.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 5F, stress map differences

# prepare data first

# concatenate TFM maps from different experiments and calculate average maps over first 20 frames and all cells to get average maps
sigma_normal_20micron_lefthalf_stim_diff = \
    np.nanmean(tissues_20micron_lefthalf_stim["MSM_data"]["sigma_normal"][:, :, 32, :] -
               tissues_20micron_lefthalf_stim["MSM_data"]["sigma_normal"][:, :, 20, :], axis=2) * 1e3  # convert to mN/m

sigma_normal_20micron_tophalf_stim_diff = \
    np.nanmean(tissues_20micron_tophalf_stim["MSM_data"]["sigma_normal"][:, :, 32, :] -
               tissues_20micron_tophalf_stim["MSM_data"]["sigma_normal"][:, :, 20, :], axis=2) * 1e3  # convert to mN/m

sigma_normal_40micron_lefthalf_stim_diff = \
    np.nanmean(tissues_40micron_lefthalf_stim["MSM_data"]["sigma_normal"][:, :, 32, :] -
               tissues_40micron_lefthalf_stim["MSM_data"]["sigma_normal"][:, :, 20, :], axis=2) * 1e3  # convert to mN/m

sigma_normal_40micron_tophalf_stim_diff = \
    np.nanmean(tissues_40micron_tophalf_stim["MSM_data"]["sigma_normal"][:, :, 32, :] -
               tissues_40micron_tophalf_stim["MSM_data"]["sigma_normal"][:, :, 20, :], axis=2) * 1e3  # convert to mN/m


# set up plot parameters
# *****************************************************************************

pixelsize = 1.296  # in µm
sigma_max = 1  # kPa
sigma_min = -1  # kPa

# create x- and y-axis for plotting maps
x_end = np.shape(sigma_normal_20micron_lefthalf_stim_diff)[1]
y_end = np.shape(sigma_normal_20micron_lefthalf_stim_diff)[0]
extent = [0, x_end * pixelsize, 0, y_end * pixelsize]

# create mesh for vectorplot
xq, yq = np.meshgrid(np.linspace(0, extent[1], x_end), np.linspace(0, extent[3], y_end))

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(2, 5.5))

im = axes[0].imshow(sigma_normal_20micron_lefthalf_stim_diff, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent,
                       vmin=sigma_min, vmax=sigma_max, aspect='auto')

axes[1].imshow(sigma_normal_20micron_tophalf_stim_diff, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent,
                  vmin=sigma_min, vmax=sigma_max, aspect='auto')

axes[2].imshow(sigma_normal_40micron_lefthalf_stim_diff, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent,
                       vmin=sigma_min, vmax=sigma_max, aspect='auto')

axes[3].imshow(sigma_normal_40micron_tophalf_stim_diff, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent,
                  vmin=sigma_min, vmax=sigma_max, aspect='auto')


# adjust space in between plots
plt.subplots_adjust(wspace=0, hspace=0)


# remove axes
for ax in axes.flat:
    ax.axis('off')
    aspectratio = 1.0
    ratio_default = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.set_aspect(ratio_default * aspectratio)

# add colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist())
cbar.ax.set_title('mN/m')

# add title
plt.suptitle('$\mathrm{\Delta}$ Normal stresses', y=0.98, x=0.5)
# plt.text(-20, 230, '$\mathrm{\Delta}$ Normal stresses')

# add annotations
plt.text(0.36, 0.85, 'n=' + str(n_20micron_ls), transform=plt.figure(1).transFigure, color='black')
plt.text(0.36, 0.64, 'n=' + str(n_20micron_ts), transform=plt.figure(1).transFigure, color='black')
plt.text(0.36, 0.45, 'n=' + str(n_40micron_ls), transform=plt.figure(1).transFigure, color='black')
plt.text(0.36, 0.28, 'n=' + str(n_20micron_fs), transform=plt.figure(1).transFigure, color='black')

# save figure
fig.savefig(figfolder + 'F.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'F.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 5G

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.linspace(-78, 78, 120)
x = x[::2]  # downsample data for nicer plotting
xticks = np.arange(-60, 60.1, 30)  # define where the major ticks are gonna be
xlabel = 'position [µm]'
ymin = -0.1
ymax = 1
yticks = np.arange(ymin, ymax + 0.001, 0.25)
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(1.2, 5.5))  # create figure and axes
plt.subplots_adjust(wspace=0.4, hspace=0.35)  # adjust space in between plots
# ******************************************************************************************************************************************

# # Set up plot parameters for fourth panel
# #######################################################################################################
ax = axes[0]
color = colors_parent[0]
ylabel = None
title = '$\mathrm{\sigma _{normal}(x)}$ [nN] \n increase'
y = tissues_20micron_lefthalf_stim["MSM_data"]["sigma_normal_x_profile_increase"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False)

# # Set up plot parameters for fifth panel
# #######################################################################################################
ax = axes[1]
color = colors_parent[1]
ylabel = None
title = None
y = tissues_20micron_tophalf_stim["MSM_data"]["sigma_normal_x_profile_increase"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False)

# # Set up plot parameters for sixth panel
# # #######################################################################################################
ax = axes[2]
color = colors_parent[2]
ylabel = None
title = None
y = tissues_40micron_lefthalf_stim["MSM_data"]["sigma_normal_x_profile_increase"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False)

# # Set up plot parameters for sixth panel
# # #######################################################################################################
ax = axes[3]
color = colors_parent[3]
ylabel = None
title = None
y = tissues_40micron_tophalf_stim["MSM_data"]["sigma_normal_x_profile_increase"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False)


plt.savefig(figfolder + 'G.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'G.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% prepare dataframe for boxplots

# initialize empty dictionaries
# concatenated_data_20micron_full_stim = {}
concatenated_data_20micron_lefthalf_stim = {}
concatenated_data_20micron_tophalf_stim = {}
# concatenated_data_40micron_full_stim = {}
concatenated_data_40micron_lefthalf_stim = {}
concatenated_data_40micron_tophalf_stim = {}
concatenated_data = {}

# loop over all keys
for key1 in tissues_20micron_full_stim:  # keys are the same for all dictionaries so I'm just taking one example here
    for key2 in tissues_20micron_full_stim[key1]:
        if tissues_20micron_full_stim[key1][key2].ndim == 1:  # only 1D data can be stored in the data frame
            # concatenate values from different experiments
            # concatenated_data_20micron_full_stim[key2] = tissues_20micron_full_stim[key1][key2]
            concatenated_data_20micron_lefthalf_stim[key2] = tissues_20micron_lefthalf_stim[key1][key2]
            concatenated_data_20micron_tophalf_stim[key2] = tissues_20micron_tophalf_stim[key1][key2]
            # concatenated_data_40micron_full_stim[key2] = tissues_40micron_full_stim[key1][key2]
            concatenated_data_40micron_lefthalf_stim[key2] = tissues_40micron_lefthalf_stim[key1][key2]
            concatenated_data_40micron_tophalf_stim[key2] = tissues_40micron_tophalf_stim[key1][key2]

            # concatenated_data[key2] = np.concatenate(
            #     (concatenated_data_20micron_full_stim[key2], concatenated_data_20micron_lefthalf_stim[key2], concatenated_data_20micron_tophalf_stim[key2],
            #      concatenated_data_40micron_full_stim[key2], concatenated_data_40micron_lefthalf_stim[key2], concatenated_data_40micron_tophalf_stim[key2]))
            concatenated_data[key2] = np.concatenate(
                (concatenated_data_20micron_lefthalf_stim[key2], concatenated_data_20micron_tophalf_stim[key2],
                 concatenated_data_40micron_lefthalf_stim[key2], concatenated_data_40micron_tophalf_stim[key2]))
key2 = 'Es_baseline'
# get number of elements for both condition
n_20micron_fs = concatenated_data_20micron_full_stim[key2].shape[0]
n_20micron_ls = concatenated_data_20micron_lefthalf_stim[key2].shape[0]
n_20micron_ts = concatenated_data_20micron_tophalf_stim[key2].shape[0]
n_40micron_fs = concatenated_data_40micron_full_stim[key2].shape[0]
n_40micron_ls = concatenated_data_40micron_lefthalf_stim[key2].shape[0]
n_40micron_ts = concatenated_data_40micron_tophalf_stim[key2].shape[0]

# create a list of keys with the same dimensions as the data
# keys20micron_fs = ['20micron_fs' for i in range(n_20micron_fs)]
keys20micron_ls = ['20micron_ls' for i in range(n_20micron_ls)]
keys20micron_ts = ['20micron_ts' for i in range(n_20micron_ts)]
# keys40micron_fs = ['40micron_fs' for i in range(n_40micron_fs)]
keys40micron_ls = ['40micron_ls' for i in range(n_40micron_ls)]
keys40micron_ts = ['40micron_ts' for i in range(n_40micron_ts)]

# keys = np.concatenate((keys20micron_fs, keys20micron_ls, keys20micron_ts, keys40micron_fs, keys40micron_ls, keys40micron_ts))
keys = np.concatenate((keys20micron_ls, keys20micron_ts, keys40micron_ls, keys40micron_ts))

# add keys to dictionary with concatenated data
concatenated_data['keys'] = keys

# Creates DataFrame
df = pd.DataFrame(concatenated_data)

# convert to more convenient units for plotting
df_plot_units = df  # all units here are in SI units
df_plot_units['Es_baseline'] *= 1e12  # convert to fJ
df_plot_units['sigma_xx_baseline'] *= 1e3  # convert to mN/m
df_plot_units['sigma_yy_baseline'] *= 1e3  # convert to mN/m
# %% plot figure 5H boxplots of attenuation length and position

# set up global plot parameters
# ******************************************************************************************************************************************
xticklabels = ['1', '2', '3', '4']  # which labels to put on x-axis
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(1.3, 5.5))  # create figure and axes
plt.subplots_adjust(wspace=0.45, hspace=0.4)  # adjust space in between plots

# Set up plot parameters for first panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'attenuation_position'  # variable that goes on the y-axis
ax = axes[0]  # define on which axis the plot goes
colors = colors_parent  # defines colors
ymin = -20  # minimum value on y-axis
ymax = 30  # maximum value on y-axis
yticks = np.arange(-20, 50, 10)  # define where to put major ticks on y-axis
ylabel = '$\mathrm{a_p}$ [µm]'  # which label to put on y-axis
title = 'Attenuation position'  # title of plot

# make plots
make_box_and_swarmplots(x, y, df, ax, ymin, ymax, yticks, xticklabels, ylabel, title, colors)
# make_box_and_swarmplots_with_test(x, y, df, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title, colors,
#                                   width_bp=0.5, ylabeloffset=-6)

# Set up plot parameters for second panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'attenuation_length'  # variable that goes on the y-axis
ax = axes[1]  # define on which axis the plot goes
colors = colors_parent  # defines colors
ymin = -5  # minimum value on y-axis
ymax = 10  # maximum value on y-axis
yticks = np.arange(-5, 10.1, 5)  # define where to put major ticks on y-axis
ylabel = '$\mathrm{a_l}$ [µm]'   # which label to put on y-axis
title = 'Attenuation length'  # title of plot
# make plots
make_box_and_swarmplots(x, y, df, ax, ymin, ymax, yticks, xticklabels, ylabel, title, colors)
# make_box_and_swarmplots_with_test(x, y, df, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title, colors,
#                                   width_bp=0.5, ylabeloffset=-6)

# Set up plot parameters for third panel
#######################################################################################################
x = 'AIC_baseline'
y = 'attenuation_position'
hue = 'keys'
ax = axes[2]
xmin = -1
xmax = 1
ymin = -20
ymax = 20
xticks = np.arange(-1, 1.1, 0.5)
yticks = np.arange(-20, 20.1, 10)
xlabel = 'Anisotropy coefficient'
ylabel = '$\mathrm{a_p}$ [µm]'

corr, p = make_correlationplotsplots(x, y, hue, df, ax, xmin, xmax, ymin, ymax, xticks, yticks, xlabel, ylabel, colors)

# annotate pearson R and p-value
plt.text(xmin + 0.1 * xmax, 1.1*ymax, 'R = ' + str(corr))
# plt.text(xmin + 0.1 * xmax, ymin + 0.1 * ymax, 'p = ' + '{:0.2e}'.format(p))

# # Set up plot parameters for fourth panel
# #######################################################################################################
# x = 'AIC_baseline'
# y = 'attenuation_length'
# hue = 'keys'
# ax = axes[3]
# xmin = -1
# xmax = 1
# ymin = -20
# ymax = 20
# xticks = np.arange(-1, 1.1, 0.5)
# yticks = np.arange(-20, 20.1, 10)
# xlabel = 'Anisotropy coefficient'
# ylabel = '$\mathrm{a_l}$ [µm]'
#
# corr, p = make_correlationplotsplots(x, y, hue, df, ax, xmin, xmax, ymin, ymax, xticks, yticks, xlabel, ylabel, colors)

# annotate pearson R and p-value
plt.text(xmin + 0.1 * xmax, 1.1*ymax, 'R = ' + str(corr))
# plt.text(xmin + 0.1 * xmax, ymin + 0.1 * ymax, 'p = ' + '{:0.2e}'.format(p))

plt.savefig(figfolder + 'H.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'H.svg', dpi=300, bbox_inches="tight")
plt.show()



