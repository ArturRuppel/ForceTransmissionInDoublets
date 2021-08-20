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
AR1to2d_halfstim = pickle.load(open(folder + "analysed_data/AR1to2d_halfstim.dat", "rb"))
AR1to1d_halfstim = pickle.load(open(folder + "analysed_data/AR1to1d_halfstim.dat", "rb"))
AR2to1d_halfstim = pickle.load(open(folder + "analysed_data/AR2to1d_halfstim.dat", "rb"))

# define some colors for the plots
colors_parent = ['#026473', '#E3CC69', '#77C8A6', '#D96248']
colors_parent_dark = ['#01353D', '#564910', '#235741', '#A93B23']

figfolder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/_Figure5/"
if not os.path.exists(figfolder):
    os.mkdir(figfolder)
# %% prepare dataframe for boxplots

# initialize empty dictionaries
concatenated_data_1to2d = {}
concatenated_data_1to1d = {}
concatenated_data_2to1d = {}
concatenated_data = {}

# loop over all keys
for key1 in AR1to1d_halfstim:  # keys are the same for all dictionaries so I'm just taking one example here
    for key2 in AR1to1d_halfstim[key1]:
        if AR1to1d_halfstim[key1][key2].ndim == 1:  # only 1D data can be stored in the data frame
            # concatenate values from different experiments
            concatenated_data_1to2d[key2] = AR1to2d_halfstim[key1][key2]
            concatenated_data_1to1d[key2] = AR1to1d_halfstim[key1][key2]
            concatenated_data_2to1d[key2] = AR2to1d_halfstim[key1][key2]

            concatenated_data[key2] = np.concatenate(
                (concatenated_data_1to2d[key2], concatenated_data_1to1d[key2], concatenated_data_2to1d[key2]))
key2 = 'Es_baseline'
# get number of elements for both condition
n_1to2d = concatenated_data_1to2d[key2].shape[0]
n_1to1d = concatenated_data_1to1d[key2].shape[0]
n_2to1d = concatenated_data_2to1d[key2].shape[0]

# create a list of keys with the same dimensions as the data
keys1to2d = ['AR1to2d' for i in range(n_1to2d)]
keys1to1d = ['AR1to1d' for i in range(n_1to1d)]
keys2to1d = ['AR2to1d' for i in range(n_2to1d)]
keys = np.concatenate((keys1to2d, keys1to1d, keys2to1d))

# add keys to dictionary with concatenated data
concatenated_data['keys'] = keys

# Creates DataFrame
df = pd.DataFrame(concatenated_data)

# convert to more convenient units for plotting
df_plot_units = df  # all units here are in SI units
df_plot_units['Es_baseline'] *= 1e12  # convert to fJ
df_plot_units['spreadingsize_baseline'] *= 1e12  # convert to µm²
df_plot_units['sigma_xx_baseline'] *= 1e3  # convert to mN/m
df_plot_units['sigma_yy_baseline'] *= 1e3  # convert to mN/m

# %% prepare dataframe for boxplots
n_1to2d = AR1to2d_halfstim['MSM_data']['RSI_xx_left'].shape[0]
n_1to1d = AR1to1d_halfstim['MSM_data']['RSI_xx_left'].shape[0]
n_2to1d = AR2to1d_halfstim['MSM_data']['RSI_xx_left'].shape[0]

RSI_data_1to2d = {}
RSI_data_1to1d = {}
RSI_data_2to1d = {}

RSI_data_1to2d['sigma'] = np.concatenate((AR1to2d_halfstim['MSM_data']['RSI_xx_left'],
                                          AR1to2d_halfstim['MSM_data']['RSI_xx_right'],
                                          AR1to2d_halfstim['MSM_data']['RSI_yy_left'],
                                          AR1to2d_halfstim['MSM_data']['RSI_yy_right']))
RSI_data_1to1d['sigma'] = np.concatenate((AR1to1d_halfstim['MSM_data']['RSI_xx_left'],
                                          AR1to1d_halfstim['MSM_data']['RSI_xx_right'],
                                          AR1to1d_halfstim['MSM_data']['RSI_yy_left'],
                                          AR1to1d_halfstim['MSM_data']['RSI_yy_right']))
RSI_data_2to1d['sigma'] = np.concatenate((AR2to1d_halfstim['MSM_data']['RSI_xx_left'],
                                          AR2to1d_halfstim['MSM_data']['RSI_xx_right'],
                                          AR2to1d_halfstim['MSM_data']['RSI_yy_left'],
                                          AR2to1d_halfstim['MSM_data']['RSI_yy_right']))

keys1to2d = np.concatenate((['RSI_xx_left' for i in range(n_1to2d)], ['RSI_xx_right' for i in range(n_1to2d)],
                            ['RSI_yy_left' for i in range(n_1to2d)], ['RSI_yy_right' for i in range(n_1to2d)]))
keys1to1d = np.concatenate((['RSI_xx_left' for i in range(n_1to1d)], ['RSI_xx_right' for i in range(n_1to1d)],
                            ['RSI_yy_left' for i in range(n_1to1d)], ['RSI_yy_right' for i in range(n_1to1d)]))
keys2to1d = np.concatenate((['RSI_xx_left' for i in range(n_2to1d)], ['RSI_xx_right' for i in range(n_2to1d)],
                            ['RSI_yy_left' for i in range(n_2to1d)], ['RSI_yy_right' for i in range(n_2to1d)]))

RSI_data_1to2d['keys'] = keys1to2d
RSI_data_1to1d['keys'] = keys1to1d
RSI_data_2to1d['keys'] = keys2to1d

df1to2d = pd.DataFrame(RSI_data_1to2d)
df1to1d = pd.DataFrame(RSI_data_1to1d)
df2to1d = pd.DataFrame(RSI_data_2to1d)

# %% plot figure 5A, force maps

# prepare data first

# Calculate average maps over first 20 frames and all cells to get average maps
Tx_1to2d_average = np.nanmean(AR1to2d_halfstim["TFM_data"]["Tx"][:, :, 0:20, :], axis=(2, 3))
Ty_1to2d_average = np.nanmean(AR1to2d_halfstim["TFM_data"]["Ty"][:, :, 0:20, :], axis=(2, 3))

Tx_1to1d_average = np.nanmean(AR1to1d_halfstim["TFM_data"]["Tx"][:, :, 0:20, :], axis=(2, 3))
Ty_1to1d_average = np.nanmean(AR1to1d_halfstim["TFM_data"]["Ty"][:, :, 0:20, :], axis=(2, 3))

Tx_2to1d_average = np.nanmean(AR2to1d_halfstim["TFM_data"]["Tx"][:, :, 0:20, :], axis=(2, 3))
Ty_2to1d_average = np.nanmean(AR2to1d_halfstim["TFM_data"]["Ty"][:, :, 0:20, :], axis=(2, 3))

# calculate amplitudes
T_1to2d_average = np.sqrt(Tx_1to2d_average ** 2 + Ty_1to2d_average ** 2)
T_1to1d_average = np.sqrt(Tx_1to1d_average ** 2 + Ty_1to1d_average ** 2)
T_2to1d_average = np.sqrt(Tx_2to1d_average ** 2 + Ty_2to1d_average ** 2)

# crop maps
crop_start = 2
crop_end = 90

Tx_1to2d_average_crop = Tx_1to2d_average[crop_start:crop_end, crop_start:crop_end] * 1e-3  # convert to kPa
Ty_1to2d_average_crop = Ty_1to2d_average[crop_start:crop_end, crop_start:crop_end] * 1e-3
T_1to2d_average_crop = T_1to2d_average[crop_start:crop_end, crop_start:crop_end] * 1e-3

Tx_1to1d_average_crop = Tx_1to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e-3  # convert to kPa
Ty_1to1d_average_crop = Ty_1to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e-3
T_1to1d_average_crop = T_1to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e-3

Tx_2to1d_average_crop = Tx_2to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e-3  # convert to kPa
Ty_2to1d_average_crop = Ty_2to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e-3
T_2to1d_average_crop = T_2to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e-3

# set up plot parameters
# *****************************************************************************
n = 4  # every nth arrow will be plotted
pixelsize = 0.864  # in µm
pmax = 2  # kPa

# create x- and y-axis for plotting maps
x_end = np.shape(T_1to1d_average_crop)[1]
y_end = np.shape(T_1to1d_average_crop)[0]
extent = [0, x_end * pixelsize, 0, y_end * pixelsize]

# create mesh for vectorplot    
xq, yq = np.meshgrid(np.linspace(0, extent[1], x_end), np.linspace(0, extent[3], y_end))

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(2.5, 4))

im = axes[0].imshow(T_1to2d_average_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0,
                    vmax=pmax, aspect='auto')
axes[0].quiver(xq[::n, ::n], yq[::n, ::n], Tx_1to2d_average_crop[::n, ::n], Ty_1to2d_average_crop[::n, ::n],
               angles='xy', scale=10, units='width', color="r")
# axes[0,0].set_title('n=1', pad=-400, color='r')

axes[1].imshow(T_1to1d_average_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0,
               vmax=pmax, aspect='auto')
axes[1].quiver(xq[::n, ::n], yq[::n, ::n], Tx_1to1d_average_crop[::n, ::n], Ty_1to1d_average_crop[::n, ::n],
               angles='xy', scale=10, units='width', color="r")

axes[2].imshow(T_2to1d_average_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0,
               vmax=pmax, aspect='auto')
axes[2].quiver(xq[::n, ::n], yq[::n, ::n], Tx_2to1d_average_crop[::n, ::n], Ty_2to1d_average_crop[::n, ::n],
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
plt.text(0.48, 0.853, 'n=' + str(n_1to2d), transform=plt.figure(1).transFigure, color='w')
plt.text(0.48, 0.598, 'n=' + str(n_1to1d), transform=plt.figure(1).transFigure, color='w')
plt.text(0.48, 0.343, 'n=' + str(n_2to1d), transform=plt.figure(1).transFigure, color='w')

fig.savefig(figfolder + 'A1.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'A1.svg', dpi=300, bbox_inches="tight")

plt.show()
# %% plot figure 5A, stress maps

# prepare data first

# Calculate average maps over first 20 frames and all cells to get average maps
sigma_xx_1to2d_average = np.nanmean(AR1to2d_halfstim["MSM_data"]["sigma_xx"][:, :, 0:20, :], axis=(2, 3))
sigma_yy_1to2d_average = np.nanmean(AR1to2d_halfstim["MSM_data"]["sigma_yy"][:, :, 0:20, :], axis=(2, 3))

sigma_xx_1to1d_average = np.nanmean(AR1to1d_halfstim["MSM_data"]["sigma_xx"][:, :, 0:20, :], axis=(2, 3))
sigma_yy_1to1d_average = np.nanmean(AR1to1d_halfstim["MSM_data"]["sigma_yy"][:, :, 0:20, :], axis=(2, 3))

sigma_xx_2to1d_average = np.nanmean(AR2to1d_halfstim["MSM_data"]["sigma_xx"][:, :, 0:20, :], axis=(2, 3))
sigma_yy_2to1d_average = np.nanmean(AR2to1d_halfstim["MSM_data"]["sigma_yy"][:, :, 0:20, :], axis=(2, 3))

# convert NaN to 0 to have black background
sigma_xx_1to2d_average[np.isnan(sigma_xx_1to2d_average)] = 0
sigma_yy_1to2d_average[np.isnan(sigma_yy_1to2d_average)] = 0

sigma_xx_1to1d_average[np.isnan(sigma_xx_1to1d_average)] = 0
sigma_yy_1to1d_average[np.isnan(sigma_yy_1to1d_average)] = 0

sigma_xx_2to1d_average[np.isnan(sigma_xx_2to1d_average)] = 0
sigma_yy_2to1d_average[np.isnan(sigma_yy_2to1d_average)] = 0

# crop maps
crop_start = 2
crop_end = 90

sigma_xx_1to2d_average_crop = sigma_xx_1to2d_average[crop_start:crop_end, crop_start:crop_end] * 1e3  # convert to mN/m
sigma_yy_1to2d_average_crop = sigma_yy_1to2d_average[crop_start:crop_end, crop_start:crop_end] * 1e3

sigma_xx_1to1d_average_crop = sigma_xx_1to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e3
sigma_yy_1to1d_average_crop = sigma_yy_1to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e3

sigma_xx_2to1d_average_crop = sigma_xx_2to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e3
sigma_yy_2to1d_average_crop = sigma_yy_2to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e3

# set up plot parameters
# *****************************************************************************
n = 4  # every nth arrow will be plotted
pixelsize = 0.864  # in µm
pmax = 10  # mN/m

# create x- and y-axis for plotting maps
x_end = np.shape(sigma_xx_1to1d_average_crop)[1]
y_end = np.shape(sigma_xx_1to1d_average_crop)[0]
extent = [0, x_end * pixelsize, 0, y_end * pixelsize]

# create mesh for vectorplot    
xq, yq = np.meshgrid(np.linspace(0, extent[1], x_end), np.linspace(0, extent[3], y_end))

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(3.32, 4))

im = axes[0, 0].imshow(sigma_xx_1to2d_average_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent,
                       vmin=0, vmax=pmax, aspect='auto')
axes[1, 0].imshow(sigma_xx_1to1d_average_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent,
                  vmin=0, vmax=pmax, aspect='auto')
axes[2, 0].imshow(sigma_xx_2to1d_average_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent,
                  vmin=0, vmax=pmax, aspect='auto')

axes[0, 1].imshow(sigma_yy_1to2d_average_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent,
                  vmin=0, vmax=pmax, aspect='auto')
axes[1, 1].imshow(sigma_yy_1to1d_average_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent,
                  vmin=0, vmax=pmax, aspect='auto')
axes[2, 1].imshow(sigma_yy_2to1d_average_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent,
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
plt.suptitle('Cell stresses', y=0.95, x=0.42)
plt.text(-55, 230, 'xx-Stress')
plt.text(20, 230, 'yy-Stress')

# add annotations
plt.text(0.25, 0.853, 'n=' + str(n_1to2d), transform=plt.figure(1).transFigure, color='w')
plt.text(0.25, 0.598, 'n=' + str(n_1to1d), transform=plt.figure(1).transFigure, color='w')
plt.text(0.25, 0.343, 'n=' + str(n_2to1d), transform=plt.figure(1).transFigure, color='w')

plt.text(0.55, 0.853, 'n=' + str(n_1to2d), transform=plt.figure(1).transFigure, color='w')
plt.text(0.55, 0.598, 'n=' + str(n_1to1d), transform=plt.figure(1).transFigure, color='w')
plt.text(0.55, 0.343, 'n=' + str(n_2to1d), transform=plt.figure(1).transFigure, color='w')

fig.savefig(figfolder + 'A2.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'A2.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 5B boxplots of strain energy and spreading sizes

# set up global plot parameters
# ******************************************************************************************************************************************
xticklabels = ['1to2', '1to1', '2to1']  # which labels to put on x-axis
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(3, 4))  # create figure and axes
plt.subplots_adjust(wspace=0.45, hspace=0.45)  # adjust space in between plots

# Set up plot parameters for first panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'spreadingsize_baseline'  # variable that goes on the y-axis
ax = axes[0, 0]  # define on which axis the plot goes
colors = [colors_parent[0], colors_parent[1], colors_parent[3]]  # defines colors
ymin = 1000  # minimum value on y-axis
ymax = 2000  # maximum value on y-axis
yticks = np.arange(1000, 2001, 250)  # define where to put major ticks on y-axis
ylabel = None  # which label to put on y-axis
title = 'Spreading size'  # title of plot

# make plots
make_four_box_and_swarmplots(x, y, df, ax, ymin, ymax, yticks, xticklabels, ylabel, title, colors)

# Set up plot parameters for second panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'Es_baseline'  # variable that goes on the y-axis
ax = axes[0, 1]  # define on which axis the plot goes
colors = [colors_parent[0], colors_parent[1], colors_parent[3]]  # defines colors
ymin = 0  # minimum value on y-axis
ymax = 1  # maximum value on y-axis
yticks = np.arange(0, 1.1, 0.25)  # define where to put major ticks on y-axis
ylabel = None  # which label to put on y-axis
title = 'Strain energy'  # title of plot

# make plots
make_four_box_and_swarmplots(x, y, df, ax, ymin, ymax, yticks, xticklabels, ylabel, title, colors)

# Set up plot parameters for third panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'sigma_xx_baseline'  # variable that goes on the y-axis
ax = axes[1, 0]  # define on which axis the plot goes
colors = [colors_parent[0], colors_parent[1], colors_parent[3]]  # defines colors
ymin = 0  # minimum value on y-axis
ymax = 10  # maximum value on y-axis
yticks = np.arange(0, 10.1, 2.5)  # define where to put major ticks on y-axis
ylabel = None  # which label to put on y-axis
title = 'xx-Stress'  # title of plot

# make plots
make_four_box_and_swarmplots(x, y, df, ax, ymin, ymax, yticks, xticklabels, ylabel, title, colors)

# Set up plot parameters for fourth panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'sigma_yy_baseline'  # variable that goes on the y-axis
ax = axes[1, 1]  # define on which axis the plot goes
colors = [colors_parent[0], colors_parent[1], colors_parent[3]]  # defines colors
ymin = 0  # minimum value on y-axis
ymax = 10  # maximum value on y-axis
yticks = np.arange(0, 10.1, 2.5)  # define where to put major ticks on y-axis
ylabel = None  # which label to put on y-axis
title = 'yy-Stress'  # title of plot

# make plots
make_four_box_and_swarmplots(x, y, df, ax, ymin, ymax, yticks, xticklabels, ylabel, title, colors)

# Set up plot parameters for fifth panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'AIC_baseline'  # variable that goes on the y-axis
ax = axes[2, 0]  # define on which axis the plot goes
colors = [colors_parent[0], colors_parent[1], colors_parent[3]]  # defines colors
ymin = -1  # minimum value on y-axis
ymax = 1  # maximum value on y-axis
yticks = np.arange(-1, 1.1, 0.5)  # define where to put major ticks on y-axis
ylabel = None  # which label to put on y-axis
title = 'AIC'  # title of plot

# make plots
make_four_box_and_swarmplots(x, y, df, ax, ymin, ymax, yticks, xticklabels, ylabel, title, colors)

# Set up plot parameters for sixth panel
#######################################################################################################
ylabeloffset = -7
xlabeloffset = 0
colors = [colors_parent[0], colors_parent[1], colors_parent[3]]  # defines colors for scatterplot

dotsize = 1.8  # size of datapoints in scatterplot
linewidth_sw = 0.3  # linewidth of dots in scatterplot
alpha_sw = 1  # transparency of dots in scatterplot

ax = axes[2, 1]
x = 'actin_angles'
y = 'AIC_baseline'
hue = 'keys'
xmin = 15
xmax = 75
ymin = -1
ymax = 1
xticks = np.arange(15, 75.1, 15)
yticks = np.arange(-1, 1.1, 0.5)
xlabel = "Actin angle"  # "'$\mathrm{\sigma_{x, MSM}}$'
ylabel = "AIC"  # '$\mathrm{\sigma_{x, CM}}$'

corr, p = make_correlationplotsplots(x, y, hue, df, ax, xmin, xmax, ymin, ymax, xticks, yticks, xlabel, ylabel, colors, ylabeloffset=-8)

# add line with slope 1 for visualisation
ax.plot([xmin, xmax], [0, 0], linewidth=0.5, linestyle=':', color='grey')
ax.plot([45, 45], [ymin, ymax], linewidth=0.5, linestyle=':', color='grey')

plt.text(0.21 * xmax, 1.1 * ymax, 'R = ' + str(corr))
# plt.text(0.52 * xmax, 1.1 * ymax, 'p = ' + '{:0.2e}'.format(p))

plt.savefig(figfolder + 'C.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'C.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% filter data to make sure that the baselines are stable
def filter_data_main(data, title):
    # concatenate data on which it will be determined which cells will be filtered
    filterdata = np.stack(
        (data["MSM_data"]["relsigma_xx_left"][0:20, :], data["MSM_data"]["relsigma_xx_right"][0:20, :],
         data["MSM_data"]["relsigma_yy_left"][0:20, :], data["MSM_data"]["relsigma_yy_right"][0:20, :]))

    # move axis of variable to the last position for consistency
    filterdata = np.moveaxis(filterdata, 0, -1)

    # maximal allowed slope for linear fit of baseline
    threshold = 0.005
    baselinefilter = create_filter(filterdata, threshold)

    # remove cells with unstable baselines
    data["TFM_data"] = apply_filter(data["TFM_data"], baselinefilter)
    data["MSM_data"] = apply_filter(data["MSM_data"], baselinefilter)
    data["shape_data"] = apply_filter(data["shape_data"], baselinefilter)

    new_N = np.sum(baselinefilter)
    print(title + ": " + str(baselinefilter.shape[0] - new_N) + " cells were filtered out")

    return data


AR1to2d_halfstim = filter_data_main(AR1to2d_halfstim, "AR1to2d_halfstim")
AR1to1d_halfstim = filter_data_main(AR1to1d_halfstim, "AR1to1d_halfstim")
AR2to1d_halfstim = filter_data_main(AR2to1d_halfstim, "AR2to1d_halfstim")

# %% prepare dataframe again after filtering

# initialize empty dictionaries
concatenated_data_1to2d = {}
concatenated_data_1to1d = {}
concatenated_data_2to1d = {}
concatenated_data = {}

# loop over all keys
for key1 in AR1to1d_halfstim:  # keys are the same for all dictionaries so I'm just taking one example here
    for key2 in AR1to1d_halfstim[key1]:
        if AR1to1d_halfstim[key1][key2].ndim == 1:  # only 1D data can be stored in the data frame
            # concatenate values from different experiments
            concatenated_data_1to2d[key2] = AR1to2d_halfstim[key1][key2]
            concatenated_data_1to1d[key2] = AR1to1d_halfstim[key1][key2]
            concatenated_data_2to1d[key2] = AR2to1d_halfstim[key1][key2]

            concatenated_data[key2] = np.concatenate(
                (concatenated_data_1to2d[key2], concatenated_data_1to1d[key2], concatenated_data_2to1d[key2]))
key2 = 'Es_baseline'
# get number of elements for both condition
n_1to2d = concatenated_data_1to2d[key2].shape[0]
n_1to1d = concatenated_data_1to1d[key2].shape[0]
n_2to1d = concatenated_data_2to1d[key2].shape[0]

# create a list of keys with the same dimensions as the data
keys1to2d = ['AR1to2d' for i in range(n_1to2d)]
keys1to1d = ['AR1to1d' for i in range(n_1to1d)]
keys2to1d = ['AR2to1d' for i in range(n_2to1d)]
keys = np.concatenate((keys1to2d, keys1to1d, keys2to1d))

# add keys to dictionary with concatenated data
concatenated_data['keys'] = keys

# Creates DataFrame
df = pd.DataFrame(concatenated_data)

# convert to more convenient units for plotting
df_plot_units = df  # all units here are in SI units
df_plot_units['Es_baseline'] *= 1e12  # convert to fJ
df_plot_units['spreadingsize_baseline'] *= 1e12  # convert to µm²
df_plot_units['sigma_xx_baseline'] *= 1e3  # convert to mN/m
df_plot_units['sigma_yy_baseline'] *= 1e3  # convert to mN/m

# %% prepare dataframe for boxplots
n_1to2d = AR1to2d_halfstim['MSM_data']['RSI_xx_left'].shape[0]
n_1to1d = AR1to1d_halfstim['MSM_data']['RSI_xx_left'].shape[0]
n_2to1d = AR2to1d_halfstim['MSM_data']['RSI_xx_left'].shape[0]

RSI_data_1to2d = {}
RSI_data_1to1d = {}
RSI_data_2to1d = {}

RSI_data_1to2d['sigma'] = np.concatenate((AR1to2d_halfstim['MSM_data']['RSI_xx_left'],
                                          AR1to2d_halfstim['MSM_data']['RSI_xx_right'],
                                          AR1to2d_halfstim['MSM_data']['RSI_yy_left'],
                                          AR1to2d_halfstim['MSM_data']['RSI_yy_right']))
RSI_data_1to1d['sigma'] = np.concatenate((AR1to1d_halfstim['MSM_data']['RSI_xx_left'],
                                          AR1to1d_halfstim['MSM_data']['RSI_xx_right'],
                                          AR1to1d_halfstim['MSM_data']['RSI_yy_left'],
                                          AR1to1d_halfstim['MSM_data']['RSI_yy_right']))
RSI_data_2to1d['sigma'] = np.concatenate((AR2to1d_halfstim['MSM_data']['RSI_xx_left'],
                                          AR2to1d_halfstim['MSM_data']['RSI_xx_right'],
                                          AR2to1d_halfstim['MSM_data']['RSI_yy_left'],
                                          AR2to1d_halfstim['MSM_data']['RSI_yy_right']))

keys1to2d = np.concatenate((['RSI_xx_left' for i in range(n_1to2d)], ['RSI_xx_right' for i in range(n_1to2d)],
                            ['RSI_yy_left' for i in range(n_1to2d)], ['RSI_yy_right' for i in range(n_1to2d)]))
keys1to1d = np.concatenate((['RSI_xx_left' for i in range(n_1to1d)], ['RSI_xx_right' for i in range(n_1to1d)],
                            ['RSI_yy_left' for i in range(n_1to1d)], ['RSI_yy_right' for i in range(n_1to1d)]))
keys2to1d = np.concatenate((['RSI_xx_left' for i in range(n_2to1d)], ['RSI_xx_right' for i in range(n_2to1d)],
                            ['RSI_yy_left' for i in range(n_2to1d)], ['RSI_yy_right' for i in range(n_2to1d)]))

RSI_data_1to2d['keys'] = keys1to2d
RSI_data_1to1d['keys'] = keys1to1d
RSI_data_2to1d['keys'] = keys2to1d

df1to2d = pd.DataFrame(RSI_data_1to2d)
df1to1d = pd.DataFrame(RSI_data_1to1d)
df2to1d = pd.DataFrame(RSI_data_2to1d)
# %% plot figure 5D, stress map differences

# prepare data first

# concatenate TFM maps from different experiments and calculate average maps over first 20 frames and all cells to get average maps
sigmaxx_1to2d_diff = np.nanmean(
    AR1to2d_halfstim["MSM_data"]["sigma_xx"][:, :, 33, :] - AR1to2d_halfstim["MSM_data"]["sigma_xx"][:, :, 20, :],
    axis=2)
sigmayy_1to2d_diff = np.nanmean(
    AR1to2d_halfstim["MSM_data"]["sigma_yy"][:, :, 33, :] - AR1to2d_halfstim["MSM_data"]["sigma_yy"][:, :, 20, :],
    axis=2)

sigmaxx_1to1d_diff = np.nanmean(
    AR1to1d_halfstim["MSM_data"]["sigma_xx"][:, :, 33, :] - AR1to1d_halfstim["MSM_data"]["sigma_xx"][:, :, 20, :],
    axis=2)
sigmayy_1to1d_diff = np.nanmean(
    AR1to1d_halfstim["MSM_data"]["sigma_yy"][:, :, 33, :] - AR1to1d_halfstim["MSM_data"]["sigma_yy"][:, :, 20, :],
    axis=2)

sigmaxx_2to1d_diff = np.nanmean(
    AR2to1d_halfstim["MSM_data"]["sigma_xx"][:, :, 33, :] - AR2to1d_halfstim["MSM_data"]["sigma_xx"][:, :, 20, :],
    axis=2)
sigmayy_2to1d_diff = np.nanmean(
    AR2to1d_halfstim["MSM_data"]["sigma_yy"][:, :, 33, :] - AR2to1d_halfstim["MSM_data"]["sigma_yy"][:, :, 20, :],
    axis=2)

# crop maps
crop_start = 2
crop_end = 90

sigmaxx_1to2d_diff_crop = sigmaxx_1to2d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3  # convert to mN/m
sigmayy_1to2d_diff_crop = sigmayy_1to2d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3
sigmaxx_1to1d_diff_crop = sigmaxx_1to1d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3  # convert to mN/m
sigmayy_1to1d_diff_crop = sigmayy_1to1d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3
sigmaxx_2to1d_diff_crop = sigmaxx_2to1d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3  # convert to mN/m
sigmayy_2to1d_diff_crop = sigmayy_2to1d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3

# set up plot parameters
# *****************************************************************************

pixelsize = 0.864  # in µm
sigma_max = 1  # kPa
sigma_min = -1  # kPa

# create x- and y-axis for plotting maps
x_end = np.shape(sigmaxx_1to1d_diff_crop)[1]
y_end = np.shape(sigmaxx_1to1d_diff_crop)[0]
extent = [0, x_end * pixelsize, 0, y_end * pixelsize]

# create mesh for vectorplot
xq, yq = np.meshgrid(np.linspace(0, extent[1], x_end), np.linspace(0, extent[3], y_end))

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(3.32, 4))

im = axes[0, 0].imshow(sigmaxx_1to2d_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent,
                       vmin=sigma_min, vmax=sigma_max, aspect='auto')
axes[0, 1].imshow(sigmayy_1to2d_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent,
                  vmin=sigma_min, vmax=sigma_max, aspect='auto')

axes[1, 0].imshow(sigmaxx_1to1d_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent,
                  vmin=sigma_min, vmax=sigma_max, aspect='auto')
axes[1, 1].imshow(sigmayy_1to1d_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent,
                  vmin=sigma_min, vmax=sigma_max, aspect='auto')

axes[2, 0].imshow(sigmaxx_2to1d_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent,
                  vmin=sigma_min, vmax=sigma_max, aspect='auto')
axes[2, 1].imshow(sigmayy_2to1d_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent,
                  vmin=sigma_min, vmax=sigma_max, aspect='auto')

# adjust space in between plots
plt.subplots_adjust(wspace=0, hspace=0)

# axes[0,0].set_xlabel("lol")
# # add annotations
# plt.text(-50,120,'sigmaxx',color = 'k')
# plt.text(20,120,'sigmayy',color = 'k')
# plt.text(-40.5,119,'n=1',color = 'white')
# plt.text(23,55.5,'n=101',color = 'white')
# plt.text(23.5,119,'n=66',color = 'white')

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
plt.suptitle('Cell stresses', y=0.95, x=0.42)
plt.text(-55, 230, 'xx-Stress')
plt.text(20, 230, 'yy-Stress')

# add annotations
plt.text(0.25, 0.853, 'n=' + str(n_1to2d), transform=plt.figure(1).transFigure, color='black')
plt.text(0.25, 0.598, 'n=' + str(n_1to1d), transform=plt.figure(1).transFigure, color='black')
plt.text(0.25, 0.343, 'n=' + str(n_2to1d), transform=plt.figure(1).transFigure, color='black')

plt.text(0.55, 0.853, 'n=' + str(n_1to2d), transform=plt.figure(1).transFigure, color='black')
plt.text(0.55, 0.598, 'n=' + str(n_1to1d), transform=plt.figure(1).transFigure, color='black')
plt.text(0.55, 0.343, 'n=' + str(n_2to1d), transform=plt.figure(1).transFigure, color='black')

# save figure
fig.savefig(figfolder + 'D.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'D.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 5E time and boxplots for stresses

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.arange(60)
x = x[::2]  # downsample data for nicer plotting
ymin = -0.1
ymax = 0.2
xticks = np.arange(0, 61, 20)  # define where the major ticks are gonna be
yticks = np.arange(ymin, ymax + 0.01, 0.1)
xlabel = 'time [min]'
xticklabels = ['left \n         $\mathrm{\sigma _ {xx}}$', 'right', 'left \n         $\mathrm{\sigma _ {yy}}$',
               'right']  # which labels to put on x-axis
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(4, 4))  # create figure and axes
plt.subplots_adjust(wspace=0.35, hspace=0.35)  # adjust space in between plots

# Set up plot parameters for first panel
#######################################################################################################
ax = axes[0, 0]
colors = [colors_parent[0], colors_parent_dark[0]]
ylabel = '1to2'
title = 'xx-Stress'
y1 = AR1to2d_halfstim["MSM_data"]["relsigma_xx_left"]
y2 = AR1to2d_halfstim["MSM_data"]["relsigma_xx_right"]
y1 = y1[::2, :]
y2 = y2[::2, :]

# make plots
plot_two_values_over_time(x, y1, y2, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors)

# Set up plot parameters for second panel
#######################################################################################################
ax = axes[1, 0]
colors = [colors_parent[1], colors_parent_dark[1]]
ylabel = '1to1'
title = None
y1 = AR1to1d_halfstim["MSM_data"]["relsigma_xx_left"]
y2 = AR1to1d_halfstim["MSM_data"]["relsigma_xx_right"]
y1 = y1[::2, :]
y2 = y2[::2, :]

# make plots
plot_two_values_over_time(x, y1, y2, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors)

# ax.plot(sim_relstress_xx_left_1to1dhs, color=colors[0])
# ax.plot(sim_relstress_xx_right_1to1dhs, color=colors[0])

# Set up plot parameters for third panel
#######################################################################################################
ax = axes[2, 0]
colors = [colors_parent[3], colors_parent_dark[3]]
ylabel = '2to1'
title = None
y1 = AR2to1d_halfstim["MSM_data"]["relsigma_xx_left"]
y2 = AR2to1d_halfstim["MSM_data"]["relsigma_xx_right"]
y1 = y1[::2, :]
y2 = y2[::2, :]

# make plots
plot_two_values_over_time(x, y1, y2, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors)

# Set up plot parameters for fourth panel
#######################################################################################################
ax = axes[0, 1]
colors = [colors_parent[0], colors_parent_dark[0]]
ylabel = None
title = 'yy-Stress'
y1 = AR1to2d_halfstim["MSM_data"]["relsigma_yy_left"]
y2 = AR1to2d_halfstim["MSM_data"]["relsigma_yy_right"]
y1 = y1[::2, :]
y2 = y2[::2, :]

# make plots
plot_two_values_over_time(x, y1, y2, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors)

# Set up plot parameters for fifth panel
#######################################################################################################
ax = axes[1, 1]
colors = [colors_parent[1], colors_parent_dark[1]]
ylabel = None
title = None
y1 = AR1to1d_halfstim["MSM_data"]["relsigma_yy_left"]
y2 = AR1to1d_halfstim["MSM_data"]["relsigma_yy_right"]
y1 = y1[::2, :]
y2 = y2[::2, :]

# make plots
plot_two_values_over_time(x, y1, y2, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors)

# Set up plot parameters for sixth panel
#######################################################################################################
ax = axes[2, 1]
colors = [colors_parent[3], colors_parent_dark[3]]
ylabel = None
title = None
y1 = AR2to1d_halfstim["MSM_data"]["relsigma_yy_left"]
y2 = AR2to1d_halfstim["MSM_data"]["relsigma_yy_right"]
y1 = y1[::2, :]
y2 = y2[::2, :]

# make plots
plot_two_values_over_time(x, y1, y2, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors)

# Set up plot parameters for seventh panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'sigma'  # variable that goes on the y-axis
ax = axes[0, 2]  # define on which axis the plot goes
colors = [colors_parent[0], colors_parent_dark[0], colors_parent[0], colors_parent_dark[0]]  # defines colors
ymin = -0.2  # minimum value on y-axis
ymax = 0.4  # maximum value on y-axis
yticks = np.arange(-0.2, 0.41, 0.2)  # define where to put major ticks on y-axis
ylabel = None  # which label to put on y-axis
title = None  # title of plot

# make plots
make_four_box_and_swarmplots(x, y, df1to2d, ax, ymin, ymax, yticks, xticklabels, ylabel, title, colors)

# Set up plot parameters for eighth panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'sigma'  # variable that goes on the y-axis
ax = axes[1, 2]  # define on which axis the plot goes
colors = [colors_parent[1], colors_parent_dark[1], colors_parent[1], colors_parent_dark[1]]  # defines colors
ymin = -0.2  # minimum value on y-axis
ymax = 0.4  # maximum value on y-axis
yticks = np.arange(-0.2, 0.41, 0.2)  # define where to put major ticks on y-axis
ylabel = None  # which label to put on y-axis
title = None  # title of plot

# make plots
make_four_box_and_swarmplots(x, y, df1to1d, ax, ymin, ymax, yticks, xticklabels, ylabel, title, colors)

# Set up plot parameters for ninth panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'sigma'  # variable that goes on the y-axis
ax = axes[2, 2]  # define on which axis the plot goes
colors = [colors_parent[3], colors_parent_dark[3], colors_parent[3], colors_parent_dark[3]]  # defines colors
ylabel = None  # which label to put on y-axis
title = None  # title of plot

# make plots
make_four_box_and_swarmplots(x, y, df2to1d, ax, ymin, ymax, yticks, xticklabels, ylabel, title, colors)

# write title for panels 1 to 4
plt.text(-11.8, 2.2, 'Relative stresses, local activation', fontsize=10)
# write title for panels 5 to 6
plt.text(-0.65, 2.08, 'Relative stress \n     increase', fontsize=10)

plt.savefig(figfolder + 'E.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'E.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 5X

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.linspace(-25, 25, 82)
x = x[::4]  # downsample data for nicer plotting
xticks = np.arange(-15, 15.1, 15)  # define where the major ticks are gonna be
xlabel = 'position [µm]'
xticklabels = ['global \n act.', 'local \n act.']  # which labels to put on x-axis
ymin = 0
ymax = 4
yticks = np.arange(ymin, ymax + 0.001, 1)
windowlength = 41
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(5, 4))  # create figure and axes
plt.subplots_adjust(wspace=0.4, hspace=0.35)  # adjust space in between plots
# ******************************************************************************************************************************************

# Set up plot parameters for first panel
#######################################################################################################
ax = axes[0, 0]
color = colors_parent[0]
ylabel = '1to2'
title = '$\mathrm{\sigma _{normal}(x)}$ baseline [nN]'
y = AR1to2d_halfstim["MSM_data"]["sigma_normal_x_profile_baseline"] * 1e3  #convert to nN

# crop a centered, 82 pixel wide window
pos_center = np.rint(y.shape[0] / 2).astype(int)
y = y[pos_center - windowlength:pos_center + windowlength]
y = y[::4, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False)

# # Set up plot parameters for second panel
# #######################################################################################################
ax = axes[1, 0]
color = colors_parent[1]
ylabel = '1to1'
title = None
y = AR1to1d_halfstim["MSM_data"]["sigma_normal_x_profile_baseline"] * 1e3  #convert to nN

# crop a centered, 82 pixel wide window
pos_center = np.rint(y.shape[0] / 2).astype(int)
y = y[pos_center - windowlength:pos_center + windowlength]
y = y[::4, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False)


# # Set up plot parameters for third panel
# #######################################################################################################
ax = axes[2, 0]
color = colors_parent[3]
ylabel = '2to1'
title = None
y = AR2to1d_halfstim["MSM_data"]["sigma_normal_x_profile_baseline"] * 1e3  #convert to nN

# crop a centered, 82 pixel wide window
pos_center = np.rint(y.shape[0] / 2).astype(int)
y = y[pos_center - windowlength:pos_center + windowlength]
y = y[::4, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False)

# # Set up plot parameters for fourth panel
# #######################################################################################################
ax = axes[0, 1]
color = colors_parent[0]
ymin = -0.1
ymax = 0.1
yticks = np.arange(ymin, ymax + 0.001, 0.1)
ylabel = 'doublet'
title = '$\mathrm{\sigma _{rel, normal}(x)}$ increase'
y = AR1to2d_halfstim["MSM_data"]["relsigma_normal_x_profile_increase"]

# crop a centered, 82 pixel wide window
pos_center = np.rint(y.shape[0] / 2).astype(int)
y = y[pos_center - windowlength:pos_center + windowlength]
y = y[::4, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False)

# # Set up plot parameters for fifth panel
# #######################################################################################################
ax = axes[1, 1]
color = colors_parent[1]
ymin = -0.1
ymax = 0.1
yticks = np.arange(ymin, ymax + 0.001, 0.1)
ylabel = None
title = None
y = AR1to1d_halfstim["MSM_data"]["relsigma_normal_x_profile_increase"]

# crop a centered, 82 pixel wide window
pos_center = np.rint(y.shape[0] / 2).astype(int)
y = y[pos_center - windowlength:pos_center + windowlength]
y = y[::4, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False)

# # Set up plot parameters for sixth panel
# # #######################################################################################################
ax = axes[2, 1]
color = colors_parent[3]
ymin = -0.1
ymax = 0.1
yticks = np.arange(ymin, ymax + 0.001, 0.1)
ylabel = None
title = None
y = AR2to1d_halfstim["MSM_data"]["relsigma_normal_x_profile_increase"]

# crop a centered, 82 pixel wide window
pos_center = np.rint(y.shape[0] / 2).astype(int)
y = y[pos_center - windowlength:pos_center + windowlength]
y = y[::4, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False)

plt.savefig(figfolder + 'X.png', dpi=300, bbox_inches="tight")
# plt.savefig(figfolder + 'F.svg', dpi=300, bbox_inches="tight")
plt.show()

# # %% plot figure 5F correlation plots
#
# # define plot parameters that are valid for the whole figure
# # ******************************************************************************************************************************************
# colors = [colors_parent[0], colors_parent[1], colors_parent[3]]  # defines colors for scatterplot
# fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(9, 2))  # create figure instance
# plt.subplots_adjust(wspace=0.3, hspace=0.3)  # adjust space in between plots
#
# # Set up plot parameters for first panel
# #######################################################################################################
# x = 'AIC_baseline'
# y = 'RSI_xx_left'
# hue = 'keys'
# ax = axes[0]
# xmin = -1
# xmax = 1
# ymin = -0.2
# ymax = 0.4
# xticks = np.arange(-1, 1.1, 0.5)
# yticks = np.arange(-0.2, 0.21, 0.1)
# xlabel = 'Anisotropy coefficient'
# ylabel = '$\mathrm{RSI_{xx, left}}$'
#
# corr, p = make_correlationplotsplots(x, y, hue, df, ax, xmin, xmax, ymin, ymax, xticks, yticks, xlabel, ylabel, colors)
#
# # annotate pearson R and p-value
# plt.text(xmin + 0.1 * xmax, ymin + 0.175 * ymax, 'R = ' + str(corr))
# plt.text(xmin + 0.1 * xmax, ymin + 0.075 * ymax, 'p = ' + '{:0.2e}'.format(p))
#
# # Set up plot parameters for second panel
# #######################################################################################################
# x = 'AIC_baseline'
# y = 'RSI_yy_left'
# hue = 'keys'
# ax = axes[1]
# xmin = -1
# xmax = 1
# ymin = -0.2
# ymax = 0.4
# xticks = np.arange(-1, 1.1, 0.5)
# yticks = np.arange(-0.2, 0.21, 0.1)
# xlabel = 'Anisotropy coefficient'
# ylabel = '$\mathrm{RSI_{yy, left}}$'
#
# corr, p = make_correlationplotsplots(x, y, hue, df, ax, xmin, xmax, ymin, ymax, xticks, yticks, xlabel, ylabel, colors)
#
# # annotate pearson R and p-value
# plt.text(xmin + 0.1 * xmax, ymin + 0.175 * ymax, 'R = ' + str(corr))
# plt.text(xmin + 0.1 * xmax, ymin + 0.075 * ymax, 'p = ' + '{:0.2e}'.format(p))
#
# # Set up plot parameters for third panel
# #######################################################################################################
# x = 'AIC_baseline'
# y = 'RSI_xx_right'
# hue = 'keys'
# ax = axes[2]
# xmin = -1
# xmax = 1
# ymin = -0.2
# ymax = 0.4
# xticks = np.arange(-1, 1.1, 0.5)
# yticks = np.arange(-0.2, 0.21, 0.1)
# xlabel = 'Anisotropy coefficient'
# ylabel = '$\mathrm{RSI_{xx, right}}$'
#
# corr, p = make_correlationplotsplots(x, y, hue, df, ax, xmin, xmax, ymin, ymax, xticks, yticks, xlabel, ylabel, colors)
#
# # annotate pearson R and p-value
# plt.text(xmin + 0.1 * xmax, ymin + 0.175 * ymax, 'R = ' + str(corr))
# plt.text(xmin + 0.1 * xmax, ymin + 0.075 * ymax, 'p = ' + '{:0.2e}'.format(p))
#
# # Set up plot parameters for fourth panel
# #######################################################################################################
# x = 'AIC_baseline'
# y = 'RSI_yy_right'
# hue = 'keys'
# ax = axes[3]
# xmin = -1
# xmax = 1
# ymin = -0.2
# ymax = 0.4
# xticks = np.arange(-1, 1.1, 0.5)
# yticks = np.arange(-0.2, 0.21, 0.1)
# xlabel = 'Anisotropy coefficient'
# ylabel = '$\mathrm{RSI_{yy, right}}$'
#
# corr, p = make_correlationplotsplots(x, y, hue, df, ax, xmin, xmax, ymin, ymax, xticks, yticks, xlabel, ylabel, colors)
#
# # annotate pearson R and p-value
# plt.text(xmin + 0.1 * xmax, ymin + 0.175 * ymax, 'R = ' + str(corr))
# plt.text(xmin + 0.1 * xmax, ymin + 0.075 * ymax, 'p = ' + '{:0.2e}'.format(p))
#
# plt.savefig(figfolder + 'F.png', dpi=300, bbox_inches="tight")
# plt.savefig(figfolder + 'F.svg', dpi=300, bbox_inches="tight")
# plt.show()
