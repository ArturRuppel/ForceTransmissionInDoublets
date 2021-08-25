# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 21:56:01 2021

@author: Artur Ruppel

"""
import os
import pickle
import pandas as pd

from plot_and_filter_functions import *

# mpl.rcParams['pdf.fonttype'] = 42


# chose an example for force maps
doublet_example = 1
singlet_example = 4

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

figfolder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/_Figure1/"
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

# %% plot figure 1C, force maps

# prepare data first

# concatenate TFM maps from different experiments and calculate average maps over first 20 frames and all cells to get average maps
Tx_1to1d_average = np.nanmean(np.concatenate((AR1to1d_halfstim["TFM_data"]["Tx"][:, :, 0:20, :],
                                              AR1to1d_fullstim_short["TFM_data"]["Tx"][:, :, 0:20, :],
                                              AR1to1d_fullstim_long["TFM_data"]["Tx"][:, :, 0:20, :]), axis=3),
                              axis=(2, 3))
Ty_1to1d_average = np.nanmean(np.concatenate((AR1to1d_halfstim["TFM_data"]["Ty"][:, :, 0:20, :],
                                              AR1to1d_fullstim_short["TFM_data"]["Ty"][:, :, 0:20, :],
                                              AR1to1d_fullstim_long["TFM_data"]["Ty"][:, :, 0:20, :]), axis=3),
                              axis=(2, 3))
Tx_1to1s_average = np.nanmean(np.concatenate((AR1to1s_halfstim["TFM_data"]["Tx"][:, :, 0:20, :],
                                              AR1to1s_fullstim_short["TFM_data"]["Tx"][:, :, 0:20, :],
                                              AR1to1s_fullstim_long["TFM_data"]["Tx"][:, :, 0:20, :]), axis=3),
                              axis=(2, 3))
Ty_1to1s_average = np.nanmean(np.concatenate((AR1to1s_halfstim["TFM_data"]["Ty"][:, :, 0:20, :],
                                              AR1to1s_fullstim_short["TFM_data"]["Ty"][:, :, 0:20, :],
                                              AR1to1s_fullstim_long["TFM_data"]["Ty"][:, :, 0:20, :]), axis=3),
                              axis=(2, 3))

# get one example
Tx_1to1d_example = AR1to1d_halfstim["TFM_data"]["Tx"][:, :, 0, doublet_example]
Ty_1to1d_example = AR1to1d_halfstim["TFM_data"]["Ty"][:, :, 0, doublet_example]
Tx_1to1s_example = AR1to1s_halfstim["TFM_data"]["Tx"][:, :, 0, singlet_example]
Ty_1to1s_example = AR1to1s_halfstim["TFM_data"]["Ty"][:, :, 0, singlet_example]

# calculate amplitudes
T_1to1d_average = np.sqrt(Tx_1to1d_average ** 2 + Ty_1to1d_average ** 2)
T_1to1s_average = np.sqrt(Tx_1to1s_average ** 2 + Ty_1to1s_average ** 2)
T_1to1d_example = np.sqrt(Tx_1to1d_example ** 2 + Ty_1to1d_example ** 2)
T_1to1s_example = np.sqrt(Tx_1to1s_example ** 2 + Ty_1to1s_example ** 2)

# crop maps 
crop_start = 8
crop_end = 84

Tx_1to1d_average_crop = Tx_1to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e-3  # convert to kPa
Ty_1to1d_average_crop = Ty_1to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e-3
T_1to1d_average_crop = T_1to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e-3

Tx_1to1d_example_crop = Tx_1to1d_example[crop_start:crop_end, crop_start:crop_end] * 1e-3
Ty_1to1d_example_crop = Ty_1to1d_example[crop_start:crop_end, crop_start:crop_end] * 1e-3
T_1to1d_example_crop = T_1to1d_example[crop_start:crop_end, crop_start:crop_end] * 1e-3

Tx_1to1s_average_crop = Tx_1to1s_average[crop_start:crop_end, crop_start:crop_end] * 1e-3
Ty_1to1s_average_crop = Ty_1to1s_average[crop_start:crop_end, crop_start:crop_end] * 1e-3
T_1to1s_average_crop = T_1to1s_average[crop_start:crop_end, crop_start:crop_end] * 1e-3

Tx_1to1s_example_crop = Tx_1to1s_example[crop_start:crop_end, crop_start:crop_end] * 1e-3
Ty_1to1s_example_crop = Ty_1to1s_example[crop_start:crop_end, crop_start:crop_end] * 1e-3
T_1to1s_example_crop = T_1to1s_example[crop_start:crop_end, crop_start:crop_end] * 1e-3

# set up plot parameters
# ******************************************************************************************************************************************
n = 4                           # every nth arrow will be plotted
pixelsize = 0.864               # in µm
pmax = 2                        # in kPa
axtitle = 'kPa'                 # unit of colorbar
suptitle = 'Traction forces'    # title of plot
x_end = np.shape(T_1to1d_average_crop)[1]   # create x- and y-axis for plotting maps
y_end = np.shape(T_1to1d_average_crop)[0]
extent = [0, x_end * pixelsize, 0, y_end * pixelsize]
xq, yq = np.meshgrid(np.linspace(0, extent[1], x_end), np.linspace(0, extent[3], y_end))  # create mesh for vectorplot
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(3, 2.5))    # create figure and axes
plt.subplots_adjust(wspace=0.02, hspace=-0.06)      # adjust space in between plots
# ******************************************************************************************************************************************

im = axes[0, 0].imshow(T_1to1d_example_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent,
                       vmin=0, vmax=pmax, aspect='auto')
axes[0, 0].quiver(xq[::n, ::n], yq[::n, ::n], Tx_1to1d_example_crop[::n, ::n], Ty_1to1d_example_crop[::n, ::n],
                  angles='xy', scale=10, units='width', color="r")
# axes[0,0].set_title('n=1', pad=-400, color='r')

axes[0, 1].imshow(T_1to1d_average_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0,
                  vmax=pmax, aspect='auto')
axes[0, 1].quiver(xq[::n, ::n], yq[::n, ::n], Tx_1to1d_average_crop[::n, ::n], Ty_1to1d_average_crop[::n, ::n],
                  angles='xy', scale=10, units='width', color="r")

axes[1, 0].imshow(T_1to1s_example_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0,
                  vmax=pmax, aspect='auto')
axes[1, 0].quiver(xq[::n, ::n], yq[::n, ::n], Tx_1to1s_example_crop[::n, ::n], Ty_1to1s_example_crop[::n, ::n],
                  angles='xy', scale=10, units='width', color="r")

axes[1, 1].imshow(T_1to1s_average_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0,
                  vmax=pmax, aspect='auto')
axes[1, 1].quiver(xq[::n, ::n], yq[::n, ::n], Tx_1to1s_average_crop[::n, ::n], Ty_1to1s_average_crop[::n, ::n],
                  angles='xy', scale=10, units='width', color="r")

# remove axes
for ax in axes.flat:
    ax.axis('off')
    aspectratio = 1.0
    ratio_default = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.set_aspect(ratio_default * aspectratio)

# add colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist())
cbar.ax.set_title(axtitle)

# add title
plt.suptitle(suptitle, y=0.94, x=0.44)

# add annotations
plt.text(0.25, 0.83, 'n=1', transform=plt.figure(1).transFigure, color='w')
plt.text(0.25, 0.455, 'n=1', transform=plt.figure(1).transFigure, color='w')
plt.text(0.55, 0.455, 'n=' + str(n_singlets), transform=plt.figure(1).transFigure, color='w')
plt.text(0.55, 0.83, 'n=' + str(n_doublets), transform=plt.figure(1).transFigure, color='w')

fig.savefig(figfolder + 'C.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'C.svg', dpi=300, bbox_inches="tight")
plt.show()
# %% plot figure 1D_1, xx-stress maps

# prepare data first

# concatenate MSM maps from different experiments and calculate average maps over first 20 frames and all cells to get average maps
sigma_xx_1to1d_average = np.nanmean(np.concatenate((AR1to1d_halfstim["MSM_data"]["sigma_xx"][:, :, 0:20, :],
                                                    AR1to1d_fullstim_short["MSM_data"]["sigma_xx"][:, :, 0:20, :],
                                                    AR1to1d_fullstim_long["MSM_data"]["sigma_xx"][:, :, 0:20, :]),
                                                   axis=3), axis=(2, 3))
sigma_xx_1to1s_average = np.nanmean(np.concatenate((AR1to1s_halfstim["MSM_data"]["sigma_xx"][:, :, 0:20, :],
                                                    AR1to1s_fullstim_short["MSM_data"]["sigma_xx"][:, :, 0:20, :],
                                                    AR1to1s_fullstim_long["MSM_data"]["sigma_xx"][:, :, 0:20, :]),
                                                   axis=3), axis=(2, 3))

# get one example
sigma_xx_1to1d_example = AR1to1d_halfstim["MSM_data"]["sigma_xx"][:, :, 0, doublet_example]
sigma_xx_1to1s_example = AR1to1s_halfstim["MSM_data"]["sigma_xx"][:, :, 0, singlet_example]

# convert NaN to 0 to have black background
sigma_xx_1to1d_average[np.isnan(sigma_xx_1to1d_average)] = 0
sigma_xx_1to1s_average[np.isnan(sigma_xx_1to1s_average)] = 0
sigma_xx_1to1d_example[np.isnan(sigma_xx_1to1d_example)] = 0
sigma_xx_1to1s_example[np.isnan(sigma_xx_1to1s_example)] = 0

# crop maps 
crop_start = 8
crop_end = 84

sigma_xx_1to1d_average_crop = sigma_xx_1to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e3  # convert to mN/m
sigma_xx_1to1d_example_crop = sigma_xx_1to1d_example[crop_start:crop_end, crop_start:crop_end] * 1e3
sigma_xx_1to1s_average_crop = sigma_xx_1to1s_average[crop_start:crop_end, crop_start:crop_end] * 1e3
sigma_xx_1to1s_example_crop = sigma_xx_1to1s_example[crop_start:crop_end, crop_start:crop_end] * 1e3

# set up plot parameters
# ******************************************************************************************************************************************
pixelsize = 0.864               # in µm
pmax = 10                       # in mN/m
axtitle = 'mN/m'                # unit of colorbar
suptitle = 'xx-Stress'          # title of plot
x_end = np.shape(T_1to1d_average_crop)[1]   # create x- and y-axis for plotting maps
y_end = np.shape(T_1to1d_average_crop)[0]
extent = [0, x_end * pixelsize, 0, y_end * pixelsize]
xq, yq = np.meshgrid(np.linspace(0, extent[1], x_end), np.linspace(0, extent[3], y_end))  # create mesh for vectorplot
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(3, 2.5))    # create figure and axes
plt.subplots_adjust(wspace=0.02, hspace=-0.06)      # adjust space in between plots
# ******************************************************************************************************************************************


im = axes[0, 0].imshow(sigma_xx_1to1d_example_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent,
                       vmin=0, vmax=pmax, aspect='auto')

axes[0, 1].imshow(sigma_xx_1to1d_average_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent,
                  vmin=0, vmax=pmax, aspect='auto')

axes[1, 0].imshow(sigma_xx_1to1s_example_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent,
                  vmin=0, vmax=pmax, aspect='auto')

axes[1, 1].imshow(sigma_xx_1to1s_average_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent,
                  vmin=0, vmax=pmax, aspect='auto')


# remove axes
for ax in axes.flat:
    ax.axis('off')
    aspectratio = 1.0
    ratio_default = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.set_aspect(ratio_default * aspectratio)

# add colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist())
cbar.ax.set_title(axtitle)

# add title
plt.suptitle(suptitle, y=0.94, x=0.44)

# add annotations
plt.text(0.25, 0.83, 'n=1', transform=plt.figure(1).transFigure, color='w')
plt.text(0.25, 0.455, 'n=1', transform=plt.figure(1).transFigure, color='w')
plt.text(0.55, 0.455, 'n=' + str(n_singlets), transform=plt.figure(1).transFigure, color='w')
plt.text(0.55, 0.83, 'n=' + str(n_doublets), transform=plt.figure(1).transFigure, color='w')

# save figure
fig.savefig(figfolder + 'D1.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'D1.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 1D_2, yy-stress maps

# prepare data first

# concatenate MSM maps from different experiments and calculate average maps over first 20 frames and all cells to get average maps
sigma_yy_1to1d_average = np.nanmean(np.concatenate((AR1to1d_halfstim["MSM_data"]["sigma_yy"][:, :, 0:20, :],
                                                    AR1to1d_fullstim_short["MSM_data"]["sigma_yy"][:, :, 0:20, :],
                                                    AR1to1d_fullstim_long["MSM_data"]["sigma_yy"][:, :, 0:20, :]),
                                                   axis=3), axis=(2, 3))
sigma_yy_1to1s_average = np.nanmean(np.concatenate((AR1to1s_halfstim["MSM_data"]["sigma_yy"][:, :, 0:20, :],
                                                    AR1to1s_fullstim_short["MSM_data"]["sigma_yy"][:, :, 0:20, :],
                                                    AR1to1s_fullstim_long["MSM_data"]["sigma_yy"][:, :, 0:20, :]),
                                                   axis=3), axis=(2, 3))

# get one example
sigma_yy_1to1d_example = AR1to1d_halfstim["MSM_data"]["sigma_yy"][:, :, 0, doublet_example]
sigma_yy_1to1s_example = AR1to1s_halfstim["MSM_data"]["sigma_yy"][:, :, 0, singlet_example]

# convert NaN to 0 to have black background
sigma_yy_1to1d_average[np.isnan(sigma_yy_1to1d_average)] = 0
sigma_yy_1to1s_average[np.isnan(sigma_yy_1to1s_average)] = 0
sigma_yy_1to1d_example[np.isnan(sigma_yy_1to1d_example)] = 0
sigma_yy_1to1s_example[np.isnan(sigma_yy_1to1s_example)] = 0

# crop maps 
crop_start = 8
crop_end = 84

sigma_yy_1to1d_average_crop = sigma_yy_1to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e3
sigma_yy_1to1d_example_crop = sigma_yy_1to1d_example[crop_start:crop_end, crop_start:crop_end] * 1e3
sigma_yy_1to1s_average_crop = sigma_yy_1to1s_average[crop_start:crop_end, crop_start:crop_end] * 1e3
sigma_yy_1to1s_example_crop = sigma_yy_1to1s_example[crop_start:crop_end, crop_start:crop_end] * 1e3

# set up plot parameters
# ******************************************************************************************************************************************
pixelsize = 0.864               # in µm
pmax = 10                       # in mN/m
axtitle = 'mN/m'                # unit of colorbar
suptitle = 'yy-Stress'          # title of plot
x_end = np.shape(T_1to1d_average_crop)[1]   # create x- and y-axis for plotting maps
y_end = np.shape(T_1to1d_average_crop)[0]
extent = [0, x_end * pixelsize, 0, y_end * pixelsize]
xq, yq = np.meshgrid(np.linspace(0, extent[1], x_end), np.linspace(0, extent[3], y_end))  # create mesh for vectorplot

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(3, 2.5))    # create figure and axes
plt.subplots_adjust(wspace=0.02, hspace=-0.06)      # adjust space in between plots
# ******************************************************************************************************************************************


im = axes[0, 0].imshow(sigma_yy_1to1d_example_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent,
                       vmin=0, vmax=pmax, aspect='auto')

axes[0, 1].imshow(sigma_yy_1to1d_average_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent,
                  vmin=0, vmax=pmax, aspect='auto')

axes[1, 0].imshow(sigma_yy_1to1s_example_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent,
                  vmin=0, vmax=pmax, aspect='auto')

axes[1, 1].imshow(sigma_yy_1to1s_average_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent,
                  vmin=0, vmax=pmax, aspect='auto')


# remove axes
for ax in axes.flat:
    ax.axis('off')
    aspectratio = 1.0
    ratio_default = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.set_aspect(ratio_default * aspectratio)

# add colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist())
cbar.ax.set_title(axtitle)

# add title
plt.suptitle(suptitle, y=0.94, x=0.44)

# add annotations
plt.text(0.25, 0.83, 'n=1', transform=plt.figure(1).transFigure, color='w')
plt.text(0.25, 0.455, 'n=1', transform=plt.figure(1).transFigure, color='w')
plt.text(0.55, 0.455, 'n=' + str(n_singlets), transform=plt.figure(1).transFigure, color='w')
plt.text(0.55, 0.83, 'n=' + str(n_doublets), transform=plt.figure(1).transFigure, color='w')

# save figure
fig.savefig(figfolder + 'D2.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'D2.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 1E boxplots

# define plot parameters that are valid for the whole figure
# ******************************************************************************************************************************************
colors = [colors_parent[1], colors_parent[2]]               # defines colors
box_pairs = [('AR1to1d', 'AR1to1s')]                        # which groups to perform statistical test on
xticklabels = ['Doublet', 'Singlet']                        # which labels to put on x-axis
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(9, 2))  # create figure instance
plt.subplots_adjust(wspace=0.5, hspace=0)                   # adjust space in between plots
# ******************************************************************************************************************************************


# Set up plot parameters for first panel
#######################################################################################################
x = 'keys'                              # variable by which to group the data
y = 'spreadingsize_baseline'            # variable that goes on the y-axis
ax = axes[0]                            # define on which axis the plot goes
ymin = 500                              # minimum value on y-axis
ymax = 2000                             # maximum value on y-axis
yticks = np.arange(500, 2001, 250)      # define where to put major ticks on y-axis
stat_annotation_offset = 0.22           # vertical offset of statistical annotation
ylabel = 'A [$\mathrm{\mu m^2}$]'       # which label to put on y-axis
title = 'Spreading size'                # title of plot

# make plots
make_box_and_swarmplots_with_test(x, y, df, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title, colors)

# Set up plot parameters for second panel
#######################################################################################################
x = 'keys'                              # variable by which to group the data
y = 'Es_baseline'                       # variable that goes on the y-axis
ax = axes[1]                            # define on which axis the plot goes
ymin = 0                                # minimum value on y-axis
ymax = 2                                # maximum value on y-axis
yticks = np.arange(0, 2.1, 0.5)         # define where to put major ticks on y-axis
stat_annotation_offset = -0.09          # vertical offset of statistical annotation
ylabel = '$\mathrm{E_s}$ [pJ]'          # which label to put on y-axis
title = 'Strain energy'                 # title of plot

# make plots
make_box_and_swarmplots_with_test(x, y, df, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title, colors)


# Set up plot parameters for third panel
#######################################################################################################
x = 'keys'                                  # variable by which to group the data
y = 'sigma_xx_baseline'                     # variable that goes on the y-axis
ax = axes[2]                                # define on which axis the plot goes
ymin = 0                                    # minimum value on y-axis
ymax = 14                                   # maximum value on y-axis
yticks = np.arange(0, 15, 2)                # define where to put major ticks on y-axis
stat_annotation_offset = -0.07              # vertical offset of statistical annotation
ylabel = '$\mathrm{\sigma _{xx}}$ [mN/m]'   # which label to put on y-axis
title = 'xx-Stress'                         # title of plot

# make plots
make_box_and_swarmplots_with_test(x, y, df, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title, colors)


# Set up plot parameters for fourth panel
#######################################################################################################
x = 'keys'                                  # variable by which to group the data
y = 'sigma_yy_baseline'                     # variable that goes on the y-axis
ax = axes[3]                                # define on which axis the plot goes
ymin = 0                                    # minimum value on y-axis
ymax = 14                                   # maximum value on y-axis
yticks = np.arange(0, 15, 2)                # define where to put major ticks on y-axis
stat_annotation_offset = 0.98               # vertical offset of statistical annotation
ylabel = '$\mathrm{\sigma _{yy}}$ [mN/m]'   # which label to put on y-axis
title = 'yy-Stress'                         # title of plot

# make plots
make_box_and_swarmplots_with_test(x, y, df, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title, colors)

# Set up plot parameters for fifth panel
#######################################################################################################
x = 'keys'                              # variable by which to group the data
y = 'AIC_baseline'                      # variable that goes on the y-axis
ax = axes[4]                            # define on which axis the plot goes
ymin = -1                               # minimum value on y-axis
ymax = 1                                # maximum value on y-axis
yticks = np.arange(-1, 1.1, 0.5)        # define where to put major ticks on y-axis
stat_annotation_offset = 0.03           # vertical offset of statistical annotation
ylabel = 'AIC'                          # which label to put on y-axis
title = 'Anisotropy coefficient'        # title of plot
ylabeloffset = -5                      # adjusts distance of ylabel to the plot

# make plots
make_box_and_swarmplots_with_test(x, y, df, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title, colors)

# save plot to file
plt.savefig(figfolder + 'E.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'E.svg', dpi=300, bbox_inches="tight")
plt.show()