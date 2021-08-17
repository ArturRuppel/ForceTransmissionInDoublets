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

# define some colors for the plots
colors_parent = ['#026473', '#E3CC69', '#77C8A6', '#D96248']
colors_parent_dark = ['#01353D', '#564910', '#235741', '#A93B23']

# %% load data for plotting
folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"

AR1to1d_fullstim_long = pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_long.dat", "rb"))
AR1to1s_fullstim_long = pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_long.dat", "rb"))
# AR1to1d_fullstim_short =  pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_short.dat", "rb"))
# AR1to1s_fullstim_short =  pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_short.dat", "rb"))
# AR1to2d_halfstim =        pickle.load(open(folder + "analysed_data/AR1to2d_halfstim.dat", "rb"))
AR1to1d_halfstim = pickle.load(open(folder + "analysed_data/AR1to1d_halfstim.dat", "rb"))
AR1to1s_halfstim = pickle.load(open(folder + "analysed_data/AR1to1s_halfstim.dat", "rb"))
# AR2to1d_halfstim =        pickle.load(open(folder + "analysed_data/AR2to1d_halfstim.dat", "rb"))

# load simulation data for fullstim experiment
sim_Es_1to1dfs = np.load(folder + "AR1to1_doublets_full_stim_long/simulation_strain_energy.npz")["energy"]
sim_stress_xx_1to1dfs = np.load(folder + "AR1to1_doublets_full_stim_long/simulation_stress_xx_yy.npz")["stress_xx"]
sim_stress_yy_1to1dfs = np.load(folder + "AR1to1_doublets_full_stim_long/simulation_stress_xx_yy.npz")["stress_yy"]

sim_stress_xx_1to1dfs = np.nanmean(np.absolute(sim_stress_xx_1to1dfs), axis=1)
sim_stress_yy_1to1dfs = np.nanmean(np.absolute(sim_stress_yy_1to1dfs), axis=1)

# normalize curves for plotting
sim_relEs_1to1dfs = sim_Es_1to1dfs / np.nanmean(sim_Es_1to1dfs[0:20]) - 1
sim_relstress_xx_1to1dfs = sim_stress_xx_1to1dfs / np.nanmean(sim_stress_xx_1to1dfs[0:20]) - 1
sim_relstress_yy_1to1dfs = sim_stress_yy_1to1dfs / np.nanmean(sim_stress_yy_1to1dfs[0:20]) - 1

# load simulation data for halfstim experiment
sim_stress_xx_left_1to1dhs = np.load(folder + "AR1to1_doublets_half_stim/simulation_stress_xx_yy.npz")["stress_xx_left"]
sim_stress_xx_right_1to1dhs = np.load(folder + "AR1to1_doublets_half_stim/simulation_stress_xx_yy.npz")["stress_xx_right"]
sim_stress_yy_left_1to1dhs = np.load(folder + "AR1to1_doublets_half_stim/simulation_stress_xx_yy.npz")["stress_yy_left"]
sim_stress_yy_right_1to1dhs = np.load(folder + "AR1to1_doublets_half_stim/simulation_stress_xx_yy.npz")["stress_yy_right"]

sim_relstress_xx_left_1to1dhs = sim_stress_xx_left_1to1dhs / np.nanmean(sim_stress_xx_left_1to1dhs[0:20]) - 1
sim_relstress_xx_right_1to1dhs = sim_stress_xx_right_1to1dhs / np.nanmean(sim_stress_xx_right_1to1dhs[0:20]) - 1
sim_relstress_yy_left_1to1dhs = sim_stress_yy_left_1to1dhs / np.nanmean(sim_stress_yy_left_1to1dhs[0:20]) - 1
sim_relstress_yy_right_1to1dhs = sim_stress_yy_right_1to1dhs / np.nanmean(sim_stress_yy_right_1to1dhs[0:20]) - 1

figfolder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/_Figure3/"
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
# %% plot figure 3B, TFM differences

# prepare data first
Tx_1to1d_fs = AR1to1d_fullstim_long["TFM_data"]["Tx"]
Ty_1to1d_fs = AR1to1d_fullstim_long["TFM_data"]["Ty"]
Tx_1to1s_fs = AR1to1s_fullstim_long["TFM_data"]["Tx"]
Ty_1to1s_fs = AR1to1s_fullstim_long["TFM_data"]["Ty"]

Tx_1to1d_hs = AR1to1d_halfstim["TFM_data"]["Tx"]
Ty_1to1d_hs = AR1to1d_halfstim["TFM_data"]["Ty"]
Tx_1to1s_hs = AR1to1s_halfstim["TFM_data"]["Tx"]
Ty_1to1s_hs = AR1to1s_halfstim["TFM_data"]["Ty"]

# calculate amplitudes
T_1to1d_fs = np.sqrt(Tx_1to1d_fs ** 2 + Ty_1to1d_fs ** 2)
T_1to1s_fs = np.sqrt(Tx_1to1s_fs ** 2 + Ty_1to1s_fs ** 2)
T_1to1d_hs = np.sqrt(Tx_1to1d_hs ** 2 + Ty_1to1d_hs ** 2)
T_1to1s_hs = np.sqrt(Tx_1to1s_hs ** 2 + Ty_1to1s_hs ** 2)

# calculate difference between after and before photoactivation
Tx_1to1d_fs_diff = np.nanmean(Tx_1to1d_fs[:, :, 33, :] - Tx_1to1d_fs[:, :, 20, :], axis=2)
Ty_1to1d_fs_diff = np.nanmean(Ty_1to1d_fs[:, :, 33, :] - Ty_1to1d_fs[:, :, 20, :], axis=2)
T_1to1d_fs_diff = np.nanmean(T_1to1d_fs[:, :, 33, :] - T_1to1d_fs[:, :, 20, :], axis=2)

Tx_1to1s_fs_diff = np.nanmean(Tx_1to1s_fs[:, :, 33, :] - Tx_1to1s_fs[:, :, 20, :], axis=2)
Ty_1to1s_fs_diff = np.nanmean(Ty_1to1s_fs[:, :, 33, :] - Ty_1to1s_fs[:, :, 20, :], axis=2)
T_1to1s_fs_diff = np.nanmean(T_1to1s_fs[:, :, 33, :] - T_1to1s_fs[:, :, 20, :], axis=2)

Tx_1to1d_hs_diff = np.nanmean(Tx_1to1d_hs[:, :, 33, :] - Tx_1to1d_hs[:, :, 20, :], axis=2)
Ty_1to1d_hs_diff = np.nanmean(Ty_1to1d_hs[:, :, 33, :] - Ty_1to1d_hs[:, :, 20, :], axis=2)
T_1to1d_hs_diff = np.nanmean(T_1to1d_hs[:, :, 33, :] - T_1to1d_hs[:, :, 20, :], axis=2)

Tx_1to1s_hs_diff = np.nanmean(Tx_1to1s_hs[:, :, 33, :] - Tx_1to1s_hs[:, :, 20, :], axis=2)
Ty_1to1s_hs_diff = np.nanmean(Ty_1to1s_hs[:, :, 33, :] - Ty_1to1s_hs[:, :, 20, :], axis=2)
T_1to1s_hs_diff = np.nanmean(T_1to1s_hs[:, :, 33, :] - T_1to1s_hs[:, :, 20, :], axis=2)

# crop maps 
crop_start = 8
crop_end = 84

Tx_1to1d_fs_diff_crop = Tx_1to1d_fs_diff[crop_start:crop_end, crop_start:crop_end] * 1e-3  # convert to kPa
Ty_1to1d_fs_diff_crop = Ty_1to1d_fs_diff[crop_start:crop_end, crop_start:crop_end] * 1e-3
T_1to1d_fs_diff_crop = T_1to1d_fs_diff[crop_start:crop_end, crop_start:crop_end] * 1e-3

Tx_1to1s_fs_diff_crop = Tx_1to1s_fs_diff[crop_start:crop_end, crop_start:crop_end] * 1e-3
Ty_1to1s_fs_diff_crop = Ty_1to1s_fs_diff[crop_start:crop_end, crop_start:crop_end] * 1e-3
T_1to1s_fs_diff_crop = T_1to1s_fs_diff[crop_start:crop_end, crop_start:crop_end] * 1e-3

Tx_1to1d_hs_diff_crop = Tx_1to1d_hs_diff[crop_start:crop_end, crop_start:crop_end] * 1e-3
Ty_1to1d_hs_diff_crop = Ty_1to1d_hs_diff[crop_start:crop_end, crop_start:crop_end] * 1e-3
T_1to1d_hs_diff_crop = T_1to1d_hs_diff[crop_start:crop_end, crop_start:crop_end] * 1e-3

Tx_1to1s_hs_diff_crop = Tx_1to1s_hs_diff[crop_start:crop_end, crop_start:crop_end] * 1e-3
Ty_1to1s_hs_diff_crop = Ty_1to1s_hs_diff[crop_start:crop_end, crop_start:crop_end] * 1e-3
T_1to1s_hs_diff_crop = T_1to1s_hs_diff[crop_start:crop_end, crop_start:crop_end] * 1e-3

# set up plot parameters
# *****************************************************************************
n = 4  # every nth arrow will be plotted
pixelsize = 0.864  # in µm
pmax = 0.2  # kPa
pmin = -0.2

# create x- and y-axis for plotting maps
x_end = np.shape(T_1to1d_fs_diff_crop)[1]
y_end = np.shape(T_1to1d_fs_diff_crop)[0]
extent = [0, x_end * pixelsize, 0, y_end * pixelsize]

# create mesh for vectorplot    
xq, yq = np.meshgrid(np.linspace(0, extent[1], x_end), np.linspace(0, extent[3], y_end))

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(3.8, 3.3))

colormap = "seismic"

im = axes[0, 0].imshow(T_1to1d_fs_diff_crop, cmap=plt.get_cmap(colormap), interpolation="bilinear", extent=extent,
                       vmin=pmin, vmax=pmax, aspect='auto')
axes[0, 0].quiver(xq[::n, ::n], yq[::n, ::n], Tx_1to1d_fs_diff_crop[::n, ::n], Ty_1to1d_fs_diff_crop[::n, ::n],
                  angles='xy', scale=1, units='width', color="r")

axes[0, 1].imshow(T_1to1d_hs_diff_crop, cmap=plt.get_cmap(colormap), interpolation="bilinear", extent=extent,
                  vmin=pmin, vmax=pmax, aspect='auto')
axes[0, 1].quiver(xq[::n, ::n], yq[::n, ::n], Tx_1to1d_hs_diff_crop[::n, ::n], Ty_1to1d_hs_diff_crop[::n, ::n],
                  angles='xy', scale=1, units='width', color="r")

axes[1, 0].imshow(T_1to1s_fs_diff_crop, cmap=plt.get_cmap(colormap), interpolation="bilinear", extent=extent,
                  vmin=pmin, vmax=pmax, aspect='auto')
axes[1, 0].quiver(xq[::n, ::n], yq[::n, ::n], Tx_1to1s_fs_diff_crop[::n, ::n], Ty_1to1s_fs_diff_crop[::n, ::n],
                  angles='xy', scale=1, units='width', color="r")

axes[1, 1].imshow(T_1to1s_hs_diff_crop, cmap=plt.get_cmap(colormap), interpolation="bilinear", extent=extent,
                  vmin=pmin, vmax=pmax, aspect='auto')
axes[1, 1].quiver(xq[::n, ::n], yq[::n, ::n], Tx_1to1s_hs_diff_crop[::n, ::n], Ty_1to1s_hs_diff_crop[::n, ::n],
                  angles='xy', scale=1, units='width', color="r")

# adjust space in between plots
plt.subplots_adjust(wspace=0.02, hspace=-0.02)

# remove axes
for ax in axes.flat:
    ax.axis('off')
    aspectratio = 1.0
    ratio_default = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.set_aspect(ratio_default * aspectratio)

# add colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist())
cbar.ax.set_title('kPa')

# add annotations
plt.suptitle('$\mathrm{\Delta}$ Traction forces', y=0.98, x=0.44)

plt.figure(1).text(0.17, 0.89, "global activation")
plt.figure(1).text(0.49, 0.89, "local activation")

plt.figure(1).text(0.24, 0.84, "n=" + str(n_d_fullstim))
plt.figure(1).text(0.24, 0.46, "n=" + str(n_s_fullstim))
plt.figure(1).text(0.55, 0.84, "n=" + str(n_d_halfstim))
plt.figure(1).text(0.55, 0.46, "n=" + str(n_s_halfstim))

# save figure
fig.savefig(figfolder + 'B.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'B.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 3C, Relative strain energy over time

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.arange(60)
x = x[::2]  # downsample data for nicer plotting
ymin = -0.1
ymax = 0.3
xticks = np.arange(0, 61, 20)  # define where the major ticks are gonna be
yticks = np.arange(ymin, ymax + 0.01, 0.1)
xlabel = 'time [min]'
xticklabels = ['global \n act.', 'local \n act.']  # which labels to put on x-axis
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(5, 3))  # create figure and axes
plt.subplots_adjust(wspace=0.35, hspace=0.35)  # adjust space in between plots
# ******************************************************************************************************************************************



# Set up plot parameters for first panel
#######################################################################################################
ax = axes[0, 0]
color = colors_parent[1]
ylabel = 'doublet'
title = 'global activation'
y = AR1to1d_fullstim_long["TFM_data"]["relEs"]
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color)

ax.plot(sim_relEs_1to1dfs, color=color)

# Set up plot parameters for second panel
#######################################################################################################
ax = axes[0, 1]
color = colors_parent[1]
ylabel = None
title = 'local activation'
y = AR1to1d_halfstim["TFM_data"]["relEs"]
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color)

# Set up plot parameters for third panel
#######################################################################################################
ax = axes[1, 0]
color = colors_parent[2]
ylabel = 'singlet'
title = None
y = AR1to1s_fullstim_long["TFM_data"]["relEs"]
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color)

# Set up plot parameters for fourth panel
#######################################################################################################
ax = axes[1, 1]
color = colors_parent[2]
ylabel = None
title = None
y = AR1to1s_halfstim["TFM_data"]["relEs"]
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color)

# Set up plot parameters for fifth panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'REI'  # variable that goes on the y-axis
ax = axes[0, 2]  # define on which axis the plot goes
colors = [colors_parent[1], colors_parent[1]]  # defines colors
ymin = -0.2  # minimum value on y-axis
ymax = 0.8  # maximum value on y-axis
yticks = np.arange(-0.2, 0.81, 0.2)  # define where to put major ticks on y-axis
stat_annotation_offset = -0.15  # vertical offset of statistical annotation
ylabel = None  # which label to put on y-axis
title = None  # title of plot
box_pairs = [('AR1to1d_fs', 'AR1to1d_hs')]  # which groups to perform statistical test on

# make plots
make_two_box_and_swarmplots(x, y, df_doublet, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title, colors)

# Set up plot parameters for sixth panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'REI'  # variable that goes on the y-axis
ax = axes[1, 2]  # define on which axis the plot goes
colors = [colors_parent[2], colors_parent[2]]  # defines colors
ymin = -0.2  # minimum value on y-axis
ymax = 0.8  # maximum value on y-axis
yticks = np.arange(-0.2, 0.81, 0.2)  # define where to put major ticks on y-axis
stat_annotation_offset = -0.15  # vertical offset of statistical annotation
ylabel = None  # which label to put on y-axis
title = None  # title of plot
box_pairs = [('AR1to1s_fs', 'AR1to1s_hs')]  # which groups to perform statistical test on

# make plots
make_two_box_and_swarmplots(x, y, df_singlet, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title, colors)

# write title for panels 1 to 4
plt.text(-5.1, 2.4, 'Relative strain energy', fontsize=10)
# write title for panels 5 to 6
plt.text(-0.61, 2.25, 'Relative energy \n     increase', fontsize=10)
# save plot to file
plt.savefig(figfolder + 'C.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'C.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 3C detrend, Relative detrend strain energy over time

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.arange(60)
x = x[::2]  # downsample data for nicer plotting
ymin = -0.1
ymax = 0.3
xticks = np.arange(0, 61, 20)  # define where the major ticks are gonna be
yticks = np.arange(ymin, ymax + 0.01, 0.1)
xlabel = 'time [min]'
xticklabels = ['global \n act.', 'local \n act.']  # which labels to put on x-axis
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(5, 3))  # create figure and axes
plt.subplots_adjust(wspace=0.35, hspace=0.35)  # adjust space in between plots
# ******************************************************************************************************************************************


# Set up plot parameters for first panel
#######################################################################################################
ax = axes[0, 0]
color = colors_parent[1]
ylabel = 'doublet'
title = 'global activation'
y = AR1to1d_fullstim_long["TFM_data"]["relEs_detrend"]
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color)

ax.plot(sim_relEs_1to1dfs, color=color)

# Set up plot parameters for second panel
#######################################################################################################
ax = axes[0, 1]
color = colors_parent[1]
ylabel = None
title = 'local activation'
y = AR1to1d_halfstim["TFM_data"]["relEs_detrend"]
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color)

# Set up plot parameters for third panel
#######################################################################################################
ax = axes[1, 0]
color = colors_parent[2]
ylabel = 'singlet'
title = None
y = AR1to1s_fullstim_long["TFM_data"]["relEs_detrend"]
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color)

# Set up plot parameters for fourth panel
#######################################################################################################
ax = axes[1, 1]
color = colors_parent[2]
ylabel = None
title = None
y = AR1to1s_halfstim["TFM_data"]["relEs_detrend"]
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color)

# Set up plot parameters for fifth panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'REI_detrend'  # variable that goes on the y-axis
ax = axes[0, 2]  # define on which axis the plot goes
colors = [colors_parent[1], colors_parent[1]]  # defines colors
ymin = -0.2  # minimum value on y-axis
ymax = 0.8  # maximum value on y-axis
yticks = np.arange(-0.2, 0.81, 0.2)  # define where to put major ticks on y-axis
stat_annotation_offset = -0.15  # vertical offset of statistical annotation
ylabel = None  # which label to put on y-axis
title = None  # title of plot
box_pairs = [('AR1to1d_fs', 'AR1to1d_hs')]  # which groups to perform statistical test on

# make plots
make_two_box_and_swarmplots(x, y, df_doublet, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title, colors)

# Set up plot parameters for sixth panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'REI_detrend'  # variable that goes on the y-axis
ax = axes[1, 2]  # define on which axis the plot goes
colors = [colors_parent[2], colors_parent[2]]  # defines colors
ymin = -0.2  # minimum value on y-axis
ymax = 0.8  # maximum value on y-axis
yticks = np.arange(-0.2, 0.81, 0.2)  # define where to put major ticks on y-axis
stat_annotation_offset = -0.22  # vertical offset of statistical annotation
ylabel = None  # which label to put on y-axis
title = None  # title of plot
box_pairs = [('AR1to1s_fs', 'AR1to1s_hs')]  # which groups to perform statistical test on

# make plots
make_two_box_and_swarmplots(x, y, df_singlet, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title, colors)

# write title for panels 1 to 4
plt.text(-5.1, 2.4, 'Relative strain energy', fontsize=10)
# write title for panels 5 to 6
plt.text(-0.61, 2.25, 'Relative energy \n     increase', fontsize=10)
# save plot to file
plt.savefig(figfolder + 'C_detrend.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'C_detrend.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 3D, stress map difference halfstim

# prepare data first

# concatenate TFM maps from different experiments and calculate average maps over first 20 frames and all cells to get average maps
sigmaxx_1to1d_diff = np.nanmean(
    AR1to1d_halfstim["MSM_data"]["sigma_xx"][:, :, 33, :] - AR1to1d_halfstim["MSM_data"]["sigma_xx"][:, :, 20, :],
    axis=2)
sigmayy_1to1d_diff = np.nanmean(
    AR1to1d_halfstim["MSM_data"]["sigma_yy"][:, :, 33, :] - AR1to1d_halfstim["MSM_data"]["sigma_yy"][:, :, 20, :],
    axis=2)
sigmaxx_1to1s_diff = np.nanmean(
    AR1to1s_halfstim["MSM_data"]["sigma_xx"][:, :, 33, :] - AR1to1s_halfstim["MSM_data"]["sigma_xx"][:, :, 20, :],
    axis=2)
sigmayy_1to1s_diff = np.nanmean(
    AR1to1s_halfstim["MSM_data"]["sigma_yy"][:, :, 33, :] - AR1to1s_halfstim["MSM_data"]["sigma_yy"][:, :, 20, :],
    axis=2)

# crop maps
crop_start = 8
crop_end = 84

sigmaxx_1to1d_diff_crop = sigmaxx_1to1d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3  # convert to mN/m
sigmayy_1to1d_diff_crop = sigmayy_1to1d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3
sigmaxx_1to1s_diff_crop = sigmaxx_1to1s_diff[crop_start:crop_end, crop_start:crop_end] * 1e3
sigmayy_1to1s_diff_crop = sigmayy_1to1s_diff[crop_start:crop_end, crop_start:crop_end] * 1e3

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

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(3.8, 3.3))

colormap = "seismic"

im = axes[0, 0].imshow(sigmaxx_1to1d_diff_crop, cmap=plt.get_cmap(colormap), interpolation="bilinear", extent=extent,
                       vmin=sigma_min, vmax=sigma_max, aspect='auto')

axes[0, 1].imshow(sigmayy_1to1d_diff_crop, cmap=plt.get_cmap(colormap), interpolation="bilinear", extent=extent,
                  vmin=sigma_min, vmax=sigma_max, aspect='auto')

axes[1, 0].imshow(sigmaxx_1to1s_diff_crop, cmap=plt.get_cmap(colormap), interpolation="bilinear", extent=extent,
                  vmin=sigma_min, vmax=sigma_max, aspect='auto')

axes[1, 1].imshow(sigmayy_1to1s_diff_crop, cmap=plt.get_cmap(colormap), interpolation="bilinear", extent=extent,
                  vmin=sigma_min, vmax=sigma_max, aspect='auto')

# adjust space in between plots
plt.subplots_adjust(wspace=0.02, hspace=-0.02)

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
plt.suptitle('$\mathrm{\Delta}$ Stresses, local activation', y=0.97, x=0.44)

# plt.suptitle('$\mathrm{\Delta}$ Traction forces', y=0.98, x=0.44)

plt.figure(1).text(0.21, 0.88, "xx-Stress")
plt.figure(1).text(0.52, 0.88, "yy-Stress")

plt.figure(1).text(0.24, 0.84, "n=" + str(n_d_halfstim))
plt.figure(1).text(0.24, 0.46, "n=" + str(n_s_halfstim))
plt.figure(1).text(0.55, 0.84, "n=" + str(n_d_halfstim))
plt.figure(1).text(0.55, 0.46, "n=" + str(n_s_halfstim))

# save figure
fig.savefig(figfolder + 'D.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'D.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 3E, Relative stress over time halfstim

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.arange(60)
x = x[::2]  # downsample data for nicer plotting
ymin = -0.1
ymax = 0.2
xticks = np.arange(0, 61, 20)  # define where the major ticks are gonna be
yticks = np.arange(ymin, ymax + 0.01, 0.1)
xlabel = 'time [min]'
xticklabels = ['left \n         $\mathrm{\Delta \sigma _ {xx}}$', 'right', 'left \n         $\mathrm{\Delta \sigma _ {yy}}$', 'right']  # which labels to put on x-axis
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(5, 3))  # create figure and axes
plt.subplots_adjust(wspace=0.35, hspace=0.35)  # adjust space in between plots
# ******************************************************************************************************************************************


# Set up plot parameters for first panel
#######################################################################################################
ax = axes[0, 0]
colors = [colors_parent[1], colors_parent_dark[1]]
ylabel = 'doublet'
title = 'xx-Stress'
y1 = AR1to1d_halfstim["MSM_data"]["relsigma_xx_left"]
y2 = AR1to1d_halfstim["MSM_data"]["relsigma_xx_right"]
y1 = y1[::2, :]
y2 = y2[::2, :]

# make plots
plot_two_values_over_time(x, y1, y2, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors)

ax.plot(sim_relstress_xx_left_1to1dhs, color=colors[0])
ax.plot(sim_relstress_xx_right_1to1dhs, color=colors[0])

# Set up plot parameters for second panel
#######################################################################################################
ax = axes[0, 1]
colors = [colors_parent[1], colors_parent_dark[1]]
ylabel = None
title = 'yy-Stress'
y1 = AR1to1d_halfstim["MSM_data"]["relsigma_yy_left"]
y2 = AR1to1d_halfstim["MSM_data"]["relsigma_yy_right"]
y1 = y1[::2, :]
y2 = y2[::2, :]

# make plots
plot_two_values_over_time(x, y1, y2, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors)

ax.plot(sim_relstress_yy_left_1to1dhs, color=colors[0])
ax.plot(sim_relstress_yy_right_1to1dhs, color=colors[0])

# Set up plot parameters for third panel
#######################################################################################################
ax = axes[1, 0]
colors = [colors_parent[2], colors_parent_dark[2]]
ylabel = 'singlet'
title = None
y1 = AR1to1s_halfstim["MSM_data"]["relsigma_xx_left"]
y2 = AR1to1s_halfstim["MSM_data"]["relsigma_xx_right"]
y1 = y1[::2, :]
y2 = y2[::2, :]

# make plots
plot_two_values_over_time(x, y1, y2, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors)

# Set up plot parameters for fourth panel
#######################################################################################################
ax = axes[1, 1]
colors = [colors_parent[2], colors_parent_dark[2]]
ylabel = None
title = None
y1 = AR1to1s_halfstim["MSM_data"]["relsigma_yy_left"]
y2 = AR1to1s_halfstim["MSM_data"]["relsigma_yy_right"]
y1 = y1[::2, :]
y2 = y2[::2, :]

# make plots
plot_two_values_over_time(x, y1, y2, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors)

# Set up plot parameters for fifth panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'sigma'  # variable that goes on the y-axis
ax = axes[0, 2]  # define on which axis the plot goes
colors = [colors_parent[1], colors_parent_dark[1], colors_parent[1], colors_parent_dark[1]]  # defines colors
ymin = -0.2  # minimum value on y-axis
ymax = 0.4  # maximum value on y-axis
yticks = np.arange(-0.2, 0.41, 0.2)  # define where to put major ticks on y-axis
ylabel = None  # which label to put on y-axis
title = None  # title of plot

# a little weird way to get the dataframe format I need for seaborn boxplots...
df1 = pd.DataFrame(df_d_hs['RSI_xx_left'])
df2 = pd.DataFrame(df_d_hs['RSI_xx_right'])
df3 = pd.DataFrame(df_d_hs['RSI_yy_left'])
df4 = pd.DataFrame(df_d_hs['RSI_yy_right'])

df1 = df1.transpose().reset_index(drop=True).transpose()
df2 = df2.transpose().reset_index(drop=True).transpose()
df3 = df3.transpose().reset_index(drop=True).transpose()
df4 = df4.transpose().reset_index(drop=True).transpose()

df_plot = pd.concat([df1, df2, df3, df4], axis=0)

df_plot.rename(columns={0: 'sigma'}, inplace=True)

keys_sx_l = ['sigma_xx_left' for i in range(n_d_halfstim)]
keys_sx_r = ['sigma_xx_right' for i in range(n_d_halfstim)]
keys_sy_l = ['sigma_yy_left' for i in range(n_d_halfstim)]
keys_sy_r = ['sigma_yy_right' for i in range(n_d_halfstim)]
keys = np.concatenate((keys_sx_l, keys_sx_r, keys_sy_l, keys_sy_r))
df_plot['keys'] = keys

# make plots
make_four_box_and_swarmplots(x, y, df_plot, ax, ymin, ymax, yticks, xticklabels, ylabel, title, colors)

ax.plot(0, sim_relstress_xx_left_1to1dhs[30], zorder=3, marker='^', markersize=3, markerfacecolor="white", markeredgecolor="black",
        markeredgewidth=0.3, alpha=1)
ax.plot(1, sim_relstress_xx_right_1to1dhs[30], zorder=3, marker='^', markersize=3, markerfacecolor="white", markeredgecolor="black",
        markeredgewidth=0.3, alpha=1)
ax.plot(2, sim_relstress_yy_left_1to1dhs[30], zorder=3, marker='^', markersize=3, markerfacecolor="white", markeredgecolor="black",
        markeredgewidth=0.3, alpha=1)
ax.plot(3, sim_relstress_yy_right_1to1dhs[30], zorder=3, marker='^', markersize=3, markerfacecolor="white", markeredgecolor="black",
        markeredgewidth=0.3, alpha=1)

# Set up plot parameters for sixth panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'sigma'  # variable that goes on the y-axis
ax = axes[1, 2]  # define on which axis the plot goes
colors = [colors_parent[2], colors_parent_dark[2], colors_parent[2], colors_parent_dark[2]]  # defines colors
ymin = -0.2  # minimum value on y-axis
ymax = 0.4  # maximum value on y-axis
yticks = np.arange(-0.2, 0.41, 0.2)  # define where to put major ticks on y-axis
ylabel = None  # which label to put on y-axis
title = None  # title of plot

# a little weird way to get the dataframe format I need for seaborn boxplots...
df1 = pd.DataFrame(df_s_hs['RSI_xx_left'])
df2 = pd.DataFrame(df_s_hs['RSI_xx_right'])
df3 = pd.DataFrame(df_s_hs['RSI_yy_left'])
df4 = pd.DataFrame(df_s_hs['RSI_yy_right'])

df1 = df1.transpose().reset_index(drop=True).transpose()
df2 = df2.transpose().reset_index(drop=True).transpose()
df3 = df3.transpose().reset_index(drop=True).transpose()
df4 = df4.transpose().reset_index(drop=True).transpose()

df_plot = pd.concat([df1, df2, df3, df4], axis=0)

df_plot.rename(columns={0: 'sigma'}, inplace=True)

keys_sx_l = ['sigma_xx_left' for i in range(n_s_halfstim)]
keys_sx_r = ['sigma_xx_right' for i in range(n_s_halfstim)]
keys_sy_l = ['sigma_yy_left' for i in range(n_s_halfstim)]
keys_sy_r = ['sigma_yy_right' for i in range(n_s_halfstim)]
keys = np.concatenate((keys_sx_l, keys_sx_r, keys_sy_l, keys_sy_r))
df_plot['keys'] = keys

# make plots
make_four_box_and_swarmplots(x, y, df_plot, ax, ymin, ymax, yticks, xticklabels, ylabel, title, colors)



# write title for panels 1 to 4
plt.text(-11, 1.35, 'Relative stresses, local activation', fontsize=10)
# write title for panels 5 to 6
plt.text(-0.65, 1.25, 'Relative stress \n     increase', fontsize=10)
#
# # save plot to file
plt.savefig(figfolder + 'E.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'E.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 3E detrend, Relative detrend stress over time halfstim

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.arange(60)
x = x[::2]  # downsample data for nicer plotting
ymin = -0.1
ymax = 0.2
xticks = np.arange(0, 61, 20)  # define where the major ticks are gonna be
yticks = np.arange(ymin, ymax + 0.01, 0.1)
xlabel = 'time [min]'
xticklabels = ['left \n         $\mathrm{\Delta \sigma _ {xx}}$', 'right', 'left \n         $\mathrm{\Delta \sigma _ {yy}}$', 'right']  # which labels to put on x-axis
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(5, 3))  # create figure and axes
plt.subplots_adjust(wspace=0.35, hspace=0.35)  # adjust space in between plots
# ******************************************************************************************************************************************


# Set up plot parameters for first panel
#######################################################################################################
ax = axes[0, 0]
colors = [colors_parent[1], colors_parent_dark[1]]
ylabel = 'doublet'
title = 'xx-Stress'
y1 = AR1to1d_halfstim["MSM_data"]["relsigma_xx_left_detrend"]
y2 = AR1to1d_halfstim["MSM_data"]["relsigma_xx_right_detrend"]
y1 = y1[::2, :]
y2 = y2[::2, :]

# make plots
plot_two_values_over_time(x, y1, y2, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors)

ax.plot(sim_relstress_xx_left_1to1dhs, color=colors[0])
ax.plot(sim_relstress_xx_right_1to1dhs, color=colors[0])

# Set up plot parameters for second panel
#######################################################################################################
ax = axes[0, 1]
colors = [colors_parent[1], colors_parent_dark[1]]
ylabel = None
title = 'yy-Stress'
y1 = AR1to1d_halfstim["MSM_data"]["relsigma_yy_left_detrend"]
y2 = AR1to1d_halfstim["MSM_data"]["relsigma_yy_right_detrend"]
y1 = y1[::2, :]
y2 = y2[::2, :]

# make plots
plot_two_values_over_time(x, y1, y2, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors)

ax.plot(sim_relstress_yy_left_1to1dhs, color=colors[0])
ax.plot(sim_relstress_yy_right_1to1dhs, color=colors[0])

# Set up plot parameters for third panel
#######################################################################################################
ax = axes[1, 0]
colors = [colors_parent[2], colors_parent_dark[2]]
ylabel = 'singlet'
title = None
y1 = AR1to1s_halfstim["MSM_data"]["relsigma_xx_left_detrend"]
y2 = AR1to1s_halfstim["MSM_data"]["relsigma_xx_right_detrend"]
y1 = y1[::2, :]
y2 = y2[::2, :]

# make plots
plot_two_values_over_time(x, y1, y2, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors)

# Set up plot parameters for fourth panel
#######################################################################################################
ax = axes[1, 1]
colors = [colors_parent[2], colors_parent_dark[2]]
ylabel = None
title = None
y1 = AR1to1s_halfstim["MSM_data"]["relsigma_yy_left_detrend"]
y2 = AR1to1s_halfstim["MSM_data"]["relsigma_yy_right_detrend"]
y1 = y1[::2, :]
y2 = y2[::2, :]

# make plots
plot_two_values_over_time(x, y1, y2, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors)

# Set up plot parameters for fifth panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'sigma'  # variable that goes on the y-axis
ax = axes[0, 2]  # define on which axis the plot goes
colors = [colors_parent[1], colors_parent_dark[1], colors_parent[1], colors_parent_dark[1]]  # defines colors
ymin = -0.2  # minimum value on y-axis
ymax = 0.4  # maximum value on y-axis
yticks = np.arange(-0.2, 0.41, 0.2)  # define where to put major ticks on y-axis
ylabel = None  # which label to put on y-axis
title = None  # title of plot

# a little weird way to get the dataframe format I need for seaborn boxplots...
df1 = pd.DataFrame(df_d_hs['RSI_xx_left_detrend'])
df2 = pd.DataFrame(df_d_hs['RSI_xx_right_detrend'])
df3 = pd.DataFrame(df_d_hs['RSI_yy_left_detrend'])
df4 = pd.DataFrame(df_d_hs['RSI_yy_right_detrend'])

df1 = df1.transpose().reset_index(drop=True).transpose()
df2 = df2.transpose().reset_index(drop=True).transpose()
df3 = df3.transpose().reset_index(drop=True).transpose()
df4 = df4.transpose().reset_index(drop=True).transpose()

df_plot = pd.concat([df1, df2, df3, df4], axis=0)

df_plot.rename(columns={0: 'sigma'}, inplace=True)

keys_sx_l = ['sigma_xx_left' for i in range(n_d_halfstim)]
keys_sx_r = ['sigma_xx_right' for i in range(n_d_halfstim)]
keys_sy_l = ['sigma_yy_left' for i in range(n_d_halfstim)]
keys_sy_r = ['sigma_yy_right' for i in range(n_d_halfstim)]
keys = np.concatenate((keys_sx_l, keys_sx_r, keys_sy_l, keys_sy_r))
df_plot['keys'] = keys

# make plots
make_four_box_and_swarmplots(x, y, df_plot, ax, ymin, ymax, yticks, xticklabels, ylabel, title, colors)

ax.plot(0, sim_relstress_xx_left_1to1dhs[30], zorder=3, marker='^', markersize=3, markerfacecolor="white", markeredgecolor="black",
        markeredgewidth=0.3, alpha=1)
ax.plot(1, sim_relstress_xx_right_1to1dhs[30], zorder=3, marker='^', markersize=3, markerfacecolor="white", markeredgecolor="black",
        markeredgewidth=0.3, alpha=1)
ax.plot(2, sim_relstress_yy_left_1to1dhs[30], zorder=3, marker='^', markersize=3, markerfacecolor="white", markeredgecolor="black",
        markeredgewidth=0.3, alpha=1)
ax.plot(3, sim_relstress_yy_right_1to1dhs[30], zorder=3, marker='^', markersize=3, markerfacecolor="white", markeredgecolor="black",
        markeredgewidth=0.3, alpha=1)

# Set up plot parameters for sixth panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'sigma'  # variable that goes on the y-axis
ax = axes[1, 2]  # define on which axis the plot goes
colors = [colors_parent[2], colors_parent_dark[2], colors_parent[2], colors_parent_dark[2]]  # defines colors
ymin = -0.2  # minimum value on y-axis
ymax = 0.4  # maximum value on y-axis
yticks = np.arange(-0.2, 0.41, 0.2)  # define where to put major ticks on y-axis
ylabel = None  # which label to put on y-axis
title = None  # title of plot
ylabeloffset = -1

# a little weird way to get the dataframe format I need for seaborn boxplots...
df1 = pd.DataFrame(df_s_hs['RSI_xx_left_detrend'])
df2 = pd.DataFrame(df_s_hs['RSI_xx_right_detrend'])
df3 = pd.DataFrame(df_s_hs['RSI_yy_left_detrend'])
df4 = pd.DataFrame(df_s_hs['RSI_yy_right_detrend'])

df1 = df1.transpose().reset_index(drop=True).transpose()
df2 = df2.transpose().reset_index(drop=True).transpose()
df3 = df3.transpose().reset_index(drop=True).transpose()
df4 = df4.transpose().reset_index(drop=True).transpose()

df_plot = pd.concat([df1, df2, df3, df4], axis=0)

df_plot.rename(columns={0: 'sigma'}, inplace=True)

keys_sx_l = ['sigma_xx_left' for i in range(n_s_halfstim)]
keys_sx_r = ['sigma_xx_right' for i in range(n_s_halfstim)]
keys_sy_l = ['sigma_yy_left' for i in range(n_s_halfstim)]
keys_sy_r = ['sigma_yy_right' for i in range(n_s_halfstim)]
keys = np.concatenate((keys_sx_l, keys_sx_r, keys_sy_l, keys_sy_r))
df_plot['keys'] = keys

# make plots
make_four_box_and_swarmplots(x, y, df_plot, ax, ymin, ymax, yticks, xticklabels, ylabel, title, colors)


# write title for panels 1 to 4
plt.text(-11, 1.35, 'Relative stresses, local activation', fontsize=10)
# write title for panels 5 to 6
plt.text(-0.65, 1.25, 'Relative stress \n     increase', fontsize=10)
#
# # save plot to file
plt.savefig(figfolder + 'E_detrend.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'E_detrend.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure S3A, actin intensity over time

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.arange(60)
x = x[::2]  # downsample data for nicer plotting
ymin = 0.95
ymax = 1.05
xticks = np.arange(0, 61, 20)
yticks = np.arange(0.95, 1.051, 0.05)
xlabel = 'time [min]'
xticklabels = ['left', 'right']  # which labels to put on x-axis
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(3, 3))  # create figure and axes
plt.subplots_adjust(wspace=0.5, hspace=0.5)  # adjust space in between plots
# ******************************************************************************************************************************************

# Set up plot parameters for first panel
#######################################################################################################
ax = axes[0, 0]
colors = [colors_parent[1], colors_parent_dark[1]]
ylabel = 'doublet'
title = 'Relative actin intensity'
y1 = AR1to1d_halfstim["shape_data"]["relactin_intensity_left"]
y2 = AR1to1d_halfstim["shape_data"]["relactin_intensity_right"]
y1 = y1[::2, :]
y2 = y2[::2, :]

# make plots
plot_two_values_over_time(x, y1, y2, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors, titleoffset=5)

# Set up plot parameters for second panel
#######################################################################################################
ax = axes[1, 0]
colors = [colors_parent[2], colors_parent_dark[2]]
ylabel = 'singlet'
title = None
y1 = AR1to1s_halfstim["shape_data"]["relactin_intensity_left"]
y2 = AR1to1s_halfstim["shape_data"]["relactin_intensity_right"]
y1 = y1[::2, :]
y2 = y2[::2, :]

# make plots
plot_two_values_over_time(x, y1, y2, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors)

# Set up plot parameters for third panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'RAI'  # variable that goes on the y-axis
ax = axes[0, 1]  # define on which axis the plot goes
colors = [colors_parent[1], colors_parent_dark[1], colors_parent[1], colors_parent_dark[1]]     # defines colors
ymin = -0.2  # minimum value on y-axis
ymax = 0.2  # maximum value on y-axis
yticks = np.arange(-0.2, 0.21, 0.1)  # define where to put major ticks on y-axis
stat_annotation_offset = 0.5  # vertical offset of statistical annotation
ylabel = None  # which label to put on y-axis
title = None  # title of plot
box_pairs = [('RAI_left', 'RAI_right')]  # which groups to perform statistical test on

# a little weird way to get the dataframe format I need for seaborn boxplots...
df1 = pd.DataFrame(df_d_hs['RAI_left'])
df2 = pd.DataFrame(df_d_hs['RAI_right'])

df1 = df1.transpose().reset_index(drop=True).transpose()
df2 = df2.transpose().reset_index(drop=True).transpose()

df_plot = pd.concat([df1, df2], axis=0)

df_plot.rename(columns={0: 'RAI'}, inplace=True)

keys_l = ['RAI_left' for i in range(n_d_halfstim)]
keys_r = ['RAI_right' for i in range(n_d_halfstim)]

keys = np.concatenate((keys_l, keys_r))
df_plot['keys'] = keys

# make plots
make_two_box_and_swarmplots(x, y, df_plot, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title, colors)

# Set up plot parameters for fourth panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'RAI'  # variable that goes on the y-axis
ax = axes[1, 1]  # define on which axis the plot goes
colors = [colors_parent[2], colors_parent_dark[2], colors_parent[2], colors_parent_dark[2]]     # defines colors
ymin = -0.2  # minimum value on y-axis
ymax = 0.2  # maximum value on y-axis
yticks = np.arange(-0.2, 0.21, 0.1)  # define where to put major ticks on y-axis
stat_annotation_offset = -0.2  # vertical offset of statistical annotation
ylabel = None  # which label to put on y-axis
title = None  # title of plot
box_pairs = [('RAI_left', 'RAI_right')]  # which groups to perform statistical test on

# a little weird way to get the dataframe format I need for seaborn boxplots...
df1 = pd.DataFrame(df_s_hs['RAI_left'])
df2 = pd.DataFrame(df_s_hs['RAI_right'])

df1 = df1.transpose().reset_index(drop=True).transpose()
df2 = df2.transpose().reset_index(drop=True).transpose()

df_plot = pd.concat([df1, df2], axis=0)

df_plot.rename(columns={0: 'RAI'}, inplace=True)

keys_l = ['RAI_left' for i in range(n_s_halfstim)]
keys_r = ['RAI_right' for i in range(n_s_halfstim)]

keys = np.concatenate((keys_l, keys_r))
df_plot['keys'] = keys

# make plots
make_two_box_and_swarmplots(x, y, df_plot, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title, colors)



# # write title for panels 1 to 4
# # plt.text(-5.7, 2.575947, 'Relative strain energy', fontsize=10)
# write title for panels 3 and 4
plt.text(-0.6, 0.82, 'Relative actin \n     increase', fontsize=10)
# # save plot to file
plt.savefig(figfolder + 'SA.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'SA.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure S3B, stress map difference fullstim

# prepare data first

# concatenate TFM maps from different experiments and calculate average maps over first 20 frames and all cells to get average maps
sigmaxx_1to1d_diff = np.nanmean(
    AR1to1d_fullstim_long["MSM_data"]["sigma_xx"][:, :, 33, :] - AR1to1d_fullstim_long["MSM_data"]["sigma_xx"][:, :, 20,
                                                                 :], axis=2)
sigmayy_1to1d_diff = np.nanmean(
    AR1to1d_fullstim_long["MSM_data"]["sigma_yy"][:, :, 33, :] - AR1to1d_fullstim_long["MSM_data"]["sigma_yy"][:, :, 20,
                                                                 :], axis=2)
sigmaxx_1to1s_diff = np.nanmean(
    AR1to1s_fullstim_long["MSM_data"]["sigma_xx"][:, :, 33, :] - AR1to1s_fullstim_long["MSM_data"]["sigma_xx"][:, :, 20,
                                                                 :], axis=2)
sigmayy_1to1s_diff = np.nanmean(
    AR1to1s_fullstim_long["MSM_data"]["sigma_yy"][:, :, 33, :] - AR1to1s_fullstim_long["MSM_data"]["sigma_yy"][:, :, 20,
                                                                 :], axis=2)

# crop maps
crop_start = 8
crop_end = 84

sigmaxx_1to1d_diff_crop = sigmaxx_1to1d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3  # convert to mN/m
sigmayy_1to1d_diff_crop = sigmayy_1to1d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3
sigmaxx_1to1s_diff_crop = sigmaxx_1to1s_diff[crop_start:crop_end, crop_start:crop_end] * 1e3
sigmayy_1to1s_diff_crop = sigmayy_1to1s_diff[crop_start:crop_end, crop_start:crop_end] * 1e3

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

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(3.8, 3.3))

colormap = "seismic"

im = axes[0, 0].imshow(sigmaxx_1to1d_diff_crop, cmap=plt.get_cmap(colormap), interpolation="bilinear", extent=extent,
                       vmin=sigma_min, vmax=sigma_max, aspect='auto')

axes[0, 1].imshow(sigmayy_1to1d_diff_crop, cmap=plt.get_cmap(colormap), interpolation="bilinear", extent=extent,
                  vmin=sigma_min, vmax=sigma_max, aspect='auto')

axes[1, 0].imshow(sigmaxx_1to1s_diff_crop, cmap=plt.get_cmap(colormap), interpolation="bilinear", extent=extent,
                  vmin=sigma_min, vmax=sigma_max, aspect='auto')

axes[1, 1].imshow(sigmayy_1to1s_diff_crop, cmap=plt.get_cmap(colormap), interpolation="bilinear", extent=extent,
                  vmin=sigma_min, vmax=sigma_max, aspect='auto')

# adjust space in between plots
plt.subplots_adjust(wspace=0.02, hspace=-0.02)

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
plt.suptitle('$\mathrm{\Delta}$ Stresses, global activation', y=0.97, x=0.44)

plt.figure(1).text(0.21, 0.88, "xx-Stress")
plt.figure(1).text(0.52, 0.88, "yy-Stress")

plt.figure(1).text(0.24, 0.84, "n=" + str(n_d_fullstim))
plt.figure(1).text(0.24, 0.46, "n=" + str(n_s_fullstim))
plt.figure(1).text(0.55, 0.84, "n=" + str(n_d_fullstim))
plt.figure(1).text(0.55, 0.46, "n=" + str(n_s_fullstim))

# save figure
fig.savefig(figfolder + 'SB.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'SB.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure S3C, Relative stress over time fullstim

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.arange(60)
x = x[::2]  # downsample data for nicer plotting
ymin = -0.1
ymax = 0.2
xticks = np.arange(0, 61, 20)  # define where the major ticks are gonna be
yticks = np.arange(ymin, ymax + 0.01, 0.1)
xlabel = 'time [min]'
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(5, 3))  # create figure and axes
plt.subplots_adjust(wspace=0.35, hspace=0.35)  # adjust space in between plots
# ******************************************************************************************************************************************

# Set up plot parameters for first panel
#######################################################################################################
ax = axes[0, 0]
color = colors_parent[1]
ylabel = 'doublet'
title = 'xx-Stress'
y = AR1to1d_fullstim_long["MSM_data"]["relsigma_xx"]
y = y[::2]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color)

ax.plot(x, sim_relstress_xx_1to1dfs[::2], color=color)


# Set up plot parameters for second panel
#######################################################################################################
ax = axes[0, 1]
color = colors_parent[1]
ylabel = None
title = 'yy-Stress'
y = AR1to1d_fullstim_long["MSM_data"]["relsigma_yy"]
y = y[::2]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color)

ax.plot(x, sim_relstress_yy_1to1dfs[::2], color=color)

# Set up plot parameters for third panel
#######################################################################################################
ax = axes[1, 0]
color = colors_parent[2]
ylabel = 'singlet'
title = None
y = AR1to1s_fullstim_long["MSM_data"]["relsigma_xx"]
y = y[::2]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color)



# Set up plot parameters for fourth panel
#######################################################################################################
ax = axes[1, 1]
color = colors_parent[2]
ylabel = None
title = None
y = AR1to1s_fullstim_long["MSM_data"]["relsigma_yy"]
y = y[::2]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color)

# Set up plot parameters for fifth panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'sigma'  # variable that goes on the y-axis
ax = axes[0, 2]  # define on which axis the plot goes
colors = [colors_parent[1], colors_parent[1]]     # defines colors
ymin = -0.2  # minimum value on y-axis
ymax = 0.6  # maximum value on y-axis
yticks = np.arange(-0.2, 0.61, 0.2)  # define where to put major ticks on y-axis
stat_annotation_offset = 0  # vertical offset of statistical annotation
ylabel = None  # which label to put on y-axis
title = None  # title of plot
box_pairs = [('sigma_xx', 'sigma_yy')] # which groups to perform statistical test on

# a little weird way to get the dataframe format I need for seaborn boxplots...
df1 = pd.DataFrame(df_d_fs['RSI_xx'])
df2 = pd.DataFrame(df_d_fs['RSI_yy'])
df1 = df1.transpose().reset_index(drop=True).transpose()
df2 = df2.transpose().reset_index(drop=True).transpose()

df_plot = pd.concat([df1, df2], axis=0)

df_plot.rename(columns={0: 'sigma'}, inplace=True)

keys_sx = ['sigma_xx' for i in range(n_d_fullstim)]
keys_sy = ['sigma_yy' for i in range(n_d_fullstim)]
keys = np.concatenate((keys_sx, keys_sy))
df_plot['keys'] = keys

# make plots
make_two_box_and_swarmplots(x, y, df_plot, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title, colors)

# Set up plot parameters for sixth panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'sigma'  # variable that goes on the y-axis
ax = axes[1, 2]  # define on which axis the plot goes
colors = [colors_parent[2], colors_parent_dark[2], colors_parent[2], colors_parent_dark[2]]     # defines colors
stat_annotation_offset = 0  # vertical offset of statistical annotation
ylabel = None  # which label to put on y-axis
title = None  # title of plot
box_pairs = [('sigma_xx', 'sigma_yy')]  # which groups to perform statistical test on

# a little weird way to get the dataframe format I need for seaborn boxplots...
df1 = pd.DataFrame(df_s_fs['RSI_xx'])
df2 = pd.DataFrame(df_s_fs['RSI_yy'])
df1 = df1.transpose().reset_index(drop=True).transpose()
df2 = df2.transpose().reset_index(drop=True).transpose()

df_plot = pd.concat([df1, df2], axis=0)

df_plot.rename(columns={0: 'sigma'}, inplace=True)

keys_sx = ['sigma_xx' for i in range(n_s_fullstim)]
keys_sy = ['sigma_yy' for i in range(n_s_fullstim)]
keys = np.concatenate((keys_sx, keys_sy))
df_plot['keys'] = keys


# make plots
make_two_box_and_swarmplots(x, y, df_plot, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title, colors)

# write title for panels 1 to 4
plt.text(-5.85, 1.82, 'Relative stresses, global activation', fontsize=10)
# write title for panels 5 to 6
plt.text(-0.5, 1.7, 'Relative stress \n     increase', fontsize=10)
# # save plot to file
plt.savefig(figfolder + 'SC.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'SC.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure S3C, Relative detrend stress over time fullstim

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.arange(60)
x = x[::2]  # downsample data for nicer plotting
ymin = -0.1
ymax = 0.2
xticks = np.arange(0, 61, 20)  # define where the major ticks are gonna be
yticks = np.arange(ymin, ymax + 0.01, 0.1)
xlabel = 'time [min]'
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(5, 3))  # create figure and axes
plt.subplots_adjust(wspace=0.35, hspace=0.35)  # adjust space in between plots
# ******************************************************************************************************************************************

# Set up plot parameters for first panel
#######################################################################################################
ax = axes[0, 0]
color = colors_parent[1]
ylabel = 'doublet'
title = 'xx-Stress'
y = AR1to1d_fullstim_long["MSM_data"]["relsigma_xx_detrend"]
y = y[::2]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color)

ax.plot(x, sim_relstress_xx_1to1dfs[::2], color=color)


# Set up plot parameters for second panel
#######################################################################################################
ax = axes[0, 1]
color = colors_parent[1]
ylabel = None
title = 'yy-Stress'
y = AR1to1d_fullstim_long["MSM_data"]["relsigma_yy_detrend"]
y = y[::2]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color)

ax.plot(x, sim_relstress_yy_1to1dfs[::2], color=color)

# Set up plot parameters for third panel
#######################################################################################################
ax = axes[1, 0]
color = colors_parent[2]
ylabel = 'singlet'
title = None
y = AR1to1s_fullstim_long["MSM_data"]["relsigma_xx_detrend"]
y = y[::2]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color)



# Set up plot parameters for fourth panel
#######################################################################################################
ax = axes[1, 1]
color = colors_parent[2]
ylabel = None
title = None
y = AR1to1s_fullstim_long["MSM_data"]["relsigma_yy_detrend"]
y = y[::2]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color)

# Set up plot parameters for fifth panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'sigma'  # variable that goes on the y-axis
ax = axes[0, 2]  # define on which axis the plot goes
colors = [colors_parent[1], colors_parent[1]]     # defines colors
ymin = -0.2  # minimum value on y-axis
ymax = 0.6  # maximum value on y-axis
yticks = np.arange(-0.2, 0.61, 0.2)  # define where to put major ticks on y-axis
stat_annotation_offset = 0  # vertical offset of statistical annotation
ylabel = None  # which label to put on y-axis
title = None  # title of plot
box_pairs = [('sigma_xx', 'sigma_yy')] # which groups to perform statistical test on

# a little weird way to get the dataframe format I need for seaborn boxplots...
df1 = pd.DataFrame(df_d_fs['RSI_xx_detrend'])
df2 = pd.DataFrame(df_d_fs['RSI_yy_detrend'])
df1 = df1.transpose().reset_index(drop=True).transpose()
df2 = df2.transpose().reset_index(drop=True).transpose()

df_plot = pd.concat([df1, df2], axis=0)

df_plot.rename(columns={0: 'sigma'}, inplace=True)

keys_sx = ['sigma_xx' for i in range(n_d_fullstim)]
keys_sy = ['sigma_yy' for i in range(n_d_fullstim)]
keys = np.concatenate((keys_sx, keys_sy))
df_plot['keys'] = keys

# make plots
make_two_box_and_swarmplots(x, y, df_plot, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title, colors)

# Set up plot parameters for sixth panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'sigma'  # variable that goes on the y-axis
ax = axes[1, 2]  # define on which axis the plot goes
colors = [colors_parent[2], colors_parent_dark[2], colors_parent[2], colors_parent_dark[2]]     # defines colors
stat_annotation_offset = 0  # vertical offset of statistical annotation
ylabel = None  # which label to put on y-axis
title = None  # title of plot
box_pairs = [('sigma_xx', 'sigma_yy')]  # which groups to perform statistical test on

# a little weird way to get the dataframe format I need for seaborn boxplots...
df1 = pd.DataFrame(df_s_fs['RSI_xx_detrend'])
df2 = pd.DataFrame(df_s_fs['RSI_yy_detrend'])
df1 = df1.transpose().reset_index(drop=True).transpose()
df2 = df2.transpose().reset_index(drop=True).transpose()

df_plot = pd.concat([df1, df2], axis=0)

df_plot.rename(columns={0: 'sigma'}, inplace=True)

keys_sx = ['sigma_xx' for i in range(n_s_fullstim)]
keys_sy = ['sigma_yy' for i in range(n_s_fullstim)]
keys = np.concatenate((keys_sx, keys_sy))
df_plot['keys'] = keys


# make plots
make_two_box_and_swarmplots(x, y, df_plot, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title, colors)

# write title for panels 1 to 4
plt.text(-5.85, 1.82, 'Relative stresses, global activation', fontsize=10)
# write title for panels 5 to 6
plt.text(-0.5, 1.7, 'Relative stress \n     increase', fontsize=10)
# # save plot to file
plt.savefig(figfolder + 'SC_detrend.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'SC_detrend.svg', dpi=300, bbox_inches="tight")
plt.show()