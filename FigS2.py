# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 21:56:01 2021

@author: Artur Ruppel

"""
import matplotlib.image as mpimg
import os
import pandas as pd
import pickle
from plot_and_filter_functions import *
from skimage.filters import threshold_otsu

mpl.rcParams['font.size'] = 8


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


# define some colors for the plots
colors_parent = ['#026473', '#E3CC69', '#77C8A6', '#D96248']
colors_parent_dark = ['#01353D', '#564910', '#235741', '#A93B23']

figfolder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/_FigureS2/"
if not os.path.exists(figfolder):
    os.mkdir(figfolder)

# %% load data for plotting
folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"

AR1to1d_fullstim_long = pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_long.dat", "rb"))
AR1to1s_fullstim_long = pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_long.dat", "rb"))

# normalize simulated strain energy curve
sim_Es_1to1dfs = np.load(folder + "_FEM_simulations/strain_energy_doublets/strain_energy_halfstim_ar1to1d_1.0.npz")["energy"]
sim_Es_1to1sfs = np.load(folder + "_FEM_simulations/strain_energy_singlets/strain_energy_halfstim_ar1to1s_1.0.npz")["energy"]

sim_relEs_1to1dfs = sim_Es_1to1dfs / np.nanmean(sim_Es_1to1dfs[0:20]) - 1
sim_relEs_1to1sfs = sim_Es_1to1sfs / np.nanmean(sim_Es_1to1sfs[0:20]) - 1

doublet_FEM_simulation = pickle.load(open(folder + "_FEM_simulations/FEM_doublets.dat", "rb"))
singlet_FEM_simulation = pickle.load(open(folder + "_FEM_simulations/FEM_singlets.dat", "rb"))

# %% filter data to make sure that the baselines are stable
def filter_data_main(data, threshold, title, withactin=False):
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

    if withactin==True:
        data["actin_images"] = apply_filter(data["actin_images"], baselinefilter)

    new_N = np.sum(baselinefilter)
    print(title + ": " + str(baselinefilter.shape[0] - new_N) + " cells were filtered out")

    return data


threshold = 0.005

AR1to1d_fullstim_long = filter_data_main(AR1to1d_fullstim_long, threshold, "AR1to1d_fullstim_long")
AR1to1s_fullstim_long = filter_data_main(AR1to1s_fullstim_long, threshold, "AR1to1s_fullstim_long")

n_d_fullstim = AR1to1d_fullstim_long["TFM_data"]["Tx"].shape[3]
n_s_fullstim = AR1to1s_fullstim_long["TFM_data"]["Tx"].shape[3]

# %% plot figure 2SA

# prepare data
Tx_1to1d_fs = AR1to1d_fullstim_long["TFM_data"]["Tx"]
Ty_1to1d_fs = AR1to1d_fullstim_long["TFM_data"]["Ty"]
Tx_1to1s_fs = AR1to1s_fullstim_long["TFM_data"]["Tx"]
Ty_1to1s_fs = AR1to1s_fullstim_long["TFM_data"]["Ty"]

# calculate amplitudes
T_1to1d_fs = np.sqrt(Tx_1to1d_fs ** 2 + Ty_1to1d_fs ** 2)
T_1to1s_fs = np.sqrt(Tx_1to1s_fs ** 2 + Ty_1to1s_fs ** 2)

# calculate difference between after and before photoactivation
Tx_1to1d_fs_diff = np.nanmean(Tx_1to1d_fs[:, :, 32, :] - Tx_1to1d_fs[:, :, 20, :], axis=2)
Ty_1to1d_fs_diff = np.nanmean(Ty_1to1d_fs[:, :, 32, :] - Ty_1to1d_fs[:, :, 20, :], axis=2)
T_1to1d_fs_diff = np.nanmean(T_1to1d_fs[:, :, 32, :] - T_1to1d_fs[:, :, 20, :], axis=2)

Tx_1to1s_fs_diff = np.nanmean(Tx_1to1s_fs[:, :, 32, :] - Tx_1to1s_fs[:, :, 20, :], axis=2)
Ty_1to1s_fs_diff = np.nanmean(Ty_1to1s_fs[:, :, 32, :] - Ty_1to1s_fs[:, :, 20, :], axis=2)
T_1to1s_fs_diff = np.nanmean(T_1to1s_fs[:, :, 32, :] - T_1to1s_fs[:, :, 20, :], axis=2)

# crop maps
crop_start = 8
crop_end = 84

Tx_1to1d_fs_diff_crop = Tx_1to1d_fs_diff[crop_start:crop_end, crop_start:crop_end] * 1e-3  # convert to kPa
Ty_1to1d_fs_diff_crop = Ty_1to1d_fs_diff[crop_start:crop_end, crop_start:crop_end] * 1e-3
T_1to1d_fs_diff_crop = T_1to1d_fs_diff[crop_start:crop_end, crop_start:crop_end] * 1e-3

Tx_1to1s_fs_diff_crop = Tx_1to1s_fs_diff[crop_start:crop_end, crop_start:crop_end] * 1e-3
Ty_1to1s_fs_diff_crop = Ty_1to1s_fs_diff[crop_start:crop_end, crop_start:crop_end] * 1e-3
T_1to1s_fs_diff_crop = T_1to1s_fs_diff[crop_start:crop_end, crop_start:crop_end] * 1e-3

# prepare simulated maps
Tx_1to1d_fs_sim = doublet_FEM_simulation["feedback1.0"]["t_x"]
Ty_1to1d_fs_sim = doublet_FEM_simulation["feedback1.0"]["t_y"]

Tx_1to1s_fs_sim = singlet_FEM_simulation["feedback1.0"]["t_x"]
Ty_1to1s_fs_sim = singlet_FEM_simulation["feedback1.0"]["t_y"]

# calculate amplitudes
T_1to1d_fs_sim = np.sqrt(Tx_1to1d_fs_sim ** 2 + Ty_1to1d_fs_sim ** 2)
T_1to1s_fs_sim = np.sqrt(Tx_1to1s_fs_sim ** 2 + Ty_1to1s_fs_sim ** 2)

Tx_1to1d_fs_sim_diff = (Tx_1to1d_fs_sim[:, :, 32] - Tx_1to1d_fs_sim[:, :, 20]) * 1e-3  # convert to kPa
Ty_1to1d_fs_sim_diff = (Ty_1to1d_fs_sim[:, :, 32] - Ty_1to1d_fs_sim[:, :, 20]) * 1e-3
T_1to1d_fs_sim_diff = (T_1to1d_fs_sim[:, :, 32] - T_1to1d_fs_sim[:, :, 20]) * 1e-3

Tx_1to1s_fs_sim_diff = (Tx_1to1s_fs_sim[:, :, 32] - Tx_1to1s_fs_sim[:, :, 20]) * 1e-3  # convert to kPa
Ty_1to1s_fs_sim_diff = (Ty_1to1s_fs_sim[:, :, 32] - Ty_1to1s_fs_sim[:, :, 20]) * 1e-3
T_1to1s_fs_sim_diff = (T_1to1s_fs_sim[:, :, 32] - T_1to1s_fs_sim[:, :, 20]) * 1e-3

# # pad simulation maps to make shapes equal. only works when shape is a square
paddingdistance = int((Tx_1to1d_fs_diff_crop.shape[0] - Tx_1to1d_fs_sim_diff.shape[0]) / 2)
Tx_1to1d_fs_sim_diff = np.pad(Tx_1to1d_fs_sim_diff, (paddingdistance, paddingdistance), 'constant', constant_values=(0, 0))
Ty_1to1d_fs_sim_diff = np.pad(Ty_1to1d_fs_sim_diff, (paddingdistance, paddingdistance), 'constant', constant_values=(0, 0))
T_1to1d_fs_sim_diff = np.pad(T_1to1d_fs_sim_diff, (paddingdistance, paddingdistance), 'constant', constant_values=(0, 0))

Tx_1to1s_fs_sim_diff = np.pad(Tx_1to1s_fs_sim_diff, (paddingdistance, paddingdistance), 'constant', constant_values=(0, 0))
Ty_1to1s_fs_sim_diff = np.pad(Ty_1to1s_fs_sim_diff, (paddingdistance, paddingdistance), 'constant', constant_values=(0, 0))
T_1to1s_fs_sim_diff = np.pad(T_1to1s_fs_sim_diff, (paddingdistance, paddingdistance), 'constant', constant_values=(0, 0))

# set up plot parameters
# *****************************************************************************
pixelsize = 0.864  # in µm
pmax = 0.3  # kPa
pmin = -0.3
suptitle = '$\mathrm{\Delta}$ Traction forces'

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4, 3.5))

im = plot_forcemaps_diff(axes[0, 0], Tx_1to1d_fs_diff_crop, Ty_1to1d_fs_diff_crop, T_1to1d_fs_diff_crop, pixelsize, pmax, pmin, n=2, scale=1)
plot_forcemaps_diff(axes[0, 1], Tx_1to1d_fs_sim_diff, Ty_1to1d_fs_sim_diff, T_1to1d_fs_sim_diff, pixelsize, pmax, pmin, n=1, scale=1)
plot_forcemaps_diff(axes[1, 0], Tx_1to1s_fs_diff_crop, Ty_1to1s_fs_diff_crop, T_1to1s_fs_diff_crop, pixelsize, pmax, pmin, n=2, scale=1)
plot_forcemaps_diff(axes[1, 1], Tx_1to1s_fs_sim_diff, Ty_1to1s_fs_sim_diff, T_1to1s_fs_sim_diff, pixelsize, pmax, pmin, n=1, scale=1)

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
plt.suptitle(suptitle, y=0.98, x=0.44)

plt.figure(1).text(0.21, 0.89, "TFM data")
plt.figure(1).text(0.48, 0.89, "FEM simulation")

plt.figure(1).text(0.24, 0.84, "n=" + str(n_d_fullstim))
plt.figure(1).text(0.24, 0.46, "n=" + str(n_s_fullstim))

# draw pattern
for ax in axes.flat:
    draw_pattern_1to1(ax)

# save figure
fig.savefig(figfolder + 'A.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'A.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 2SB, xx-stress maps

# prepare data first
sigma_xx_1to1d = AR1to1d_fullstim_long["MSM_data"]["sigma_xx"]
sigma_xx_1to1s = AR1to1s_fullstim_long["MSM_data"]["sigma_xx"]
# concatenate MSM maps from different experiments and calculate average maps over first 20 frames and all cells to get average maps
Tx_1to1d_fs_diff = np.nanmean(Tx_1to1d_fs[:, :, 32, :] - Tx_1to1d_fs[:, :, 20, :], axis=2)

sigma_xx_1to1d_diff = np.nanmean(sigma_xx_1to1d[:, :, 32, :] - sigma_xx_1to1d[:, :, 20, :], axis=2)
sigma_xx_1to1s_diff = np.nanmean(sigma_xx_1to1s[:, :, 32, :] - sigma_xx_1to1s[:, :, 20, :], axis=2)

# load simulation data
sigma_xx_sim_d = (doublet_FEM_simulation["feedback1.0"]["sigma_xx"][:, :, 32] - doublet_FEM_simulation["feedback1.0"]["sigma_xx"][:, :, 20]) * 1e3
sigma_xx_sim_s = (singlet_FEM_simulation["feedback1.0"]["sigma_xx"][:, :, 32] - singlet_FEM_simulation["feedback1.0"]["sigma_xx"][:, :, 20]) * 1e3

# convert NaN to 0 to have black background
sigma_xx_1to1d_diff[np.isnan(sigma_xx_1to1d_diff)] = 0
sigma_xx_1to1s_diff[np.isnan(sigma_xx_1to1s_diff)] = 0

# crop maps
crop_start = 8
crop_end = 84

sigma_xx_1to1d_diff_crop = sigma_xx_1to1d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3
sigma_xx_1to1s_diff_crop = sigma_xx_1to1s_diff[crop_start:crop_end, crop_start:crop_end] * 1e3

# pad with 0 to match shape of experimental data
paddingdistance_x = int((sigma_xx_1to1d_diff_crop.shape[0] - sigma_xx_sim_d.shape[0]) / 2)
paddingdistance_y = int((sigma_xx_1to1d_diff_crop.shape[0] - sigma_xx_sim_d.shape[1]) / 2)
sigma_xx_sim_d = np.pad(sigma_xx_sim_d, (paddingdistance_x, paddingdistance_y), 'constant', constant_values=(0, 0))
sigma_xx_sim_s = np.pad(sigma_xx_sim_s, (paddingdistance_x, paddingdistance_y), 'constant', constant_values=(0, 0))

# set up plot parameters
# ******************************************************************************************************************************************
pixelsize = 0.864  # in µm
pmin = -1  # in mN/m
pmax = 1  # in mN/m
axtitle = 'mN/m'  # unit of colorbar
suptitle = '$\mathrm{\Delta \sigma _{xx}(x,y)}$'  # title of plot

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4, 3.5))  # create figure and axes
plt.subplots_adjust(wspace=0.02, hspace=-0.06)  # adjust space in between plots
# ******************************************************************************************************************************************

im = plot_stressmaps(axes[0, 0], sigma_xx_1to1d_diff_crop, pixelsize, pmax, pmin, cmap="seismic")
plot_stressmaps(axes[0, 1], sigma_xx_sim_d, pixelsize, pmax, pmin, cmap="seismic")
plot_stressmaps(axes[1, 0], sigma_xx_1to1s_diff_crop, pixelsize, pmax, pmin, cmap="seismic")
plot_stressmaps(axes[1, 1], sigma_xx_sim_s, pixelsize, pmax, pmin, cmap="seismic")

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
plt.suptitle(suptitle, y=0.98, x=0.44)

plt.figure(1).text(0.21, 0.89, "MSM data")
plt.figure(1).text(0.48, 0.89, "FEM simulation")

# add annotations
plt.text(0.24, 0.83, 'n=' + str(n_d_fullstim), transform=plt.figure(1).transFigure)
plt.text(0.24, 0.455, 'n=' + str(n_s_fullstim), transform=plt.figure(1).transFigure)

# draw pattern
for ax in axes.flat:
    draw_pattern_1to1(ax)

# save figure
fig.savefig(figfolder + 'B.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'B.svg', dpi=300, bbox_inches="tight")
plt.show()


# %% plot figure 2SC xx-stress x-profile

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.linspace(-40, 40, 92)
x = x[::2]  # downsample data for nicer plotting
xticks = np.arange(-40, 40.1, 20)  # define where the major ticks are gonna be
xlabel = 'position [µm]'
ymin = -0.2
ymax = 0.51
yticks = np.arange(ymin, ymax + 0.001, 0.1)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(1.4, 3.3))  # create figure and axes
plt.subplots_adjust(wspace=0.4, hspace=0.35)  # adjust space in between plots
# ******************************************************************************************************************************************

# Set up plot parameters for first panel
#######################################################################################################
ax = axes[0]
color = colors_parent[1]
ylabel = None
title = '$\mathrm{\Delta \sigma _{xx}(x)}$ [mN/m]'
y = AR1to1d_fullstim_long["MSM_data"]["sigma_xx_x_profile_increase"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False, xmin=-40, xmax=40)

y_sim = doublet_FEM_simulation["feedback1.0"]["sigma_xx_x_profile_increase"]* 1e3
x_sim = np.linspace(-22.5, 22.5, y_sim.shape[0])
ax.plot(x_sim, y_sim, color=color)

# # Set up plot parameters for second panel
# #######################################################################################################
ax = axes[1]
color = colors_parent[2]
ylabel = None
title = None
y = AR1to1s_fullstim_long["MSM_data"]["sigma_xx_x_profile_increase"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False, xmin=-40, xmax=40)

y_sim = singlet_FEM_simulation["feedback1.0"]["sigma_xx_x_profile_increase"] * 1e3
x_sim = np.linspace(-22.5, 22.5, y_sim.shape[0])
ax.plot(x_sim, y_sim, color=color)

for ax in axes.flat:
    # add line at y=0 for visualisation
    ax.plot([x[0], x[-1]], [0, 0], linewidth=0.5, linestyle=":", color="grey")


plt.savefig(figfolder + 'C.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'C.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 3SD, Relative strain energy over time and relative energy increase

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.arange(60)
x = x[::2]  # downsample data for nicer plotting
ymin = -0.1
ymax = 0.3
xticks = np.arange(0, 61, 20)  # define where the major ticks are gonna be
yticks = np.arange(ymin, ymax + 0.01, 0.1)
xlabel = 'time [min]'
xticklabels = ['doublet', 'singlet']  # which labels to put on x-axis
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(1.4, 2.8)) # create figure and axes

plt.subplots_adjust(wspace=0.35, hspace=0.35)  # adjust space in between plots
# ******************************************************************************************************************************************


# Set up plot parameters for first panel
#######################################################################################################
ax = axes[0]
color = colors_parent[1]
ylabel = 'doublet'
title = 'Relative strain \n energy'
y = AR1to1d_fullstim_long["TFM_data"]["Es"]
y = (y - np.nanmean(y[0:20], axis=0)) / np.nanmean(y[0:20], axis=(0, 1))  # normalize
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, xmax=60)

ax.plot(sim_relEs_1to1dfs, color=color)

# Set up plot parameters for second panel
#######################################################################################################
ax = axes[1]
color = colors_parent[2]
ylabel = 'singlet'
title = None
y = AR1to1s_fullstim_long["TFM_data"]["Es"]
y = (y - np.nanmean(y[0:20], axis=0)) / np.nanmean(y[0:20], axis=(0, 1))  # normalize
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, xmax=60)

ax.plot(sim_relEs_1to1sfs, color=color)

for ax in axes.flat:
    # add line at y=0 for visualisation
    ax.plot([x[0], x[-1]], [0, 0], linewidth=0.5, linestyle=":", color="grey")

plt.savefig(figfolder + 'D.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'D.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 2SE, yy-stress maps

# prepare data first
sigma_yy_1to1d = AR1to1d_fullstim_long["MSM_data"]["sigma_yy"]
sigma_yy_1to1s = AR1to1s_fullstim_long["MSM_data"]["sigma_yy"]
# concatenate MSM maps from different experiments and calculate average maps over first 20 frames and all cells to get average maps
Tx_1to1d_fs_diff = np.nanmean(Tx_1to1d_fs[:, :, 32, :] - Tx_1to1d_fs[:, :, 20, :], axis=2)

sigma_yy_1to1d_diff = np.nanmean(sigma_yy_1to1d[:, :, 32, :] - sigma_yy_1to1d[:, :, 20, :], axis=2)
sigma_yy_1to1s_diff = np.nanmean(sigma_yy_1to1s[:, :, 32, :] - sigma_yy_1to1s[:, :, 20, :], axis=2)

# load simulation data
sigma_yy_sim_d = (doublet_FEM_simulation["feedback1.0"]["sigma_yy"][:, :, 32] - doublet_FEM_simulation["feedback1.0"]["sigma_yy"][:, :, 20]) * 1e3
sigma_yy_sim_s = (singlet_FEM_simulation["feedback1.0"]["sigma_yy"][:, :, 32] - singlet_FEM_simulation["feedback1.0"]["sigma_yy"][:, :, 20]) * 1e3

# convert NaN to 0 to have black background
sigma_yy_1to1d_diff[np.isnan(sigma_yy_1to1d_diff)] = 0
sigma_yy_1to1s_diff[np.isnan(sigma_yy_1to1s_diff)] = 0

# crop maps
crop_start = 8
crop_end = 84

sigma_yy_1to1d_diff_crop = sigma_yy_1to1d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3
sigma_yy_1to1s_diff_crop = sigma_yy_1to1s_diff[crop_start:crop_end, crop_start:crop_end] * 1e3

# pad with 0 to match shape of experimental data
paddingdistance_x = int((sigma_yy_1to1d_diff_crop.shape[0] - sigma_yy_sim_d.shape[0]) / 2)
paddingdistance_y = int((sigma_yy_1to1d_diff_crop.shape[0] - sigma_yy_sim_d.shape[1]) / 2)
sigma_yy_sim_d = np.pad(sigma_yy_sim_d, (paddingdistance_x, paddingdistance_y), 'constant', constant_values=(0, 0))
sigma_yy_sim_s = np.pad(sigma_yy_sim_s, (paddingdistance_x, paddingdistance_y), 'constant', constant_values=(0, 0))

# set up plot parameters
# ******************************************************************************************************************************************
pixelsize = 0.864  # in µm
pmin = -1  # in mN/m
pmax = 1  # in mN/m
axtitle = 'mN/m'  # unit of colorbar
suptitle = '$\mathrm{\Delta \sigma _{yy}(x,y)}$'  # title of plot

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4, 3.5))  # create figure and axes
plt.subplots_adjust(wspace=0.02, hspace=-0.06)  # adjust space in between plots
# ******************************************************************************************************************************************

im = plot_stressmaps(axes[0, 0], sigma_yy_1to1d_diff_crop, pixelsize, pmax, pmin, cmap="seismic")
plot_stressmaps(axes[0, 1], sigma_yy_sim_d, pixelsize, pmax, pmin, cmap="seismic")
plot_stressmaps(axes[1, 0], sigma_yy_1to1s_diff_crop, pixelsize, pmax, pmin, cmap="seismic")
plot_stressmaps(axes[1, 1], sigma_yy_sim_s, pixelsize, pmax, pmin, cmap="seismic")

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
plt.suptitle(suptitle, y=0.98, x=0.44)

plt.figure(1).text(0.21, 0.89, "MSM data")
plt.figure(1).text(0.48, 0.89, "FEM simulation")

# add annotations
plt.text(0.24, 0.83, 'n=' + str(n_d_fullstim), transform=plt.figure(1).transFigure)
plt.text(0.24, 0.455, 'n=' + str(n_s_fullstim), transform=plt.figure(1).transFigure)

# draw pattern
for ax in axes.flat:
    draw_pattern_1to1(ax)

# save figure
fig.savefig(figfolder + 'E.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'E.svg', dpi=300, bbox_inches="tight")
plt.show()


# %% plot figure 2SF yy-stress x-profile

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.linspace(-40, 40, 92)
x = x[::2]  # downsample data for nicer plotting
xticks = np.arange(-40, 40.1, 20)  # define where the major ticks are gonna be
xlabel = 'position [µm]'
ymin = -0.2
ymax = 0.51
yticks = np.arange(ymin, ymax + 0.001, 0.1)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(1.4, 3.3))  # create figure and axes
plt.subplots_adjust(wspace=0.4, hspace=0.35)  # adjust space in between plots
# ******************************************************************************************************************************************

# Set up plot parameters for first panel
#######################################################################################################
ax = axes[0]
color = colors_parent[1]
ylabel = None
title = '$\mathrm{\Delta \sigma _{yy}(x)}$ [mN/m]'
y = AR1to1d_fullstim_long["MSM_data"]["sigma_yy_x_profile_increase"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False, xmin=-40, xmax=40)

y_sim = doublet_FEM_simulation["feedback1.0"]["sigma_yy_x_profile_increase"]* 1e3
x_sim = np.linspace(-22.5, 22.5, y_sim.shape[0])
ax.plot(x_sim, y_sim, color=color)

# # Set up plot parameters for second panel
# #######################################################################################################
ax = axes[1]
color = colors_parent[2]
ylabel = None
title = None
y = AR1to1s_fullstim_long["MSM_data"]["sigma_yy_x_profile_increase"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False, xmin=-40, xmax=40)

y_sim = singlet_FEM_simulation["feedback1.0"]["sigma_yy_x_profile_increase"] * 1e3
x_sim = np.linspace(-22.5, 22.5, y_sim.shape[0])
ax.plot(x_sim, y_sim, color=color)

for ax in axes.flat:
    # add line at y=0 for visualisation
    ax.plot([x[0], x[-1]], [0, 0], linewidth=0.5, linestyle=":", color="grey")


plt.savefig(figfolder + 'F.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'F.svg', dpi=300, bbox_inches="tight")
plt.show()