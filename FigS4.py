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

figfolder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/_FigureS4/"
if not os.path.exists(figfolder):
    os.mkdir(figfolder)

# %% load data for plotting
folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"

AR1to1d_halfstim = pickle.load(open(folder + "analysed_data/AR1to1d_halfstim.dat", "rb"))
AR1to1s_halfstim = pickle.load(open(folder + "analysed_data/AR1to1s_halfstim.dat", "rb"))

# load simulation data for plotting
doublet_FEM_simulation = pickle.load(open(folder + "_FEM_simulations/FEM_doublets.dat", "rb"))
singlet_FEM_simulation = pickle.load(open(folder + "_FEM_simulations/FEM_singlets.dat", "rb"))

sim_Es_1to1dhs = 0.5 * (doublet_FEM_simulation["feedback0.0"]["t_x"] * doublet_FEM_simulation["feedback0.0"]["u_substrate"] +
                 doublet_FEM_simulation["feedback0.0"]["t_y"] * doublet_FEM_simulation["feedback0.0"]["v_substrate"])
sim_Es_1to1shs = 0.5 * (singlet_FEM_simulation["feedback0.0"]["t_x"] * singlet_FEM_simulation["feedback0.0"]["u_substrate"] +
                 singlet_FEM_simulation["feedback0.0"]["t_y"] * singlet_FEM_simulation["feedback0.0"]["v_substrate"])

sim_Es_1to1dhs_left = np.sum(sim_Es_1to1dhs[:, 0:26], axis=(0, 1))
sim_Es_1to1dhs_right = np.sum(sim_Es_1to1dhs[:, 26:52], axis=(0, 1))
sim_Es_1to1shs_left = np.sum(sim_Es_1to1shs[:, 0:26], axis=(0, 1))
sim_Es_1to1shs_right = np.sum(sim_Es_1to1shs[:, 26:52], axis=(0, 1))

# normalize simulated strain energy curve
sim_relEs_1to1dhs_left = sim_Es_1to1dhs_left / np.nanmean(sim_Es_1to1dhs_left[0:20]) - 1
sim_relEs_1to1dhs_right = sim_Es_1to1dhs_right / np.nanmean(sim_Es_1to1dhs_right[0:20]) - 1
sim_relEs_1to1shs_left = sim_Es_1to1shs_left / np.nanmean(sim_Es_1to1shs_left[0:20]) - 1
sim_relEs_1to1shs_right = sim_Es_1to1shs_right / np.nanmean(sim_Es_1to1shs_right[0:20]) - 1


# normalize simulated strain energy curve
sim_Es_1to1s_hs_left_fluidization = np.load(folder + "_FEM_simulations/strain_energy_singlets/strain_energy_fluidization.npz")["Es_l"]
sim_Es_1to1s_hs_right_fluidization = np.load(folder + "_FEM_simulations/strain_energy_singlets/strain_energy_fluidization.npz")["Es_r"]

sim_relEs_1to1s_hs_left_fluidization = sim_Es_1to1s_hs_left_fluidization / np.nanmean(sim_Es_1to1s_hs_left_fluidization[0:20]) - 1
sim_relEs_1to1s_hs_right_fluidization = sim_Es_1to1s_hs_right_fluidization / np.nanmean(sim_Es_1to1s_hs_right_fluidization[0:20]) - 1




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
AR1to1d_halfstim = filter_data_main(AR1to1d_halfstim, threshold, "AR1to1d_halfstim")
AR1to1s_halfstim = filter_data_main(AR1to1s_halfstim, threshold, "AR1to1s_halfstim")

n_d_halfstim = AR1to1d_halfstim["TFM_data"]["Tx"].shape[3]
n_s_halfstim = AR1to1s_halfstim["TFM_data"]["Tx"].shape[3]

# %% plot figure 3A, TFM differences no coupling

# prepare data
Tx_1to1d_hs = AR1to1d_halfstim["TFM_data"]["Tx"]
Ty_1to1d_hs = AR1to1d_halfstim["TFM_data"]["Ty"]

Tx_1to1s_hs = AR1to1s_halfstim["TFM_data"]["Tx"]
Ty_1to1s_hs = AR1to1s_halfstim["TFM_data"]["Ty"]

# calculate amplitudes
T_1to1d_hs = np.sqrt(Tx_1to1d_hs ** 2 + Ty_1to1d_hs ** 2)
T_1to1s_hs = np.sqrt(Tx_1to1s_hs ** 2 + Ty_1to1s_hs ** 2)

# calculate difference between after and before photoactivation
Tx_1to1d_hs_diff = np.nanmean(Tx_1to1d_hs[:, :, 32, :] - Tx_1to1d_hs[:, :, 20, :], axis=2)
Ty_1to1d_hs_diff = np.nanmean(Ty_1to1d_hs[:, :, 32, :] - Ty_1to1d_hs[:, :, 20, :], axis=2)
T_1to1d_hs_diff = np.nanmean(T_1to1d_hs[:, :, 32, :] - T_1to1d_hs[:, :, 20, :], axis=2)

Tx_1to1s_hs_diff = np.nanmean(Tx_1to1s_hs[:, :, 32, :] - Tx_1to1s_hs[:, :, 20, :], axis=2)
Ty_1to1s_hs_diff = np.nanmean(Ty_1to1s_hs[:, :, 32, :] - Ty_1to1s_hs[:, :, 20, :], axis=2)
T_1to1s_hs_diff = np.nanmean(T_1to1s_hs[:, :, 32, :] - T_1to1s_hs[:, :, 20, :], axis=2)

# crop maps
crop_start = 8
crop_end = 84

Tx_1to1d_hs_diff_crop = Tx_1to1d_hs_diff[crop_start:crop_end, crop_start:crop_end] * 1e-3  # convert to kPa
Ty_1to1d_hs_diff_crop = Ty_1to1d_hs_diff[crop_start:crop_end, crop_start:crop_end] * 1e-3
T_1to1d_hs_diff_crop = T_1to1d_hs_diff[crop_start:crop_end, crop_start:crop_end] * 1e-3

Tx_1to1s_hs_diff_crop = Tx_1to1s_hs_diff[crop_start:crop_end, crop_start:crop_end] * 1e-3
Ty_1to1s_hs_diff_crop = Ty_1to1s_hs_diff[crop_start:crop_end, crop_start:crop_end] * 1e-3
T_1to1s_hs_diff_crop = T_1to1s_hs_diff[crop_start:crop_end, crop_start:crop_end] * 1e-3

# prepare simulated maps
Tx_1to1d_hs_sim = doublet_FEM_simulation["feedback0.0"]["t_x"]
Ty_1to1d_hs_sim = doublet_FEM_simulation["feedback0.0"]["t_y"]

Tx_1to1s_hs_sim = singlet_FEM_simulation["feedback0.0"]["t_x"]
Ty_1to1s_hs_sim = singlet_FEM_simulation["feedback0.0"]["t_y"]

# calculate amplitudes
T_1to1d_hs_sim = np.sqrt(Tx_1to1d_hs_sim ** 2 + Ty_1to1d_hs_sim ** 2)
T_1to1s_hs_sim = np.sqrt(Tx_1to1s_hs_sim ** 2 + Ty_1to1s_hs_sim ** 2)

Tx_1to1d_hs_sim_diff = (Tx_1to1d_hs_sim[:, :, 32] - Tx_1to1d_hs_sim[:, :, 20]) * 1e-3  # convert to kPa
Ty_1to1d_hs_sim_diff = (Ty_1to1d_hs_sim[:, :, 32] - Ty_1to1d_hs_sim[:, :, 20]) * 1e-3
T_1to1d_hs_sim_diff = (T_1to1d_hs_sim[:, :, 32] - T_1to1d_hs_sim[:, :, 20]) * 1e-3

Tx_1to1s_hs_sim_diff = (Tx_1to1s_hs_sim[:, :, 32] - Tx_1to1s_hs_sim[:, :, 20]) * 1e-3  # convert to kPa
Ty_1to1s_hs_sim_diff = (Ty_1to1s_hs_sim[:, :, 32] - Ty_1to1s_hs_sim[:, :, 20]) * 1e-3
T_1to1s_hs_sim_diff = (T_1to1s_hs_sim[:, :, 32] - T_1to1s_hs_sim[:, :, 20]) * 1e-3

# # pad simulation maps to make shapes equal. only works when shape is a square
paddingdistance = int((Tx_1to1d_hs_diff_crop.shape[0] - Tx_1to1d_hs_sim_diff.shape[0]) / 2)
Tx_1to1d_hs_sim_diff = np.pad(Tx_1to1d_hs_sim_diff, (paddingdistance, paddingdistance), 'constant', constant_values=(0, 0))
Ty_1to1d_hs_sim_diff = np.pad(Ty_1to1d_hs_sim_diff, (paddingdistance, paddingdistance), 'constant', constant_values=(0, 0))
T_1to1d_hs_sim_diff = np.pad(T_1to1d_hs_sim_diff, (paddingdistance, paddingdistance), 'constant', constant_values=(0, 0))

Tx_1to1s_hs_sim_diff = np.pad(Tx_1to1s_hs_sim_diff, (paddingdistance, paddingdistance), 'constant', constant_values=(0, 0))
Ty_1to1s_hs_sim_diff = np.pad(Ty_1to1s_hs_sim_diff, (paddingdistance, paddingdistance), 'constant', constant_values=(0, 0))
T_1to1s_hs_sim_diff = np.pad(T_1to1s_hs_sim_diff, (paddingdistance, paddingdistance), 'constant', constant_values=(0, 0))

# set up plot parameters
# *****************************************************************************
n = 4  # every nth arrow will be plotted
pixelsize = 0.864  # in µm
pmax = 0.3  # kPa
pmin = -0.3

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4, 3))

im = plot_forcemaps_diff(axes[0, 0], Tx_1to1d_hs_diff_crop, Ty_1to1d_hs_diff_crop, T_1to1d_hs_diff_crop, pixelsize, pmax, pmin, n=2,
                         scale=1)
plot_forcemaps_diff(axes[0, 1], Tx_1to1d_hs_sim_diff, Ty_1to1d_hs_sim_diff, T_1to1d_hs_sim_diff, pixelsize, pmax, pmin, n=1, scale=1)
plot_forcemaps_diff(axes[1, 0], Tx_1to1s_hs_diff_crop, Ty_1to1s_hs_diff_crop, T_1to1s_hs_diff_crop, pixelsize, pmax, pmin, n=2, scale=1)
plot_forcemaps_diff(axes[1, 1], Tx_1to1s_hs_sim_diff, Ty_1to1s_hs_sim_diff, T_1to1s_hs_sim_diff, pixelsize, pmax, pmin, n=1, scale=1)

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

plt.figure(1).text(0.21, 0.89, "TFM data")
plt.figure(1).text(0.48, 0.89, "FEM simulation")

plt.figure(1).text(0.24, 0.84, "n=" + str(n_d_halfstim))
plt.figure(1).text(0.24, 0.46, "n=" + str(n_s_halfstim))

# draw pattern
for ax in axes.flat:
    draw_pattern_1to1(ax)

# save figure
fig.savefig(figfolder + 'A.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'A.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 3B, Relative strain energy over time

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.arange(60)
x = x[::2]  # downsample data for nicer plotting
xticks = np.arange(0, 61, 20)  # define where the major ticks are gonna be
ymin = -0.1
ymax = 0.2
yticks = np.arange(ymin, ymax + 0.01, 0.1)
xlabel = 'time [min]'
xticklabels = ['left', 'right', 'left', 'right']  # which labels to put on x-axis
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4, 2))  # create figure and axes
plt.subplots_adjust(wspace=0.35, hspace=0.35)  # adjust space in between plots

# ******************************************************************************************************************************************

# Set up plot parameters for first panel
#######################################################################################################
ax = axes[0]
colors = [colors_parent[1], colors_parent_dark[1]]
ylabel = None
title = 'Relative strain \n energy'
y1 = AR1to1d_halfstim["TFM_data"]["relEs_left"]
y2 = AR1to1d_halfstim["TFM_data"]["relEs_right"]
y1 = y1[::2, :]
y2 = y2[::2, :]

# make plots
plot_two_values_over_time(x, y1, y2, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors)

# add line at y=0 for visualisation
ax.plot([x[0], x[-1]], [0, 0], linewidth=0.5, linestyle=":", color="grey")

ax.plot(sim_relEs_1to1dhs_left, color=colors_parent[1])
ax.plot(sim_relEs_1to1dhs_right, color=colors_parent_dark[1])

# # Set up plot parameters for second panel
# #######################################################################################################
ax = axes[1]
colors = [colors_parent[2], colors_parent_dark[2]]
ylabel = None
title = 'Relative strain \n energy'
y1 = AR1to1s_halfstim["TFM_data"]["relEs_left"]
y2 = AR1to1s_halfstim["TFM_data"]["relEs_right"]
y1 = y1[::2, :]
y2 = y2[::2, :]

# make plots
plot_two_values_over_time(x, y1, y2, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors)

# add line at y=0 for visualisation
ax.plot([x[0], x[-1]], [0, 0], linewidth=0.5, linestyle=":", color="grey")

ax.plot(sim_relEs_1to1shs_left, color=colors_parent[2])
ax.plot(sim_relEs_1to1shs_right, color=colors_parent_dark[2])

plt.savefig(figfolder + 'B.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'B.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 3D, Relative strain energy over time fluidization

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.arange(60)
x = x[::2]  # downsample data for nicer plotting
xticks = np.arange(0, 61, 20)  # define where the major ticks are gonna be
xlabel = 'time [min]'
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4, 2)) # create figure and axes

plt.subplots_adjust(wspace=0.35, hspace=0.35)  # adjust space in between plots
# ******************************************************************************************************************************************


# Set up plot parameters for first panel
#######################################################################################################
ymin = -0.8
ymax = 0.8
yticks = np.arange(ymin, ymax + 0.01, 0.8)
ax = axes[0]
colors = [colors_parent[2], colors_parent_dark[2]]
ylabel = None
title = "FEM simulation"
titleoffset=3.5

ax.plot(sim_relEs_1to1s_hs_left_fluidization, color=colors[0])
ax.plot(sim_relEs_1to1s_hs_right_fluidization, color=colors[1])

# ax.set_xlabel(xlabel=xlabel, labelpad=xlabeloffset)
# ax.set_ylabel(ylabel=ylabel, labelpad=ylabeloffset)
ax.set_title(label=title, pad=titleoffset)

# add anotations for opto pulses
ax.axline((20, ymin), (20, ymax), linewidth=0.1, color="cyan")

# set ticks
ax.xaxis.set_ticks(xticks)
ax.yaxis.set_ticks(yticks)

# provide info on tick parameters
ax.minorticks_on()
ax.tick_params(direction='in', which='minor', length=3, bottom=True, top=False, left=True, right=True)
ax.tick_params(direction='in', which='major', length=6, bottom=True, top=False, left=True, right=True)

# set limits
ax.set_ylim(ymin=ymin, ymax=ymax)

# Set up plot parameters for second panel
#######################################################################################################
ax = axes[1]
colors = [colors_parent[2], colors_parent_dark[2]]
ylabel = None
title = "TFM data"
ymin = -0.1
ymax = 0.1
yticks = np.arange(ymin, ymax + 0.01, 0.1)
y1 = AR1to1s_halfstim["TFM_data"]["relEs_left"]
y2 = AR1to1s_halfstim["TFM_data"]["relEs_right"]
y1 = y1[::2, :]
y2 = y2[::2, :]

# make plots
plot_two_values_over_time(x, y1, y2, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors)

# add line at y=0 for visualisation
ax.plot([x[0], x[-1]], [0, 0], linewidth=0.5, linestyle=":", color="grey")

for ax in axes.flat:
    # add line at y=0 for visualisation
    ax.plot([x[0], x[-1]], [0, 0], linewidth=0.5, linestyle=":", color="grey")

plt.suptitle('Relative strain energy', y=1.1)

plt.savefig(figfolder + 'D.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'D.svg', dpi=300, bbox_inches="tight")
plt.show()

