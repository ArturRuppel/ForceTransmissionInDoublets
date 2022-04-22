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

figfolder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/_FigureTS1/"
if not os.path.exists(figfolder):
    os.mkdir(figfolder)

# %% load data for plotting
folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"

AR1to1d_fullstim_long = pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_long.dat", "rb"))
AR1to1s_fullstim_long = pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_long.dat", "rb"))
AR1to1d_halfstim = pickle.load(open(folder + "analysed_data/AR1to1d_halfstim.dat", "rb"))
AR1to1s_halfstim = pickle.load(open(folder + "analysed_data/AR1to1s_halfstim.dat", "rb"))
AR1to1d_fullstim_short = pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_short.dat", "rb"))
AR1to1s_fullstim_short = pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_short.dat", "rb"))

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
AR1to1d_halfstim = filter_data_main(AR1to1d_halfstim, threshold, "AR1to1d_halfstim")

AR1to1s_fullstim_long = filter_data_main(AR1to1s_fullstim_long, threshold, "AR1to1s_fullstim_long")
AR1to1s_halfstim = filter_data_main(AR1to1s_halfstim, threshold, "AR1to1s_halfstim")

n_d_fullstim = AR1to1d_fullstim_long["TFM_data"]["Tx"].shape[3]
n_s_fullstim = AR1to1s_fullstim_long["TFM_data"]["Tx"].shape[3]

# %% plot figure 1SA

# prepare data first

# concatenate TFM maps from different experiments and calculate average maps over first 20 frames and all cells to get average maps
Tx_1to1d_average = np.nanmean(AR1to1d_fullstim_long["TFM_data"]["Tx"][:, :, 0:20, :], axis=(2, 3))
Ty_1to1d_average = np.nanmean(AR1to1d_fullstim_long["TFM_data"]["Ty"][:, :, 0:20, :], axis=(2, 3))
Tx_1to1s_average = np.nanmean(AR1to1s_fullstim_long["TFM_data"]["Tx"][:, :, 0:20, :], axis=(2, 3))
Ty_1to1s_average = np.nanmean(AR1to1s_fullstim_long["TFM_data"]["Ty"][:, :, 0:20, :], axis=(2, 3))

# load simulation data
t_x_sim_d = doublet_FEM_simulation["feedback1.0"]["t_x"][:, :, 0] * 1e-3  # convert to kPa
t_y_sim_d = doublet_FEM_simulation["feedback1.0"]["t_y"][:, :, 0] * 1e-3
t_x_sim_s = singlet_FEM_simulation["feedback1.0"]["t_x"][:, :, 0] * 1e-3
t_y_sim_s = singlet_FEM_simulation["feedback1.0"]["t_y"][:, :, 0] * 1e-3

# calculate amplitudes
T_1to1d_average = np.sqrt(Tx_1to1d_average ** 2 + Ty_1to1d_average ** 2)
T_1to1s_average = np.sqrt(Tx_1to1s_average ** 2 + Ty_1to1s_average ** 2)

# crop maps
crop_start = 8
crop_end = 84

Tx_1to1d_average_crop = Tx_1to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e-3  # convert to kPa
Ty_1to1d_average_crop = Ty_1to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e-3
T_1to1d_average_crop = T_1to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e-3

Tx_1to1s_average_crop = Tx_1to1s_average[crop_start:crop_end, crop_start:crop_end] * 1e-3
Ty_1to1s_average_crop = Ty_1to1s_average[crop_start:crop_end, crop_start:crop_end] * 1e-3
T_1to1s_average_crop = T_1to1s_average[crop_start:crop_end, crop_start:crop_end] * 1e-3

# pad with 0 to match shape of experimental data
paddingdistance_x = int((Tx_1to1d_average_crop.shape[0] - t_x_sim_d.shape[0]) / 2)
paddingdistance_y = int((Ty_1to1d_average_crop.shape[0] - t_y_sim_d.shape[0]) / 2)
t_x_sim_d = np.pad(t_x_sim_d, (paddingdistance_x, paddingdistance_y), 'constant', constant_values=(0, 0))
t_y_sim_d = np.pad(t_y_sim_d, (paddingdistance_x, paddingdistance_y), 'constant', constant_values=(0, 0))
t_x_sim_s = np.pad(t_x_sim_s, (paddingdistance_x, paddingdistance_y), 'constant', constant_values=(0, 0))
t_y_sim_s = np.pad(t_y_sim_s, (paddingdistance_x, paddingdistance_y), 'constant', constant_values=(0, 0))

t_sim_d = np.sqrt(t_x_sim_d ** 2 + t_y_sim_d ** 2)
t_sim_s = np.sqrt(t_x_sim_s ** 2 + t_y_sim_s ** 2)

# set up plot parameters
# ******************************************************************************************************************************************
pixelsize = 0.864  # in µm
pmin = 0  # in kPa
pmax = 2  # in kPa
axtitle = 'kPa'  # unit of colorbar
suptitle = 'Traction forces'  # title of plot

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4, 3.5))  # create figure and axes
plt.subplots_adjust(wspace=0.02, hspace=-0.06)  # adjust space in between plots
# ******************************************************************************************************************************************

im = plot_forcemaps(axes[0, 0], Tx_1to1d_average_crop, Ty_1to1d_average_crop, pixelsize, pmax, pmin)
plot_forcemaps(axes[0, 1], t_x_sim_d, t_y_sim_d, pixelsize, pmax, pmin)
plot_forcemaps(axes[1, 0], Tx_1to1s_average_crop, Ty_1to1s_average_crop, pixelsize, pmax, pmin)
plot_forcemaps(axes[1, 1], t_x_sim_s, t_y_sim_s, pixelsize, pmax, pmin)

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

plt.figure(1).text(0.21, 0.89, "TFM data")
plt.figure(1).text(0.48, 0.89, "FEM simulation")

# add annotations
plt.text(0.24, 0.83, 'n=' + str(n_d_fullstim), transform=plt.figure(1).transFigure, color='w')
plt.text(0.24, 0.455, 'n=' + str(n_s_fullstim), transform=plt.figure(1).transFigure, color='w')

# draw pattern
for ax in axes.flat:
    draw_pattern_1to1(ax)

fig.savefig(figfolder + 'A.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'A.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 1SB, xx-stress maps

# prepare data first

# concatenate MSM maps from different experiments and calculate average maps over first 20 frames and all cells to get average maps
sigma_xx_1to1d_average = np.nanmean(AR1to1d_fullstim_long["MSM_data"]["sigma_xx"][:, :, 0:20, :], axis=(2, 3))
sigma_xx_1to1s_average = np.nanmean(AR1to1s_fullstim_long["MSM_data"]["sigma_xx"][:, :, 0:20, :], axis=(2, 3))

# load simulation data
sigma_xx_sim_d = doublet_FEM_simulation["feedback1.0"]["sigma_xx"][:, :, 2] * 1e3
sigma_xx_sim_s = singlet_FEM_simulation["feedback1.0"]["sigma_xx"][:, :, 2] * 1e3

# convert NaN to 0 to have black background
sigma_xx_1to1d_average[np.isnan(sigma_xx_1to1d_average)] = 0
sigma_xx_1to1s_average[np.isnan(sigma_xx_1to1s_average)] = 0

# crop maps
crop_start = 8
crop_end = 84

sigma_xx_1to1d_average_crop = sigma_xx_1to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e3
sigma_xx_1to1s_average_crop = sigma_xx_1to1s_average[crop_start:crop_end, crop_start:crop_end] * 1e3

# pad with 0 to match shape of experimental data
paddingdistance_x = int((sigma_xx_1to1d_average_crop.shape[0] - sigma_xx_sim_d.shape[0]) / 2)
paddingdistance_y = int((sigma_xx_1to1d_average_crop.shape[0] - sigma_xx_sim_d.shape[1]) / 2)
sigma_xx_sim_d = np.pad(sigma_xx_sim_d, (paddingdistance_x, paddingdistance_y), 'constant', constant_values=(0, 0))
sigma_xx_sim_s = np.pad(sigma_xx_sim_s, (paddingdistance_x, paddingdistance_y), 'constant', constant_values=(0, 0))

# set up plot parameters
# ******************************************************************************************************************************************
pixelsize = 0.864  # in µm
pmin = 0  # in mN/m
pmax = 8  # in mN/m
axtitle = 'mN/m'  # unit of colorbar
suptitle = '$\mathrm{\sigma _{xx}(x,y)}$'  # title of plot

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4, 3.5))  # create figure and axes
plt.subplots_adjust(wspace=0.02, hspace=-0.06)  # adjust space in between plots
# ******************************************************************************************************************************************

im = plot_stressmaps(axes[0, 0], sigma_xx_1to1d_average_crop, pixelsize, pmax, pmin)
plot_stressmaps(axes[0, 1], sigma_xx_sim_d, pixelsize, pmax, pmin)
plot_stressmaps(axes[1, 0], sigma_xx_1to1s_average_crop, pixelsize, pmax, pmin)
plot_stressmaps(axes[1, 1], sigma_xx_sim_s, pixelsize, pmax, pmin)

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
plt.text(0.24, 0.83, 'n=' + str(n_d_fullstim), transform=plt.figure(1).transFigure, color='w')
plt.text(0.24, 0.455, 'n=' + str(n_s_fullstim), transform=plt.figure(1).transFigure, color='w')

# draw pattern
for ax in axes.flat:
    draw_pattern_1to1(ax)

# save figure
fig.savefig(figfolder + 'B.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'B.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 1SC xx-stress x-profile

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.linspace(-40, 40, 92)
x = x[::2]  # downsample data for nicer plotting
xticks = np.arange(-40, 40.1, 20)  # define where the major ticks are gonna be
xlabel = 'position [µm]'
ymin = 0
ymax = 6
yticks = np.arange(ymin, ymax + 0.001, 1)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(1.5, 3.3))  # create figure and axes
plt.subplots_adjust(wspace=0.4, hspace=0.35)  # adjust space in between plots
# ******************************************************************************************************************************************

# Set up plot parameters for first panel
#######################################################################################################
ax = axes[0]
color = colors_parent[1]
ylabel = None
title = '$\mathrm{\sigma _{xx}(x)}$ [mN/m]'
y = AR1to1d_fullstim_long["MSM_data"]["sigma_xx_x_profile_baseline"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False, xmin=-40, xmax=40)

y_sim = doublet_FEM_simulation["feedback1.0"]["sigma_avg_norm_x_profile"][:, 2] * 1e3
x_sim = np.linspace(-22.5, 22.5, y_sim.shape[0])
ax.plot(x_sim, y_sim, color=color)

# # Set up plot parameters for second panel
# #######################################################################################################
ax = axes[1]
color = colors_parent[2]
ylabel = None
title = None
y = AR1to1s_fullstim_long["MSM_data"]["sigma_xx_x_profile_baseline"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False, xmin=-40, xmax=40)

y_sim = singlet_FEM_simulation["feedback1.0"]["sigma_xx_x_profile"][:, 2] * 1e3
x_sim = np.linspace(-22.5, 22.5, y_sim.shape[0])
ax.plot(x_sim, y_sim, color=color)

plt.savefig(figfolder + 'C.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'C.svg', dpi=300, bbox_inches="tight")
plt.show()
# %% plot figure 1SD, yy-stress maps

# prepare data first

# concatenate MSM maps from different experiments and calculate average maps over first 20 frames and all cells to get average maps
sigma_yy_1to1d_average = np.nanmean(AR1to1d_fullstim_long["MSM_data"]["sigma_yy"][:, :, 0:20, :], axis=(2, 3))
sigma_yy_1to1s_average = np.nanmean(AR1to1s_fullstim_long["MSM_data"]["sigma_yy"][:, :, 0:20, :], axis=(2, 3))

# load simulation data
sigma_yy_sim_d = doublet_FEM_simulation["feedback1.0"]["sigma_yy"][:, :, 2] * 1e3
sigma_yy_sim_s = singlet_FEM_simulation["feedback1.0"]["sigma_yy"][:, :, 2] * 1e3

# convert NaN to 0 to have black background
sigma_yy_1to1d_average[np.isnan(sigma_yy_1to1d_average)] = 0
sigma_yy_1to1s_average[np.isnan(sigma_yy_1to1s_average)] = 0

# crop maps
crop_start = 8
crop_end = 84

sigma_yy_1to1d_average_crop = sigma_yy_1to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e3
sigma_yy_1to1s_average_crop = sigma_yy_1to1s_average[crop_start:crop_end, crop_start:crop_end] * 1e3

# pad with 0 to match shape of experimental data
paddingdistance_x = int((sigma_yy_1to1d_average_crop.shape[0] - sigma_yy_sim_d.shape[0]) / 2)
paddingdistance_y = int((sigma_yy_1to1d_average_crop.shape[0] - sigma_yy_sim_d.shape[1]) / 2)
sigma_yy_sim_d = np.pad(sigma_yy_sim_d, (paddingdistance_x, paddingdistance_y), 'constant', constant_values=(0, 0))
sigma_yy_sim_s = np.pad(sigma_yy_sim_s, (paddingdistance_x, paddingdistance_y), 'constant', constant_values=(0, 0))

# set up plot parameters
# ******************************************************************************************************************************************
pixelsize = 0.864  # in µm
pmin = 0  # in mN/m
pmax = 8  # in mN/m
axtitle = 'mN/m'  # unit of colorbar
suptitle = '$\mathrm{\sigma _{yy}(x,y)}$'  # title of plot

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4, 3.5))  # create figure and axes
plt.subplots_adjust(wspace=0.02, hspace=-0.06)  # adjust space in between plots
# ******************************************************************************************************************************************

im = plot_stressmaps(axes[0, 0], sigma_yy_1to1d_average_crop, pixelsize, pmax, pmin)
plot_stressmaps(axes[0, 1], sigma_yy_sim_d, pixelsize, pmax, pmin)
plot_stressmaps(axes[1, 0], sigma_yy_1to1s_average_crop, pixelsize, pmax, pmin)
plot_stressmaps(axes[1, 1], sigma_yy_sim_s, pixelsize, pmax, pmin)

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
plt.text(0.24, 0.83, 'n=' + str(n_d_fullstim), transform=plt.figure(1).transFigure, color='w')
plt.text(0.24, 0.455, 'n=' + str(n_s_fullstim), transform=plt.figure(1).transFigure, color='w')

# draw pattern
for ax in axes.flat:
    draw_pattern_1to1(ax)

# save figure
fig.savefig(figfolder + 'D.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'D.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 1SE yy-stress x-profile

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.linspace(-40, 40, 92)
x = x[::2]  # downsample data for nicer plotting
xticks = np.arange(-40, 40.1, 20)  # define where the major ticks are gonna be
xlabel = 'position [µm]'
ymin = 0
ymax = 6
yticks = np.arange(ymin, ymax + 0.001, 1)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(1.5, 3.3))  # create figure and axes
plt.subplots_adjust(wspace=0.4, hspace=0.35)  # adjust space in between plots
# ******************************************************************************************************************************************

# Set up plot parameters for first panel
#######################################################################################################
ax = axes[0]
color = colors_parent[1]
ylabel = None
title = '$\mathrm{\sigma _{yy}(x)}$ [mN/m]'
y = AR1to1d_fullstim_long["MSM_data"]["sigma_yy_x_profile_baseline"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False, xmin=-40, xmax=40)

y_sim = doublet_FEM_simulation["feedback1.0"]["sigma_avg_norm_x_profile"][:, 2] * 1e3
x_sim = np.linspace(-22.5, 22.5, y_sim.shape[0])
ax.plot(x_sim, y_sim, color=color)

# # Set up plot parameters for second panel
# #######################################################################################################
ax = axes[1]
color = colors_parent[2]
ylabel = None
title = None
y = AR1to1s_fullstim_long["MSM_data"]["sigma_yy_x_profile_baseline"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False, xmin=-40, xmax=40)

y_sim = singlet_FEM_simulation["feedback1.0"]["sigma_yy_x_profile"][:, 2] * 1e3
x_sim = np.linspace(-22.5, 22.5, y_sim.shape[0])
ax.plot(x_sim, y_sim, color=color)

plt.savefig(figfolder + 'E.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'E.svg', dpi=300, bbox_inches="tight")
plt.show()