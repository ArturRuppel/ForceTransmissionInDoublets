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


# define some colors for the plots
colors_parent = ["#026473", "#E3CC69", "#77C8A6", "#D96248"]

mpl.rcParams["font.size"] = 8


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

folder = "C:/Users/aruppel/Documents/_forcetransmission_in_cell_doublets_raw/"
figfolder = folder + "_Figure4/"
if not os.path.exists(figfolder):
    os.mkdir(figfolder)

# %% load data for plotting

AR1to1d_fullstim_long = pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_long.dat", "rb"))
AR1to1d_fullstim_short = pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_short.dat", "rb"))
AR1to1d_halfstim = pickle.load(open(folder + "analysed_data/AR1to1d_halfstim.dat", "rb"))

doublet_CM_simulation = pickle.load(open(folder + "_contour_simulations/CM_1to1_simulation.dat", "rb"))
doublet_FEM_simulation = pickle.load(open(folder + "_FEM_simulations/FEM_doublets.dat", "rb"))
singlet_FEM_simulation = pickle.load(open(folder + "_FEM_simulations/FEM_singlets.dat", "rb"))

# %% filter data to remove cells that have an unstable baseline

def filter_data_main(data, threshold, title):
    # concatenate data on which it will be determined which cells will be filtered
    filterdata = np.stack(
        (data["TFM_data"]["relEs"][0:20, :], data["MSM_data"]["relsigma_xx"][0:20, :], data["MSM_data"]["relsigma_yy"][0:20, :],
         data["MSM_data"]["relsigma_xx_left"][0:20, :], data["MSM_data"]["relsigma_xx_right"][0:20, :],
         data["MSM_data"]["relsigma_yy_left"][0:20, :], data["MSM_data"]["relsigma_yy_right"][0:20, :]))

    # move axis of variable to the last position for consistency
    filterdata = np.moveaxis(filterdata, 0, -1)

    baselinefilter = create_baseline_filter(filterdata, threshold)

    # remove cells with unstable baselines
    data["TFM_data"] = apply_filter(data["TFM_data"], baselinefilter)
    data["MSM_data"] = apply_filter(data["MSM_data"], baselinefilter)
    data["shape_data"] = apply_filter(data["shape_data"], baselinefilter)

    new_N = np.sum(baselinefilter)
    print(title + ": " + str(baselinefilter.shape[0] - new_N) + " cells were filtered out")

    return data

# maximal allowed slope for linear fit of baseline
threshold = 0.0075

AR1to1d_fullstim_long = filter_data_main(AR1to1d_fullstim_long, threshold, "AR1to1d_fullstim_long")
AR1to1d_fullstim_short = filter_data_main(AR1to1d_fullstim_short, threshold, "AR1to1d_fullstim_short")
AR1to1d_halfstim = filter_data_main(AR1to1d_halfstim, threshold, "AR1to1d_halfstim")

# %% prepare dataframe for boxplots
# initialize empty dictionaries
concatenated_data_fs = {}
concatenated_data_hs = {}
concatenated_data_doublet = {}
concatenated_data_singlet = {}

# loop over all keys
for key1 in AR1to1d_fullstim_long:  # keys are the same for all dictionaries so I"m just taking one example here
    for key2 in AR1to1d_fullstim_long[key1]:
        if AR1to1d_fullstim_long[key1][key2].ndim == 1:  # only 1D data can be stored in the data frame
            concatenated_data_doublet[key2] = np.concatenate(
                (AR1to1d_fullstim_long[key1][key2], AR1to1d_halfstim[key1][key2]))

key1 = "TFM_data"
key2 = "Es_baseline"

# get number of elements for both condition
n_d_fullstim = AR1to1d_fullstim_long[key1][key2].shape[0]
n_d_halfstim = AR1to1d_halfstim[key1][key2].shape[0]

# create a list of keys with the same dimensions as the data
keys1to1d_fs = ["AR1to1d_fs" for i in range(n_d_fullstim)]
keys1to1d_hs = ["AR1to1d_hs" for i in range(n_d_halfstim)]

keys_doublet = np.concatenate((keys1to1d_fs, keys1to1d_hs))

# add keys to dictionary with concatenated data
concatenated_data_doublet["keys"] = keys_doublet

# Creates DataFrame
df_doublet = pd.DataFrame(concatenated_data_doublet)

# %% plot figure 4A1, xx-stress maps difference experiment vs simulation

# prepare data first
key = "feedback0.0"
sigma_xx_doublet_sim = (doublet_FEM_simulation[key]["sigma_xx"][:, :, 32] - doublet_FEM_simulation[key]["sigma_xx"][:, :, 20]) * 1e3    # convert to mN/m
key = "feedback0.2"
sigma_yy_doublet_sim = (doublet_FEM_simulation[key]["sigma_yy"][:, :, 32] - doublet_FEM_simulation[key]["sigma_yy"][:, :, 20]) * 1e3


sigma_yy_1to1d_diff = np.nanmean(
    AR1to1d_halfstim["MSM_data"]["sigma_yy"][:, :, 32, :] - AR1to1d_halfstim["MSM_data"]["sigma_yy"][:, :, 20, :], axis=2) * 1e3  # convert to mN/m

sigma_xx_1to1d_diff = np.nanmean(
    AR1to1d_halfstim["MSM_data"]["sigma_xx"][:, :, 32, :] - AR1to1d_halfstim["MSM_data"]["sigma_xx"][:, :, 20, :], axis=2) * 1e3  # convert to mN/m

# crop maps
crop_start = 8
crop_end = 84

sigma_xx_1to1d_diff_crop = sigma_xx_1to1d_diff[crop_start:crop_end, crop_start:crop_end]  # convert to mN/m
sigma_yy_1to1d_diff_crop = sigma_yy_1to1d_diff[crop_start:crop_end, crop_start:crop_end]  # convert to mN/m

# # pad simulation maps to make shapes equal. only works when shape is a square
paddingdistance = int((sigma_xx_1to1d_diff_crop.shape[0] - sigma_xx_doublet_sim.shape[0]) / 2)
sigma_xx_doublet_sim = np.pad(sigma_xx_doublet_sim, (paddingdistance, paddingdistance), "constant", constant_values=(0, 0))
sigma_yy_doublet_sim = np.pad(sigma_yy_doublet_sim, (paddingdistance, paddingdistance), "constant", constant_values=(0, 0))

# set up plot parameters
# ******************************************************************************************************************************************
pixelsize = 0.864  # in µm
pmin = -1
pmax = 1  # in mN/m
axtitle = "mN/m"  # unit of colorbar
cmap = "seismic"

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=((3, 3)))  # create figure and axes
plt.subplots_adjust(wspace=0.02, hspace=-0.06)  # adjust space in between plots
# ******************************************************************************************************************************************

axes[0, 0].set_title("$\mathrm{\Delta \sigma _{xx}(x,y)}$")
axes[0, 1].set_title("$\mathrm{\Delta \sigma _{yy}(x,y)}$")



im = plot_stressmaps(axes[0, 0], sigma_xx_1to1d_diff_crop, pixelsize, pmax, pmin, cmap=cmap)
plot_stressmaps(axes[0, 1], sigma_yy_1to1d_diff_crop, pixelsize, pmax, pmin, cmap=cmap)
plot_stressmaps(axes[1, 0], sigma_xx_doublet_sim, pixelsize, pmax, pmin, cmap=cmap)
plot_stressmaps(axes[1, 1], sigma_yy_doublet_sim, pixelsize, pmax, pmin, cmap=cmap)

# remove axes
for ax in axes.flat:
    ax.axis("off")
    aspectratio = 1.0
    ratio_default = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.set_aspect(ratio_default * aspectratio)


# add colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist())
cbar.ax.set_title(axtitle)


# draw pattern
for ax in axes.flat:
    draw_pattern_1to1(ax)

# save figure
fig.savefig(figfolder + "A.png", dpi=300, bbox_inches="tight")
fig.savefig(figfolder + "A.svg", dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 4B1, xx-stress profile increase MSM vs FEM
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(3, 2.7))  # create figure and axes
plt.subplots_adjust(wspace=0.5, hspace=0.35)  # adjust space in between plots
color = colors_parent[1]
x = np.linspace(-40, 40, 92)
x = x[::2]  # downsample data for nicer plotting
xticks = np.arange(-40, 40.1, 20)  # define where the major ticks are gonna be
xlabel = "position [µm]"
ymin = -0.2
ymax = 0.51
yticks = np.arange(ymin, ymax + 0.001, 0.1)
ylabel = None


# panel 1
title = "$\mathrm{\Delta \sigma _{xx}(x)}$ [mN/m]"
y = AR1to1d_halfstim["MSM_data"]["sigma_xx_x_profile_increase"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, axes[0, 0], color, optolinewidth=False)

# panel 2
title = "$\mathrm{\Delta \sigma _{yy}(x)}$ [mN/m]"
y = AR1to1d_halfstim["MSM_data"]["sigma_yy_x_profile_increase"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, axes[0, 1], color, optolinewidth=False)


# panel 3
ax = axes[1, 0]
c = 0
for key in doublet_FEM_simulation:
    c += 1
    y_sim = doublet_FEM_simulation[key]["sigma_xx_x_profile_increase"] * 1e3  # convert to nN
    x_sim = np.linspace(-22.5, 22.5, y_sim.shape[0])
    ax.plot(x_sim, y_sim, color=adjust_lightness(color, c / 10), alpha=1, linewidth=0.7)

# panel 4
ax = axes[1, 1]
c = 0
for key in doublet_FEM_simulation:
    c += 1
    y_sim = doublet_FEM_simulation[key]["sigma_yy_x_profile_increase"] * 1e3  # convert to nN
    x_sim = np.linspace(-22.5, 22.5, y_sim.shape[0])
    ax.plot(x_sim, y_sim, color=adjust_lightness(color, c / 10), alpha=1, linewidth=0.7)

# fig.suptitle("", y=1.02)

for ax in axes.flat:
    # set ticks
    ax.xaxis.set_ticks(xticks)
    ax.yaxis.set_ticks(yticks)

    # provide info on tick parameters
    ax.minorticks_on()
    ax.tick_params(direction="in", which="minor", length=3, bottom=True, top=False, left=True, right=True)
    ax.tick_params(direction="in", which="major", length=6, bottom=True, top=False, left=True, right=True)

    # set limits
    ax.set_ylim(ymin=ymin, ymax=ymax)
    ax.set_xlim(xmin=min(x))

    ax.set_xlabel(xlabel=xlabel, labelpad=1)

    # add line at y=0 for visualisation
    ax.plot([x[0], x[-1]], [0, 0], linewidth=0.5, linestyle=":", color="grey")

    # add line at x=-10 to show opto stimulation border
    ax.axvline(x=-10, ymin=0.0, ymax=1, linewidth=0.5, color="cyan")

fig.savefig(figfolder + "B.png", dpi=300, bbox_inches="tight")
fig.savefig(figfolder + "B.svg", dpi=300, bbox_inches="tight")
plt.show()

# %% plot activation profiles
def sigmoid(x, left_asymptote, right_asymptote, x0, l0):
    return (left_asymptote - right_asymptote) / (1 + np.exp((x - x0) / l0)) + right_asymptote

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2, 2))  # create figure and axes

feedbacks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
c=0
for fb in feedbacks:
    x = np.linspace(-22.5, 22.5, 50)
    c += 1
    left_asymptote = 1
    right_asymptote = fb
    x0 = -10
    l0 = 1
    y = sigmoid(x, left_asymptote, right_asymptote, x0, l0)
    ax.plot(x, y, color=adjust_lightness(color, c / 10), alpha=1, linewidth=0.7)

fig.savefig(figfolder + "Bx.png", dpi=300, bbox_inches="tight")
fig.savefig(figfolder + "Bx.svg", dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure C, stress-coupling
def find_x_position_of_point_on_array(x_list, y_list, y_point):
    f = interp1d(y_list, x_list, kind="linear")
    return f(y_point)


xticks = np.arange(-0.5, 1.001, 0.5)
color = colors_parent[1]

# make some calculations on the simulated data first
xx_stress_increase_ratio_sim = []
yy_stress_increase_ratio_sim = []

feedbacks = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for fb in feedbacks:
    xx_stress_increase_ratio_sim.append(singlet_FEM_simulation["feedback" + str(fb)]["xx_stress_increase_ratio"])
    yy_stress_increase_ratio_sim.append(singlet_FEM_simulation["feedback" + str(fb)]["yy_stress_increase_ratio"])

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(1.3, 2.8))
plt.subplots_adjust(hspace=0.4)  # adjust space in between plots

axes[0].plot(feedbacks, xx_stress_increase_ratio_sim, color="gray")
axes[1].plot(feedbacks, yy_stress_increase_ratio_sim, color="gray")

# add data points
sigma_xx_x_profile_increase_d = np.nanmean(AR1to1d_halfstim["MSM_data"]["sigma_xx_x_profile_increase"], axis=1)
sigma_xx_x_profile_increase_d_sem = np.nanstd(AR1to1d_halfstim["MSM_data"]["sigma_xx_x_profile_increase"], axis=1) / np.sqrt(
    np.shape(AR1to1d_halfstim["MSM_data"]["sigma_xx_x_profile_increase"])[1])

sigma_yy_x_profile_increase_d = np.nanmean(AR1to1d_halfstim["MSM_data"]["sigma_yy_x_profile_increase"], axis=1)
sigma_yy_x_profile_increase_d_sem = np.nanstd(AR1to1d_halfstim["MSM_data"]["sigma_yy_x_profile_increase"], axis=1) / np.sqrt(
    np.shape(AR1to1d_halfstim["MSM_data"]["sigma_yy_x_profile_increase"])[1])


center = int(sigma_xx_x_profile_increase_d.shape[0] / 2)

# calculate error with propagation of uncertainty
SI_xx_right_d = np.nansum(sigma_xx_x_profile_increase_d[center:-1])
SI_xx_right_d_err = np.sqrt(np.nansum(sigma_xx_x_profile_increase_d_sem[center:-1] ** 2))

SI_xx_left_d = np.nansum(sigma_xx_x_profile_increase_d[0:center])
SI_xx_left_d_err = np.sqrt(np.nansum(sigma_xx_x_profile_increase_d_sem[0:center] ** 2))

SI_yy_right_d = np.nansum(sigma_yy_x_profile_increase_d[center:-1])
SI_yy_right_d_err = np.sqrt(np.nansum(sigma_yy_x_profile_increase_d_sem[center:-1] ** 2))

SI_yy_left_d = np.nansum(sigma_yy_x_profile_increase_d[0:center])
SI_yy_left_d_err = np.sqrt(np.nansum(sigma_yy_x_profile_increase_d_sem[0:center] ** 2))

# calculate error with propagation of uncertainty
xx_stress_increase_ratio_d = SI_xx_right_d / (SI_xx_left_d + SI_xx_right_d)
xx_stress_increase_ratio_d_err = (SI_xx_right_d_err * SI_xx_left_d + SI_xx_left_d_err * SI_xx_right_d) / ((SI_xx_left_d + SI_xx_right_d) ** 2)

yy_stress_increase_ratio_d = SI_yy_right_d / (SI_yy_left_d + SI_yy_right_d)
yy_stress_increase_ratio_d_err = (SI_yy_right_d_err * SI_yy_left_d + SI_yy_left_d_err * SI_yy_right_d) / ((SI_yy_left_d + SI_yy_right_d) ** 2)


x = find_x_position_of_point_on_array(feedbacks, xx_stress_increase_ratio_sim, xx_stress_increase_ratio_d)
# xlo = find_x_position_of_point_on_array(feedbacks, xx_stress_increase_ratio_sim, xx_stress_increase_ratio_d - xx_stress_increase_ratio_d_err)
# xhi = find_x_position_of_point_on_array(feedbacks, xx_stress_increase_ratio_sim, xx_stress_increase_ratio_d + xx_stress_increase_ratio_d_err)
# x_err = np.zeros((2, 1))
# x_err[0] = xlo - x
# x_err[1] = xhi - x

axes[0].errorbar(x, xx_stress_increase_ratio_d, yerr=xx_stress_increase_ratio_d_err, mfc="w", color=color, marker="s", ms=5, linewidth=0.5,
            ls="none",
            markeredgewidth=0.5)

x = find_x_position_of_point_on_array(feedbacks, yy_stress_increase_ratio_sim, yy_stress_increase_ratio_d)
# xlo = find_x_position_of_point_on_array(feedbacks, yy_stress_increase_ratio_sim, yy_stress_increase_ratio_d - yy_stress_increase_ratio_d_err)
# xhi = find_x_position_of_point_on_array(feedbacks, yy_stress_increase_ratio_sim, yy_stress_increase_ratio_d + yy_stress_increase_ratio_d_err)
# x_err = np.zeros((2, 1))
# x_err[0] = xlo - x
# x_err[1] = xhi - x

axes[1].errorbar(x, yy_stress_increase_ratio_d, yerr=yy_stress_increase_ratio_d_err, mfc="w", color=color, marker="s", ms=5, linewidth=0.5,
            ls="none",
            markeredgewidth=0.5)

# set labels and titles
axes[1].set_xlabel(xlabel="Degree of active coupling")
axes[0].set_title(label="$\mathrm{\sigma _ {xx}}$")
axes[1].set_title(label="$\mathrm{\sigma _ {yy}}$")

# provide info on tick parameters
for ax in axes.flat:
    ax.minorticks_on()
    ax.tick_params(direction="in", which="minor", length=3, bottom=True, top=False, left=True, right=True)
    ax.tick_params(direction="in", which="major", length=6, bottom=True, top=False, left=True, right=True)
    ax.xaxis.set_ticks(xticks)
    ax.axvline(x=0, ymin=0, ymax=1, linewidth=0.5, color="grey", linestyle="--")

# plt.ylabel("Normalized xx-stress increase of \n non-activated area")
# plt.suptitle("Stress coupling", y=0.95)
plt.savefig(figfolder + "C.png", dpi=300, bbox_inches="tight")
plt.savefig(figfolder + "C.svg", dpi=300, bbox_inches="tight")
plt.show()


# %% plot figure D, contour strain measurement
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

# tracking data doesn"t live in the same coordinate system as the actin image, so I transform it
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
        color=colors_parent[0], marker="o", markerfacecolor="none", markersize=3, markeredgewidth=0.75, linestyle="none", alpha=0.3)
ax.plot(x_tracking_bottom_bs[::2], y_tracking_bottom_bs[::2],
        color=colors_parent[0], marker="o", markerfacecolor="none", markersize=3, markeredgewidth=0.75, linestyle="none", alpha=0.3)

ax.plot(x_tracking_top_as[::2], y_tracking_top_as[::2],
        color=colors_parent[0], marker="o", markerfacecolor="none", markersize=3, markeredgewidth=0.75, linestyle="none", alpha=1)
ax.plot(x_tracking_bottom_as[::2], y_tracking_bottom_as[::2],
        color=colors_parent[0], marker="o", markerfacecolor="none", markersize=3, markeredgewidth=0.75, linestyle="none", alpha=1)

ax.set_xlim([-0.1 * extent[1], 1.1 * extent[1]])
ax.set_ylim([-0.1 * extent[3], 1.1 * extent[3]])

ax.axis("off")

fig.savefig(figfolder + "D.png", dpi=300, bbox_inches="tight")
plt.savefig(figfolder + "D.svg", dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure E, contour strain after photoactivation

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.linspace(-17.5, 17.5, 50)
x = x[::2]  # downsample data for nicer plotting
ymin = -0.05
ymax = 0
xticks = np.arange(-15, 15.1, 15)  # define where the major ticks are gonna be
yticks = np.arange(ymin, ymax + 0.001, 0.01)
xlabel = "position [µm]"
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(3.2, 1.3))  # create figure and axes
plt.subplots_adjust(wspace=0.5, hspace=0.35)  # adjust space in between plots
# ******************************************************************************************************************************************

# Set up plot parameters for first panel
#######################################################################################################
ax = axes[0]
color = colors_parent[1]
ylabel = None
title = "Contour strain \n measurement"
y = AR1to1d_halfstim["shape_data"]["contour_strain"]
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False)

# Set up plot parameters for second panel
#######################################################################################################

ax = axes[1]
color = colors_parent[1]
ylabel = None
title = "Contour strain \n simulation"
# ymin = -0.03

feedbacks = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for fb in feedbacks:
    y_sim = doublet_CM_simulation["EA300"]["FB" + str(fb)]["epsilon_yy"]
    x_sim = np.linspace(-17.5, 17.5, y_sim.shape[0])
    ax.plot(x_sim, y_sim, color=adjust_lightness(color, fb), alpha=1, linewidth=0.7)
    ax.set_ylim(ymin=ymin)
    ax.set_ylim(ymax=ymax)
    ax.set_title(label=title, pad=3.5)
    ax.set_xlabel(xlabel=xlabel, labelpad=1)
    ax.minorticks_on()
    ax.xaxis.set_ticks(xticks)
    ax.yaxis.set_ticks(yticks)
    ax.tick_params(direction='in', which='minor', length=3, bottom=True, top=False, left=True, right=True)
    ax.tick_params(direction='in', which='major', length=6, bottom=True, top=False, left=True, right=True)


plt.savefig(figfolder + "E.png", dpi=300, bbox_inches="tight")
plt.savefig(figfolder + "E.svg", dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure F, active coupling of contour

# make some calculations on the simulated data first
strain_ratio_d_sim = []
strain_ratio_s_sim = []
xticks = np.arange(-0.5, 1.01, 0.5)

feedbacks = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for fb in feedbacks:
    epsilon = doublet_CM_simulation["EA300"]["FB" + str(fb)]["epsilon_yy"]
    center = int(epsilon.shape[0] / 2)
    epsilon_ratio = - np.nansum(epsilon[center:-1]) / (np.nansum(np.abs(epsilon)))
    strain_ratio_d_sim.append(epsilon_ratio)


def find_x_position_of_point_on_array(x_list, y_list, y_point):
    f = interp1d(y_list, x_list, kind="cubic")
    return f(y_point)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(1.3, 1.3))

ax.plot(feedbacks, strain_ratio_d_sim, color='gray')

# add data points
contour_strain_d = np.nanmean(AR1to1d_halfstim["shape_data"]["contour_strain"], axis=1)
contour_strain_d_sem = np.nanstd(AR1to1d_halfstim["shape_data"]["contour_strain"], axis=1) / np.sqrt(
    np.shape(AR1to1d_halfstim["shape_data"]["contour_strain"])[1])

center = int(contour_strain_d.shape[0] / 2)

strain_right_d = np.nansum(contour_strain_d[center:-1])
strain_right_d_err = np.sqrt(np.nansum(contour_strain_d_sem[center:-1] ** 2))


strain_left_d = np.nansum(contour_strain_d[0:center])
strain_left_d_err = np.sqrt(np.nansum(contour_strain_d_sem[0:center] ** 2))


# calculate error with propagation of uncertainty
strain_ratio_d = strain_right_d / (strain_left_d + strain_right_d)
strain_ratio_d_err = (strain_right_d_err * strain_left_d + strain_left_d_err * strain_right_d) / ((strain_left_d + strain_right_d) ** 2)


x = find_x_position_of_point_on_array(feedbacks, strain_ratio_d_sim, strain_ratio_d)
ax.errorbar(x, strain_ratio_d, yerr=strain_ratio_d_err, mfc="w", color=colors_parent[1], marker="s", ms=5, linewidth=0.5, ls="none",
            markeredgewidth=0.5)

ax.axvline(x=0, ymin=0, ymax=1, linewidth=0.5, color="grey", linestyle="--")

ax.xaxis.set_ticks(xticks)

# provide info on tick parameters
ax.minorticks_on()
ax.tick_params(direction="in", which="minor", length=3, bottom=True, top=False, left=True, right=True)
ax.tick_params(direction="in", which="major", length=6, bottom=True, top=False, left=True, right=True)

plt.xlabel("Degree of active coupling")
plt.ylabel("Normalized response of \n right cell")
# plt.title("Contour activation ratio")
plt.savefig(figfolder + "F.png", dpi=300, bbox_inches="tight")
plt.savefig(figfolder + "F.svg", dpi=300, bbox_inches="tight")
plt.show()
