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

figfolder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/_FigureS3/"
if not os.path.exists(figfolder):
    os.mkdir(figfolder)

# %% load data for plotting
folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"

AR1to1s_halfstim = pickle.load(open(folder + "analysed_data/AR1to1s_halfstim.dat", "rb"))

# normalize simulated strain energy curve
sim_Es_1to1s_hs_left = np.load(folder + "_FEM_simulations/strain_energy_singlets/strain_energy_fluidization.npz")["Es_l"]
sim_Es_1to1s_hs_right = np.load(folder + "_FEM_simulations/strain_energy_singlets/strain_energy_fluidization.npz")["Es_r"]

sim_relEs_1to1s_hs_left = sim_Es_1to1s_hs_left / np.nanmean(sim_Es_1to1s_hs_left[0:20]) - 1
sim_relEs_1to1s_hs_right = sim_Es_1to1s_hs_right / np.nanmean(sim_Es_1to1s_hs_right[0:20]) - 1

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

AR1to1s_halfstim = filter_data_main(AR1to1s_halfstim, threshold, "AR1to1s_halfstim")

n_s_halfstim = AR1to1s_halfstim["TFM_data"]["Tx"].shape[3]

# %% plot figure 3SD, Relative strain energy over time and relative energy increase

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.arange(60)
x = x[::2]  # downsample data for nicer plotting
xticks = np.arange(0, 61, 20)  # define where the major ticks are gonna be
xlabel = 'time [min]'
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(3.5, 1.5)) # create figure and axes

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

ax.plot(sim_relEs_1to1s_hs_left, color=colors[0])
ax.plot(sim_relEs_1to1s_hs_right, color=colors[1])

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

plt.savefig(figfolder + 'A.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'A.svg', dpi=300, bbox_inches="tight")
plt.show()

