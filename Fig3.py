# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 21:56:01 2021

@author: Artur Ruppel

"""
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from plot_and_filter_functions import *
import matplotlib.image as mpimg
from skimage.filters import threshold_otsu
from skimage import transform, io
from skimage.morphology import disk, erosion

mpl.rcParams['font.size'] = 8


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


# define some colors for the plots
colors_parent = ['#026473', '#E3CC69', '#77C8A6', '#D96248']
colors_parent_dark = ['#01353D', '#564910', '#235741', '#A93B23']

figfolder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/_Figure3/"
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

# load simulation data for plotting
sim_Es_1to1dfs = np.load(folder + "_FEM_simulations/strain_energy_doublets/strain_energy_halfstim_ar1to1d_1.0.npz")["energy"]
sim_Es_1to1sfs = np.load(folder + "_FEM_simulations/strain_energy_singlets/strain_energy_halfstim_ar1to1s_1.0.npz")["energy"]
sim_Es_1to1dhs_nocoupling = np.load(folder + "_FEM_simulations/strain_energy_doublets/strain_energy_halfstim_ar1to1d_0.0.npz")["energy"]
sim_Es_1to1shs_nocoupling = np.load(folder + "_FEM_simulations/strain_energy_singlets/strain_energy_halfstim_ar1to1s_0.0.npz")["energy"]
sim_Es_1to1dhs = np.load(folder + "_FEM_simulations/strain_energy_doublets/strain_energy_halfstim_ar1to1d_0.3.npz")["energy"]
sim_Es_1to1shs = np.load(folder + "_FEM_simulations/strain_energy_singlets/strain_energy_halfstim_ar1to1s_-0.5.npz")["energy"]

doublet_FEM_simulation = pickle.load(open(folder + "_FEM_simulations/FEM_doublets.dat", "rb"))
singlet_FEM_simulation = pickle.load(open(folder + "_FEM_simulations/FEM_singlets.dat", "rb"))

# normalize simulated strain energy curve
sim_relEs_1to1dfs = sim_Es_1to1dfs / np.nanmean(sim_Es_1to1dfs[0:20]) - 1
sim_relEs_1to1sfs = sim_Es_1to1sfs / np.nanmean(sim_Es_1to1sfs[0:20]) - 1
sim_relEs_1to1dhs_nocoupling = sim_Es_1to1dhs_nocoupling / np.nanmean(sim_Es_1to1dhs_nocoupling[0:20]) - 1
sim_relEs_1to1shs_nocoupling = sim_Es_1to1shs_nocoupling / np.nanmean(sim_Es_1to1shs_nocoupling[0:20]) - 1
sim_relEs_1to1dhs = sim_Es_1to1dhs / np.nanmean(sim_Es_1to1dhs[0:20]) - 1
sim_relEs_1to1shs = sim_Es_1to1shs / np.nanmean(sim_Es_1to1shs[0:20]) - 1

# %% load actin images
n_doublets = AR1to1d_halfstim["TFM_data"]["Dx"].shape[3]
n_singlets = AR1to1s_halfstim["TFM_data"]["Dx"].shape[3]

AR1to1d_halfstim["actin_images"] = {}
AR1to1s_halfstim["actin_images"] = {}

AR1to1d_halfstim["actin_images"]["beforestim"] = np.zeros((500, 500, n_doublets))
AR1to1d_halfstim["actin_images"]["afterstim"] = np.zeros((500, 500, n_doublets))
AR1to1s_halfstim["actin_images"]["beforestim"] = np.zeros((500, 500, n_singlets))
AR1to1s_halfstim["actin_images"]["afterstim"] = np.zeros((500, 500, n_singlets))

masks_d = AR1to1d_halfstim["shape_data"]["masks"][:, :, 33, :]
masks_s = AR1to1s_halfstim["shape_data"]["masks"][:, :, 33, :]



# resize to stressmapshape
# mask_resized = transform.resize(mask_cropped, (92, 92))

for cell in np.arange(n_doublets):

    actin_image_path = folder + "AR1to1_doublets_half_stim/actin_images/cell" + str(cell) + "frame20.png"
    actin_image = rgb2gray(mpimg.imread(actin_image_path))
    th = threshold_otsu(actin_image)
    actin_image[actin_image < th] = 0
    center = int(actin_image.shape[0] / 2)
    AR1to1d_halfstim["actin_images"]["beforestim"][:, :, cell] = actin_image[center - 250:center + 250, center - 250:center + 250]

    actin_image_path = folder + "AR1to1_doublets_half_stim/actin_images/cell" + str(cell) + "frame32.png"
    actin_image = rgb2gray(mpimg.imread(actin_image_path))
    th = threshold_otsu(actin_image)
    actin_image[actin_image < th] = 0
    center = int(actin_image.shape[0] / 2)
    AR1to1d_halfstim["actin_images"]["afterstim"][:, :, cell] = actin_image[center - 250:center + 250, center - 250:center + 250]

for cell in np.arange(n_singlets):
    actin_image_path = folder + "AR1to1_singlets_half_stim/actin_images/cell" + str(cell) + "frame20.png"
    actin_image = rgb2gray(mpimg.imread(actin_image_path))
    th = threshold_otsu(actin_image)
    actin_image[actin_image < th] = 0
    center = int(actin_image.shape[0] / 2)
    AR1to1s_halfstim["actin_images"]["beforestim"][:, :, cell] = actin_image[center - 250:center + 250, center - 250:center + 250]

    actin_image_path = folder + "AR1to1_singlets_half_stim/actin_images/cell" + str(cell) + "frame32.png"
    actin_image = rgb2gray(mpimg.imread(actin_image_path))
    th = threshold_otsu(actin_image)
    actin_image[actin_image < th] = 0
    center = int(actin_image.shape[0] / 2)
    AR1to1s_halfstim["actin_images"]["afterstim"][:, :, cell] = actin_image[center - 250:center + 250, center - 250:center + 250]


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
AR1to1d_halfstim = filter_data_main(AR1to1d_halfstim, threshold, "AR1to1d_halfstim", withactin=True)

AR1to1s_fullstim_long = filter_data_main(AR1to1s_fullstim_long, threshold, "AR1to1s_fullstim_long")
AR1to1s_halfstim = filter_data_main(AR1to1s_halfstim, threshold, "AR1to1s_halfstim", withactin=True)

# %% prepare dataframe for boxplots

# initialize empty dictionaries
concatenated_data_fs = {}
concatenated_data_hs = {}
concatenated_data_hs_lr = {}
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

concatenated_data_hs_lr["REI"] = np.concatenate((AR1to1d_halfstim["TFM_data"]["REI_left"], AR1to1d_halfstim["TFM_data"]["REI_right"],
                                                 AR1to1s_halfstim["TFM_data"]["REI_left"], AR1to1s_halfstim["TFM_data"]["REI_right"]))
concatenated_data_hs_lr["RAI"] = np.concatenate((AR1to1d_halfstim["shape_data"]["RAI_left"], AR1to1d_halfstim["shape_data"]["RAI_right"],
                                                 AR1to1s_halfstim["shape_data"]["RAI_left"], AR1to1s_halfstim["shape_data"]["RAI_right"]))

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

keys1to1d_hs_l = ['AR1to1d_hs_l' for i in range(n_d_halfstim)]
keys1to1d_hs_r = ['AR1to1d_hs_r' for i in range(n_d_halfstim)]
keys1to1s_hs_l = ['AR1to1s_hs_l' for i in range(n_s_halfstim)]
keys1to1s_hs_r = ['AR1to1s_hs_r' for i in range(n_s_halfstim)]

keys_fs = np.concatenate((keys1to1d_fs, keys1to1s_fs))
keys_hs = np.concatenate((keys1to1d_hs, keys1to1s_hs))
keys_doublet = np.concatenate((keys1to1d_fs, keys1to1d_hs))
keys_singlet = np.concatenate((keys1to1s_fs, keys1to1s_hs))
keys_hs_lr = np.concatenate((keys1to1d_hs_l, keys1to1d_hs_r, keys1to1s_hs_l, keys1to1s_hs_r))

# add keys to dictionary with concatenated data
concatenated_data_fs['keys'] = keys_fs
concatenated_data_hs['keys'] = keys_hs
concatenated_data_doublet['keys'] = keys_doublet
concatenated_data_singlet['keys'] = keys_singlet
concatenated_data_hs_lr['keys'] = keys_hs_lr

# Creates DataFrame
df_fs = pd.DataFrame(concatenated_data_fs)
df_hs = pd.DataFrame(concatenated_data_hs)
df_doublet = pd.DataFrame(concatenated_data_doublet)
df_singlet = pd.DataFrame(concatenated_data_singlet)
df_hs_lr = pd.DataFrame(concatenated_data_hs_lr)

df_d_fs = df_fs[df_fs["keys"] == "AR1to1d_fs"]
df_d_hs = df_hs[df_hs["keys"] == "AR1to1d_hs"]

df_s_fs = df_fs[df_fs["keys"] == "AR1to1s_fs"]
df_s_hs = df_hs[df_hs["keys"] == "AR1to1s_hs"]


# %% plot figure 3C, TFM differences

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
Tx_1to1d_hs_sim = doublet_FEM_simulation["feedback0.3"]["t_x"]
Ty_1to1d_hs_sim = doublet_FEM_simulation["feedback0.3"]["t_y"]

Tx_1to1s_hs_sim = singlet_FEM_simulation["feedback-0.5"]["t_x"]
Ty_1to1s_hs_sim = singlet_FEM_simulation["feedback-0.5"]["t_y"]

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
    draw_pattern(ax)

# save figure
fig.savefig(figfolder + 'C.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'C.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 3D, Relative strain energy over time and relative energy increase

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
fig = plt.figure(figsize=(3.5, 2.8))  # create figure and axes
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[:, 1])

plt.subplots_adjust(wspace=0.35, hspace=0.35)  # adjust space in between plots
# ******************************************************************************************************************************************

# Set up plot parameters for first panel
#######################################################################################################
ax = ax1
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

# ax.plot(sim_relEs_1to1dhs_nocoupling, color=color, linestyle="--")
# ax.plot(sim_relEs_1to1dhs, color=color)

# Set up plot parameters for second panel
#######################################################################################################
ax = ax2
colors = [colors_parent[2], colors_parent_dark[2]]
ylabel = None
title = None
ymin = -0.1
ymax = 0.2
yticks = np.arange(ymin, ymax + 0.01, 0.1)
y1 = AR1to1s_halfstim["TFM_data"]["relEs_left"]
y2 = AR1to1s_halfstim["TFM_data"]["relEs_right"]
y1 = y1[::2, :]
y2 = y2[::2, :]

# make plots
plot_two_values_over_time(x, y1, y2, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors)

# add line at y=0 for visualisation
ax.plot([x[0], x[-1]], [0, 0], linewidth=0.5, linestyle=":", color="grey")

# ax.plot(sim_relEs_1to1shs_nocoupling, color=color, linestyle="--")
# ax.plot(sim_relEs_1to1shs, color=color)

# Set up plot parameters for third panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'REI'  # variable that goes on the y-axis
ax = ax3  # define on which axis the plot goes
colors = [colors_parent[1], colors_parent_dark[1], colors_parent[2], colors_parent_dark[2]]  # defines colors
ymin = -0.2  # minimum value on y-axis
ymax = 0.6  # maximum value on y-axis
yticks = np.arange(-0.2, 0.81, 0.2)  # define where to put major ticks on y-axis
stat_annotation_offset = 0.25  # vertical offset of statistical annotation
ylabel = None  # which label to put on y-axis
title = "Relative energy \n increase"  # title of plot

# make plots
make_box_and_swarmplots(x, y, df_hs_lr, ax, ymin, ymax, yticks, xticklabels, ylabel, title, colors)

ax.plot([-1, 4], [0, 0], linewidth=0.5, linestyle=":", color="grey")
ax.plot([1.5, 1.5], [ymin, ymax], linewidth=0.5, linestyle=":", color="grey")

plt.savefig(figfolder + 'D.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'D.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 3E, actin differences

doublet_example = 2
singlet_example = 2

# prepare data
actin_diff_d_example = AR1to1d_halfstim["actin_images"]["afterstim"][:, :, doublet_example] - AR1to1d_halfstim["actin_images"]["beforestim"][:, :, doublet_example]
actin_diff_s_example = AR1to1s_halfstim["actin_images"]["afterstim"][:, :, singlet_example] - AR1to1s_halfstim["actin_images"]["beforestim"][:, :, singlet_example]

actin_diff_d_average = np.nanmean(AR1to1d_halfstim["actin_images"]["afterstim"] - AR1to1d_halfstim["actin_images"]["beforestim"], axis=2)
actin_diff_s_average = np.nanmean(AR1to1s_halfstim["actin_images"]["afterstim"] - AR1to1s_halfstim["actin_images"]["beforestim"], axis=2)


# set up plot parameters
# *****************************************************************************

pixelsize = 0.108  # in µm


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4, 3))

pmax = 1.5 * np.max(actin_diff_s_example)
pmin = -1.5 * np.max(actin_diff_s_example)
im = plot_stressmaps(axes[0, 0], actin_diff_d_example, pixelsize, pmax, pmin, cmap="seismic")
plot_stressmaps(axes[1, 0], actin_diff_s_example, pixelsize, pmax, pmin, cmap="seismic")

pmax = 1.5 * np.max(actin_diff_s_average)
pmin = -1.5 * np.max(actin_diff_s_average)
plot_stressmaps(axes[0, 1], actin_diff_d_average, pixelsize, pmax, pmin, cmap="seismic")
plot_stressmaps(axes[1, 1], actin_diff_s_average, pixelsize, pmax, pmin, cmap="seismic")

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
cbar.ax.set_title('AU')

# add annotations
plt.suptitle('$\mathrm{\Delta}$ Actin', y=0.98, x=0.44)

# plt.text(0.21, 0.89, "Doublet")
# plt.text(0.48, 0.89, "Singlet")
#
plt.text(-65, 75, "n=" + str(1))
plt.text(-5, 75, "n=" + str(n_d_halfstim))

plt.text(-65, 20, "n=" + str(1))
plt.text(-5, 20, "n=" + str(n_s_halfstim))


# draw pattern
for ax in axes.flat:
    draw_pattern(ax)

# save figure
fig.savefig(figfolder + 'E.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'E.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 3F, actin intensity over time

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.arange(60)
x = x[::2]  # downsample data for nicer plotting
ymin = -0.05
ymax = 0.075
xticks = np.arange(0, 61, 20)
yticks = np.arange(-0.05, 0.0751, 0.05)
xlabel = 'time [min]'
xticklabels = ['left', 'right', 'left', 'right']  # which labels to put on x-axis
fig = plt.figure(figsize=(3.5, 2.8))  # create figure and axes

plt.subplots_adjust(wspace=0.35, hspace=0.35)

gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[:, 1])
# ******************************************************************************************************************************************

# Set up plot parameters for first panel
#######################################################################################################
ax = ax1
colors = [colors_parent[1], colors_parent_dark[1]]
ylabel = None
title = 'Relative actin intensity'
y1 = AR1to1d_halfstim["shape_data"]["relactin_intensity_left"]
y2 = AR1to1d_halfstim["shape_data"]["relactin_intensity_right"]
y1 = y1[::2, :]
y2 = y2[::2, :]

# make plots
plot_two_values_over_time(x, y1, y2, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors, titleoffset=5)

ax.plot([x[0], x[-1]], [0, 0], linewidth=0.5, linestyle=':', color='grey')

# Set up plot parameters for second panel
#######################################################################################################
ax = ax2
colors = [colors_parent[2], colors_parent_dark[2]]
ylabel = None
title = None
y1 = AR1to1s_halfstim["shape_data"]["relactin_intensity_left"]
y2 = AR1to1s_halfstim["shape_data"]["relactin_intensity_right"]
y1 = y1[::2, :]
y2 = y2[::2, :]

# make plots
plot_two_values_over_time(x, y1, y2, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors)

ax.plot([x[0], x[-1]], [0, 0], linewidth=0.5, linestyle=':', color='grey')

# Set up plot parameters for third panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'RAI'  # variable that goes on the y-axis
ax = ax3  # define on which axis the plot goes
colors = [colors_parent[1], colors_parent_dark[1], colors_parent[2], colors_parent_dark[2]]  # defines colors
ymin = -0.15  # minimum value on y-axis
ymax = 0.15  # maximum value on y-axis
yticks = np.arange(-0.15, 0.151, 0.1)  # define where to put major ticks on y-axis

ylabel = None  # which label to put on y-axis
title = None  # title of plot
box_pairs = [('RAI_left', 'RAI_right')]  # which groups to perform statistical test on


# make plots
make_box_and_swarmplots(x, y, df_hs_lr, ax, ymin, ymax, yticks, xticklabels, ylabel, title, colors)

# add line at y=0 for visualisation
ax.plot([-1, 4], [0, 0], linewidth=0.5, linestyle=':', color='grey')
ax.plot([1.5, 1.5], [ymin, ymax], linewidth=0.5, linestyle=":", color="grey")

# # write title for panels 1 to 4
# # plt.text(-5.7, 2.575947, 'Relative strain energy', fontsize=10)
# write title for panels 3 and 4
plt.text(0, 0.16, 'Relative actin \n     increase', fontsize=10)
# # save plot to file
plt.savefig(figfolder + 'F.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'F.svg', dpi=300, bbox_inches="tight")
plt.show()





# # %% plot figure 3E1, normal-stress maps difference
#
# # prepare data first
#
# sigma_avg_normal_1to1d_diff = np.nanmean(
#     AR1to1d_halfstim["MSM_data"]["sigma_avg_normal"][:, :, 32, :] - AR1to1d_halfstim["MSM_data"]["sigma_avg_normal"][:, :, 20, :],
#     axis=2)
# sigma_avg_normal_1to1s_diff = np.nanmean(
#     AR1to1s_halfstim["MSM_data"]["sigma_avg_normal"][:, :, 32, :] - AR1to1s_halfstim["MSM_data"]["sigma_avg_normal"][:, :, 20, :],
#     axis=2)
#
# # crop maps
# crop_start = 8
# crop_end = 84
#
# sigma_avg_normal_1to1d_diff_crop = sigma_avg_normal_1to1d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3  # convert to mN/m
# sigma_avg_normal_1to1s_diff_crop = sigma_avg_normal_1to1s_diff[crop_start:crop_end, crop_start:crop_end] * 1e3
#
# # set up plot parameters
# # ******************************************************************************************************************************************
# pixelsize = 0.864  # in µm
# pmin = -1
# pmax = 1  # in mN/m
# axtitle = 'mN/m'  # unit of colorbar
# suptitle = '$\mathrm{\Delta \sigma _{avg. normal}(x,y)}$'  # title of plot
# x_end = np.shape(sigma_avg_normal_1to1d_diff_crop)[1]  # create x- and y-axis for plotting maps
# y_end = np.shape(sigma_avg_normal_1to1d_diff_crop)[0]
# extent = [0, x_end * pixelsize, 0, y_end * pixelsize]
# xq, yq = np.meshgrid(np.linspace(0, extent[1], x_end), np.linspace(0, extent[3], y_end))  # create mesh for vectorplot
#
# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(2.5, 3.3))  # create figure and axes
# plt.subplots_adjust(wspace=0.02, hspace=-0.06)  # adjust space in between plots
# # ******************************************************************************************************************************************
#
#
# im = axes[0].imshow(sigma_avg_normal_1to1d_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent,
#                     vmin=pmin, vmax=pmax, aspect='auto')
#
# axes[1].imshow(sigma_avg_normal_1to1s_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent,
#                vmin=pmin, vmax=pmax, aspect='auto')
#
# # remove axes
# for ax in axes.flat:
#     ax.axis('off')
#     aspectratio = 1.0
#     ratio_default = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
#     ax.set_aspect(ratio_default * aspectratio)
#
# # add colorbar
# cbar = fig.colorbar(im, ax=axes.ravel().tolist())
# cbar.ax.set_title(axtitle)
#
# # add title
# plt.suptitle(suptitle, y=0.94, x=0.44)
#
# # add annotations
# plt.text(0.43, 0.455, 'n=' + str(n_s_halfstim), transform=plt.figure(1).transFigure, color='black')
# plt.text(0.43, 0.83, 'n=' + str(n_d_halfstim), transform=plt.figure(1).transFigure, color='black')
#
# # save figure
# fig.savefig(figfolder + 'E1.png', dpi=300, bbox_inches="tight")
# fig.savefig(figfolder + 'E1.svg', dpi=300, bbox_inches="tight")
# plt.show()
# # %% plot figure 5E2
#
# # set up global plot parameters
# # ******************************************************************************************************************************************
# x = np.linspace(-40, 40, 92)
# x = x[::2]  # downsample data for nicer plotting
# xticks = np.arange(-40, 40.1, 20)  # define where the major ticks are gonna be
# xlabel = 'position [µm]'
# ymin = -0.1
# ymax = 0.5
# yticks = np.arange(ymin, ymax + 0.001, 0.1)
# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(1.5, 3.3))  # create figure and axes
# plt.subplots_adjust(wspace=0.4, hspace=0.35)  # adjust space in between plots
# # ******************************************************************************************************************************************
#
# # # Set up plot parameters for first panel
# # #######################################################################################################
# ax = axes[0]
# color = colors_parent[1]
# ylabel = None
# title = '$\mathrm{\Delta \sigma _{avg. normal}(x)}$ [mN/m]'
# y = AR1to1d_halfstim["MSM_data"]["sigma_avg_normal_x_profile_increase"] * 1e3  # convert to nN
# y = y[::2, :]
#
# # make plots
# plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False, xmax=40)
#
# # add line at y=0 for visualisation
# ax.plot([x[0], x[-1]], [0, 0], linewidth=0.5, linestyle=':', color='grey')
#
# # # Set up plot parameters for second panel
# # #######################################################################################################
# ax = axes[1]
# color = colors_parent[2]
# ylabel = None
# title = None
# y = AR1to1s_halfstim["MSM_data"]["sigma_avg_normal_x_profile_increase"] * 1e3  # convert to nN
# y = y[::2, :]
#
# # make plots
# plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False, xmax=40)
#
# # add line at y=0 for visualisation
# ax.plot([x[0], x[-1]], [0, 0], linewidth=0.5, linestyle=':', color='grey')
#
# plt.savefig(figfolder + 'E2.png', dpi=300, bbox_inches="tight")
# plt.savefig(figfolder + 'E2.svg', dpi=300, bbox_inches="tight")
# plt.show()


# # %% plot figure 3E, Relative stress over time halfstim
#
# # set up global plot parameters
# # ******************************************************************************************************************************************
# x = np.arange(60)
# x = x[::2]  # downsample data for nicer plotting
# ymin = -0.1
# ymax = 0.2
# xticks = np.arange(0, 61, 20)  # define where the major ticks are gonna be
# yticks = np.arange(ymin, ymax + 0.01, 0.1)
# xlabel = 'time [min]'
# xticklabels = ['left \n         $\mathrm{\Delta \sigma _ {xx}}$', 'right', 'left \n         $\mathrm{\Delta \sigma _ {yy}}$',
#                'right']  # which labels to put on x-axis
# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(5, 3))  # create figure and axes
# plt.subplots_adjust(wspace=0.35, hspace=0.35)  # adjust space in between plots
# # ******************************************************************************************************************************************
#
#
# # Set up plot parameters for first panel
# #######################################################################################################
# ax = axes[0, 0]
# colors = [colors_parent[1], colors_parent_dark[1]]
# ylabel = 'doublet'
# title = 'xx-Stress'
# y1 = AR1to1d_halfstim["MSM_data"]["relsigma_xx_left"]
# y2 = AR1to1d_halfstim["MSM_data"]["relsigma_xx_right"]
# y1 = y1[::2, :]
# y2 = y2[::2, :]
#
# # make plots
# plot_two_values_over_time(x, y1, y2, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors)
#
# ax.plot(sim_relstress_xx_left_1to1dhs, color=colors[0])
# ax.plot(sim_relstress_xx_right_1to1dhs, color=colors[0])
#
# # Set up plot parameters for second panel
# #######################################################################################################
# ax = axes[0, 1]
# colors = [colors_parent[1], colors_parent_dark[1]]
# ylabel = None
# title = 'yy-Stress'
# y1 = AR1to1d_halfstim["MSM_data"]["relsigma_yy_left"]
# y2 = AR1to1d_halfstim["MSM_data"]["relsigma_yy_right"]
# y1 = y1[::2, :]
# y2 = y2[::2, :]
#
# # make plots
# plot_two_values_over_time(x, y1, y2, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors)
#
# ax.plot(sim_relstress_yy_left_1to1dhs, color=colors[0])
# ax.plot(sim_relstress_yy_right_1to1dhs, color=colors[0])
#
# # Set up plot parameters for third panel
# #######################################################################################################
# ax = axes[1, 0]
# colors = [colors_parent[2], colors_parent_dark[2]]
# ylabel = 'singlet'
# title = None
# y1 = AR1to1s_halfstim["MSM_data"]["relsigma_xx_left"]
# y2 = AR1to1s_halfstim["MSM_data"]["relsigma_xx_right"]
# y1 = y1[::2, :]
# y2 = y2[::2, :]
#
# # make plots
# plot_two_values_over_time(x, y1, y2, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors)
#
# # Set up plot parameters for fourth panel
# #######################################################################################################
# ax = axes[1, 1]
# colors = [colors_parent[2], colors_parent_dark[2]]
# ylabel = None
# title = None
# y1 = AR1to1s_halfstim["MSM_data"]["relsigma_yy_left"]
# y2 = AR1to1s_halfstim["MSM_data"]["relsigma_yy_right"]
# y1 = y1[::2, :]
# y2 = y2[::2, :]
#
# # make plots
# plot_two_values_over_time(x, y1, y2, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors)
#
# # Set up plot parameters for fifth panel
# #######################################################################################################
# x = 'keys'  # variable by which to group the data
# y = 'sigma'  # variable that goes on the y-axis
# ax = axes[0, 2]  # define on which axis the plot goes
# colors = [colors_parent[1], colors_parent_dark[1], colors_parent[1], colors_parent_dark[1]]  # defines colors
# ymin = -0.2  # minimum value on y-axis
# ymax = 0.4  # maximum value on y-axis
# yticks = np.arange(-0.2, 0.41, 0.2)  # define where to put major ticks on y-axis
# ylabel = None  # which label to put on y-axis
# title = None  # title of plot
#
# # a little weird way to get the dataframe format I need for seaborn boxplots...
# df1 = pd.DataFrame(df_d_hs['RSI_xx_left'])
# df2 = pd.DataFrame(df_d_hs['RSI_xx_right'])
# df3 = pd.DataFrame(df_d_hs['RSI_yy_left'])
# df4 = pd.DataFrame(df_d_hs['RSI_yy_right'])
#
# df1 = df1.transpose().reset_index(drop=True).transpose()
# df2 = df2.transpose().reset_index(drop=True).transpose()
# df3 = df3.transpose().reset_index(drop=True).transpose()
# df4 = df4.transpose().reset_index(drop=True).transpose()
#
# df_plot = pd.concat([df1, df2, df3, df4], axis=0)
#
# df_plot.rename(columns={0: 'sigma'}, inplace=True)
#
# keys_sx_l = ['sigma_xx_left' for i in range(n_d_halfstim)]
# keys_sx_r = ['sigma_xx_right' for i in range(n_d_halfstim)]
# keys_sy_l = ['sigma_yy_left' for i in range(n_d_halfstim)]
# keys_sy_r = ['sigma_yy_right' for i in range(n_d_halfstim)]
# keys = np.concatenate((keys_sx_l, keys_sx_r, keys_sy_l, keys_sy_r))
# df_plot['keys'] = keys
#
# # make plots
# make_box_and_swarmplots(x, y, df_plot, ax, ymin, ymax, yticks, xticklabels, ylabel, title, colors)
#
# ax.plot(0, sim_relstress_xx_left_1to1dhs[30], zorder=3, marker='^', markersize=3, markerfacecolor="white", markeredgecolor="black",
#         markeredgewidth=0.3, alpha=1)
# ax.plot(1, sim_relstress_xx_right_1to1dhs[30], zorder=3, marker='^', markersize=3, markerfacecolor="white", markeredgecolor="black",
#         markeredgewidth=0.3, alpha=1)
# ax.plot(2, sim_relstress_yy_left_1to1dhs[30], zorder=3, marker='^', markersize=3, markerfacecolor="white", markeredgecolor="black",
#         markeredgewidth=0.3, alpha=1)
# ax.plot(3, sim_relstress_yy_right_1to1dhs[30], zorder=3, marker='^', markersize=3, markerfacecolor="white", markeredgecolor="black",
#         markeredgewidth=0.3, alpha=1)
#
# # Set up plot parameters for sixth panel
# #######################################################################################################
# x = 'keys'  # variable by which to group the data
# y = 'sigma'  # variable that goes on the y-axis
# ax = axes[1, 2]  # define on which axis the plot goes
# colors = [colors_parent[2], colors_parent_dark[2], colors_parent[2], colors_parent_dark[2]]  # defines colors
# ymin = -0.2  # minimum value on y-axis
# ymax = 0.4  # maximum value on y-axis
# yticks = np.arange(-0.2, 0.41, 0.2)  # define where to put major ticks on y-axis
# ylabel = None  # which label to put on y-axis
# title = None  # title of plot
#
# # a little weird way to get the dataframe format I need for seaborn boxplots...
# df1 = pd.DataFrame(df_s_hs['RSI_xx_left'])
# df2 = pd.DataFrame(df_s_hs['RSI_xx_right'])
# df3 = pd.DataFrame(df_s_hs['RSI_yy_left'])
# df4 = pd.DataFrame(df_s_hs['RSI_yy_right'])
#
# df1 = df1.transpose().reset_index(drop=True).transpose()
# df2 = df2.transpose().reset_index(drop=True).transpose()
# df3 = df3.transpose().reset_index(drop=True).transpose()
# df4 = df4.transpose().reset_index(drop=True).transpose()
#
# df_plot = pd.concat([df1, df2, df3, df4], axis=0)
#
# df_plot.rename(columns={0: 'sigma'}, inplace=True)
#
# keys_sx_l = ['sigma_xx_left' for i in range(n_s_halfstim)]
# keys_sx_r = ['sigma_xx_right' for i in range(n_s_halfstim)]
# keys_sy_l = ['sigma_yy_left' for i in range(n_s_halfstim)]
# keys_sy_r = ['sigma_yy_right' for i in range(n_s_halfstim)]
# keys = np.concatenate((keys_sx_l, keys_sx_r, keys_sy_l, keys_sy_r))
# df_plot['keys'] = keys
#
# # make plots
# make_box_and_swarmplots(x, y, df_plot, ax, ymin, ymax, yticks, xticklabels, ylabel, title, colors)
#
# # write title for panels 1 to 4
# plt.text(-11, 1.35, 'Relative stresses, local activation', fontsize=10)
# # write title for panels 5 to 6
# plt.text(-0.65, 1.25, 'Relative stress \n     increase', fontsize=10)
# #
# # # save plot to file
# plt.savefig(figfolder + 'E.png', dpi=300, bbox_inches="tight")
# plt.savefig(figfolder + 'E.svg', dpi=300, bbox_inches="tight")
# plt.show()

# # %% plot figure S3B, stress map difference fullstim
#
# # prepare data first
#
# # concatenate TFM maps from different experiments and calculate average maps over first 20 frames and all cells to get average maps
# sigmaxx_1to1d_diff = np.nanmean(
#     AR1to1d_fullstim_long["MSM_data"]["sigma_xx"][:, :, 32, :] - AR1to1d_fullstim_long["MSM_data"]["sigma_xx"][:, :, 20,
#                                                                  :], axis=2)
# sigmayy_1to1d_diff = np.nanmean(
#     AR1to1d_fullstim_long["MSM_data"]["sigma_yy"][:, :, 32, :] - AR1to1d_fullstim_long["MSM_data"]["sigma_yy"][:, :, 20,
#                                                                  :], axis=2)
# sigmaxx_1to1s_diff = np.nanmean(
#     AR1to1s_fullstim_long["MSM_data"]["sigma_xx"][:, :, 32, :] - AR1to1s_fullstim_long["MSM_data"]["sigma_xx"][:, :, 20,
#                                                                  :], axis=2)
# sigmayy_1to1s_diff = np.nanmean(
#     AR1to1s_fullstim_long["MSM_data"]["sigma_yy"][:, :, 32, :] - AR1to1s_fullstim_long["MSM_data"]["sigma_yy"][:, :, 20,
#                                                                  :], axis=2)
#
# # crop maps
# crop_start = 8
# crop_end = 84
#
# sigmaxx_1to1d_diff_crop = sigmaxx_1to1d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3  # convert to mN/m
# sigmayy_1to1d_diff_crop = sigmayy_1to1d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3
# sigmaxx_1to1s_diff_crop = sigmaxx_1to1s_diff[crop_start:crop_end, crop_start:crop_end] * 1e3
# sigmayy_1to1s_diff_crop = sigmayy_1to1s_diff[crop_start:crop_end, crop_start:crop_end] * 1e3
#
# # set up plot parameters
# # *****************************************************************************
#
# pixelsize = 0.864  # in µm
# sigma_max = 1  # kPa
# sigma_min = -1  # kPa
#
# # create x- and y-axis for plotting maps
# x_end = np.shape(sigmaxx_1to1d_diff_crop)[1]
# y_end = np.shape(sigmaxx_1to1d_diff_crop)[0]
# extent = [0, x_end * pixelsize, 0, y_end * pixelsize]
#
# # create mesh for vectorplot
# xq, yq = np.meshgrid(np.linspace(0, extent[1], x_end), np.linspace(0, extent[3], y_end))
#
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(3.8, 3.3))
#
# colormap = "seismic"
#
# im = axes[0, 0].imshow(sigmaxx_1to1d_diff_crop, cmap=plt.get_cmap(colormap), interpolation="bilinear", extent=extent,
#                        vmin=sigma_min, vmax=sigma_max, aspect='auto')
#
# axes[0, 1].imshow(sigmayy_1to1d_diff_crop, cmap=plt.get_cmap(colormap), interpolation="bilinear", extent=extent,
#                   vmin=sigma_min, vmax=sigma_max, aspect='auto')
#
# axes[1, 0].imshow(sigmaxx_1to1s_diff_crop, cmap=plt.get_cmap(colormap), interpolation="bilinear", extent=extent,
#                   vmin=sigma_min, vmax=sigma_max, aspect='auto')
#
# axes[1, 1].imshow(sigmayy_1to1s_diff_crop, cmap=plt.get_cmap(colormap), interpolation="bilinear", extent=extent,
#                   vmin=sigma_min, vmax=sigma_max, aspect='auto')
#
# # adjust space in between plots
# plt.subplots_adjust(wspace=0.02, hspace=-0.02)
#
# # remove axes
# for ax in axes.flat:
#     ax.axis('off')
#     aspectratio = 1.0
#     ratio_default = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
#     ax.set_aspect(ratio_default * aspectratio)
#
# # add colorbar
# cbar = fig.colorbar(im, ax=axes.ravel().tolist())
# cbar.ax.set_title('mN/m')
#
# # add title
# plt.suptitle('$\mathrm{\Delta}$ Stresses, global activation', y=0.97, x=0.44)
#
# plt.figure(1).text(0.21, 0.88, "xx-Stress")
# plt.figure(1).text(0.52, 0.88, "yy-Stress")
#
# plt.figure(1).text(0.24, 0.84, "n=" + str(n_d_fullstim))
# plt.figure(1).text(0.24, 0.46, "n=" + str(n_s_fullstim))
# plt.figure(1).text(0.55, 0.84, "n=" + str(n_d_fullstim))
# plt.figure(1).text(0.55, 0.46, "n=" + str(n_s_fullstim))
#
# # save figure
# fig.savefig(figfolder + 'SA.png', dpi=300, bbox_inches="tight")
# fig.savefig(figfolder + 'SA.svg', dpi=300, bbox_inches="tight")
# plt.show()
#
# # %% plot figure S3C, Relative stress over time fullstim
#
# # set up global plot parameters
# # ******************************************************************************************************************************************
# x = np.arange(60)
# x = x[::2]  # downsample data for nicer plotting
# ymin = -0.1
# ymax = 0.2
# xticks = np.arange(0, 61, 20)  # define where the major ticks are gonna be
# yticks = np.arange(ymin, ymax + 0.01, 0.1)
# xlabel = 'time [min]'
# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(5, 3))  # create figure and axes
# plt.subplots_adjust(wspace=0.35, hspace=0.35)  # adjust space in between plots
# # ******************************************************************************************************************************************
#
# # Set up plot parameters for first panel
# #######################################################################################################
# ax = axes[0, 0]
# color = colors_parent[1]
# ylabel = 'doublet'
# title = 'xx-Stress'
# y = AR1to1d_fullstim_long["MSM_data"]["relsigma_xx"]
# y = y[::2]
#
# # make plots
# plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color)
#
# ax.plot(x, sim_relstress_xx_1to1dfs[::2], color=color)
#
# # Set up plot parameters for second panel
# #######################################################################################################
# ax = axes[0, 1]
# color = colors_parent[1]
# ylabel = None
# title = 'yy-Stress'
# y = AR1to1d_fullstim_long["MSM_data"]["relsigma_yy"]
# y = y[::2]
#
# # make plots
# plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color)
#
# ax.plot(x, sim_relstress_yy_1to1dfs[::2], color=color)
#
# # Set up plot parameters for third panel
# #######################################################################################################
# ax = axes[1, 0]
# color = colors_parent[2]
# ylabel = 'singlet'
# title = None
# y = AR1to1s_fullstim_long["MSM_data"]["relsigma_xx"]
# y = y[::2]
#
# # make plots
# plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color)
#
# ax.plot(x, sim_relstress_xx_1to1sfs[::2], color=color)
#
# # Set up plot parameters for fourth panel
# #######################################################################################################
# ax = axes[1, 1]
# color = colors_parent[2]
# ylabel = None
# title = None
# y = AR1to1s_fullstim_long["MSM_data"]["relsigma_yy"]
# y = y[::2]
#
# # make plots
# plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color)
#
# ax.plot(x, sim_relstress_yy_1to1sfs[::2], color=color)
#
# # Set up plot parameters for fifth panel
# #######################################################################################################
# x = 'keys'  # variable by which to group the data
# y = 'sigma'  # variable that goes on the y-axis
# ax = axes[0, 2]  # define on which axis the plot goes
# colors = [colors_parent[1], colors_parent[1]]  # defines colors
# ymin = -0.2  # minimum value on y-axis
# ymax = 0.6  # maximum value on y-axis
# yticks = np.arange(-0.2, 0.61, 0.2)  # define where to put major ticks on y-axis
# stat_annotation_offset = 0  # vertical offset of statistical annotation
# ylabel = None  # which label to put on y-axis
# title = None  # title of plot
# box_pairs = [('sigma_xx', 'sigma_yy')]  # which groups to perform statistical test on
#
# # a little weird way to get the dataframe format I need for seaborn boxplots...
# df1 = pd.DataFrame(df_d_fs['RSI_xx'])
# df2 = pd.DataFrame(df_d_fs['RSI_yy'])
# df1 = df1.transpose().reset_index(drop=True).transpose()
# df2 = df2.transpose().reset_index(drop=True).transpose()
#
# df_plot = pd.concat([df1, df2], axis=0)
#
# df_plot.rename(columns={0: 'sigma'}, inplace=True)
#
# keys_sx = ['sigma_xx' for i in range(n_d_fullstim)]
# keys_sy = ['sigma_yy' for i in range(n_d_fullstim)]
# keys = np.concatenate((keys_sx, keys_sy))
# df_plot['keys'] = keys
#
# # make plots
# make_box_and_swarmplots_with_test(x, y, df_plot, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title,
#                                   colors)
#
# # Set up plot parameters for sixth panel
# #######################################################################################################
# x = 'keys'  # variable by which to group the data
# y = 'sigma'  # variable that goes on the y-axis
# ax = axes[1, 2]  # define on which axis the plot goes
# colors = [colors_parent[2], colors_parent_dark[2], colors_parent[2], colors_parent_dark[2]]  # defines colors
# stat_annotation_offset = 0  # vertical offset of statistical annotation
# ylabel = None  # which label to put on y-axis
# title = None  # title of plot
# box_pairs = [('sigma_xx', 'sigma_yy')]  # which groups to perform statistical test on
#
# # a little weird way to get the dataframe format I need for seaborn boxplots...
# df1 = pd.DataFrame(df_s_fs['RSI_xx'])
# df2 = pd.DataFrame(df_s_fs['RSI_yy'])
# df1 = df1.transpose().reset_index(drop=True).transpose()
# df2 = df2.transpose().reset_index(drop=True).transpose()
#
# df_plot = pd.concat([df1, df2], axis=0)
#
# df_plot.rename(columns={0: 'sigma'}, inplace=True)
#
# keys_sx = ['sigma_xx' for i in range(n_s_fullstim)]
# keys_sy = ['sigma_yy' for i in range(n_s_fullstim)]
# keys = np.concatenate((keys_sx, keys_sy))
# df_plot['keys'] = keys
#
# # make plots
# make_box_and_swarmplots_with_test(x, y, df_plot, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title,
#                                   colors)
#
# # write title for panels 1 to 4
# plt.text(-5.85, 1.82, 'Relative stresses, global activation', fontsize=10)
# # write title for panels 5 to 6
# plt.text(-0.5, 1.7, 'Relative stress \n     increase', fontsize=10)
# # # save plot to file
# plt.savefig(figfolder + 'SB.png', dpi=300, bbox_inches="tight")
# plt.savefig(figfolder + 'SB.svg', dpi=300, bbox_inches="tight")
# plt.show()
#
# # %% plot figure 3D, stress map difference halfstim
#
# # prepare data first
#
# sigmaxx_1to1d_diff = np.nanmean(
#     AR1to1d_halfstim["MSM_data"]["sigma_xx"][:, :, 32, :] - AR1to1d_halfstim["MSM_data"]["sigma_xx"][:, :, 20, :],
#     axis=2)
# sigmayy_1to1d_diff = np.nanmean(
#     AR1to1d_halfstim["MSM_data"]["sigma_yy"][:, :, 32, :] - AR1to1d_halfstim["MSM_data"]["sigma_yy"][:, :, 20, :],
#     axis=2)
# sigmaxx_1to1s_diff = np.nanmean(
#     AR1to1s_halfstim["MSM_data"]["sigma_xx"][:, :, 32, :] - AR1to1s_halfstim["MSM_data"]["sigma_xx"][:, :, 20, :],
#     axis=2)
# sigmayy_1to1s_diff = np.nanmean(
#     AR1to1s_halfstim["MSM_data"]["sigma_yy"][:, :, 32, :] - AR1to1s_halfstim["MSM_data"]["sigma_yy"][:, :, 20, :],
#     axis=2)
#
# # crop maps
# crop_start = 8
# crop_end = 84
#
# sigmaxx_1to1d_diff_crop = sigmaxx_1to1d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3  # convert to mN/m
# sigmayy_1to1d_diff_crop = sigmayy_1to1d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3
# sigmaxx_1to1s_diff_crop = sigmaxx_1to1s_diff[crop_start:crop_end, crop_start:crop_end] * 1e3
# sigmayy_1to1s_diff_crop = sigmayy_1to1s_diff[crop_start:crop_end, crop_start:crop_end] * 1e3
#
# # set up plot parameters
# # *****************************************************************************
#
# pixelsize = 0.864  # in µm
# sigma_max = 1  # kPa
# sigma_min = -1  # kPa
#
# # create x- and y-axis for plotting maps
# x_end = np.shape(sigmaxx_1to1d_diff_crop)[1]
# y_end = np.shape(sigmaxx_1to1d_diff_crop)[0]
# extent = [0, x_end * pixelsize, 0, y_end * pixelsize]
#
# # create mesh for vectorplot
# xq, yq = np.meshgrid(np.linspace(0, extent[1], x_end), np.linspace(0, extent[3], y_end))
#
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(3.8, 3.3))
#
# colormap = "seismic"
#
# im = axes[0, 0].imshow(sigmaxx_1to1d_diff_crop, cmap=plt.get_cmap(colormap), interpolation="bilinear", extent=extent,
#                        vmin=sigma_min, vmax=sigma_max, aspect='auto')
#
# axes[0, 1].imshow(sigmayy_1to1d_diff_crop, cmap=plt.get_cmap(colormap), interpolation="bilinear", extent=extent,
#                   vmin=sigma_min, vmax=sigma_max, aspect='auto')
#
# axes[1, 0].imshow(sigmaxx_1to1s_diff_crop, cmap=plt.get_cmap(colormap), interpolation="bilinear", extent=extent,
#                   vmin=sigma_min, vmax=sigma_max, aspect='auto')
#
# axes[1, 1].imshow(sigmayy_1to1s_diff_crop, cmap=plt.get_cmap(colormap), interpolation="bilinear", extent=extent,
#                   vmin=sigma_min, vmax=sigma_max, aspect='auto')
#
# # adjust space in between plots
# plt.subplots_adjust(wspace=0.02, hspace=-0.02)
#
# # remove axes
# for ax in axes.flat:
#     ax.axis('off')
#     aspectratio = 1.0
#     ratio_default = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
#     ax.set_aspect(ratio_default * aspectratio)
#
# # add colorbar
# cbar = fig.colorbar(im, ax=axes.ravel().tolist())
# cbar.ax.set_title('mN/m')
#
# # add title
# plt.suptitle('$\mathrm{\Delta}$ Stresses, local activation', y=0.97, x=0.44)
#
# plt.figure(1).text(0.21, 0.88, "xx-Stress")
# plt.figure(1).text(0.52, 0.88, "yy-Stress")
#
# plt.figure(1).text(0.24, 0.84, "n=" + str(n_d_halfstim))
# plt.figure(1).text(0.24, 0.46, "n=" + str(n_s_halfstim))
# plt.figure(1).text(0.55, 0.84, "n=" + str(n_d_halfstim))
# plt.figure(1).text(0.55, 0.46, "n=" + str(n_s_halfstim))
#
# # save figure
# fig.savefig(figfolder + 'D.png', dpi=300, bbox_inches="tight")
# fig.savefig(figfolder + 'D.svg', dpi=300, bbox_inches="tight")
# plt.show()
