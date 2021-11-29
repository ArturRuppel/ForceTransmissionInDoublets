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


# %% plot figure 3SA

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
n = 2  # every nth arrow will be plotted
pixelsize = 0.864  # in µm
pmin = 0  # in kPa
pmax = 2  # in kPa
axtitle = 'kPa'  # unit of colorbar
suptitle = 'Traction forces'  # title of plot
x_end = np.shape(T_1to1d_average_crop)[1]  # create x- and y-axis for plotting maps
y_end = np.shape(T_1to1d_average_crop)[0]
extent = [-int(x_end * pixelsize / 2), int(x_end * pixelsize / 2), -int(y_end * pixelsize / 2), int(y_end * pixelsize / 2)]

xq, yq = np.meshgrid(np.linspace(extent[0], extent[1], x_end), np.linspace(extent[2], extent[3], y_end))
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

# %% plot figure 3SB, normal-stress maps

# prepare data first

# concatenate MSM maps from different experiments and calculate average maps over first 20 frames and all cells to get average maps
sigma_avg_normal_1to1d_average = np.nanmean(AR1to1d_fullstim_long["MSM_data"]["sigma_avg_normal"][:, :, 0:20, :], axis=(2, 3))
sigma_avg_normal_1to1s_average = np.nanmean(AR1to1s_fullstim_long["MSM_data"]["sigma_avg_normal"][:, :, 0:20, :], axis=(2, 3))

# load simulation data
sigma_avg_normal_sim_d = doublet_FEM_simulation["feedback1.0"]["sigma_avg_norm"][:, :, 2] * 1e3
sigma_avg_normal_sim_s = singlet_FEM_simulation["feedback1.0"]["sigma_avg_norm"][:, :, 2] * 1e3

# convert NaN to 0 to have black background
sigma_avg_normal_1to1d_average[np.isnan(sigma_avg_normal_1to1d_average)] = 0
sigma_avg_normal_1to1s_average[np.isnan(sigma_avg_normal_1to1s_average)] = 0

# crop maps
crop_start = 8
crop_end = 84

sigma_avg_normal_1to1d_average_crop = sigma_avg_normal_1to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e3
sigma_avg_normal_1to1s_average_crop = sigma_avg_normal_1to1s_average[crop_start:crop_end, crop_start:crop_end] * 1e3

# pad with 0 to match shape of experimental data
paddingdistance_x = int((sigma_avg_normal_1to1d_average_crop.shape[0] - sigma_avg_normal_sim_d.shape[0]) / 2)
paddingdistance_y = int((sigma_avg_normal_1to1d_average_crop.shape[0] - sigma_avg_normal_sim_d.shape[1]) / 2)
sigma_avg_normal_sim_d = np.pad(sigma_avg_normal_sim_d, (paddingdistance_x, paddingdistance_y), 'constant', constant_values=(0, 0))
sigma_avg_normal_sim_s = np.pad(sigma_avg_normal_sim_s, (paddingdistance_x, paddingdistance_y), 'constant', constant_values=(0, 0))

# set up plot parameters
# ******************************************************************************************************************************************
pixelsize = 0.864  # in µm
pmin = 0  # in mN/m
pmax = 5  # in mN/m
axtitle = 'mN/m'  # unit of colorbar
suptitle = '$\mathrm{\sigma _{avg. normal}(x,y)}$'  # title of plot
x_end = np.shape(sigma_avg_normal_1to1d_average_crop)[1]  # create x- and y-axis for plotting maps
y_end = np.shape(sigma_avg_normal_1to1d_average_crop)[0]
extent = [-int(x_end * pixelsize / 2), int(x_end * pixelsize / 2), -int(y_end * pixelsize / 2), int(y_end * pixelsize / 2)]
xq, yq = np.meshgrid(np.linspace(0, extent[1], x_end), np.linspace(0, extent[3], y_end))  # create mesh for vectorplot

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4, 3.5))  # create figure and axes
plt.subplots_adjust(wspace=0.02, hspace=-0.06)  # adjust space in between plots
# ******************************************************************************************************************************************

plot_stressmaps(axes[0, 0], sigma_avg_normal_1to1d_average_crop, pixelsize, pmax, pmin)
plot_stressmaps(axes[0, 1], sigma_avg_normal_sim_d, pixelsize, pmax, pmin)
plot_stressmaps(axes[1, 0], sigma_avg_normal_1to1s_average_crop, pixelsize, pmax, pmin)
plot_stressmaps(axes[1, 1], sigma_avg_normal_sim_s, pixelsize, pmax, pmin)

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
plt.text(0.24, 0.455, 'n=' + str(n_d_fullstim), transform=plt.figure(1).transFigure, color='w')

# draw pattern
for ax in axes.flat:
    draw_pattern_1to1(ax)

# save figure
fig.savefig(figfolder + 'B.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'B.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 3SC average normal stress x-profile

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.linspace(-40, 40, 92)
x = x[::2]  # downsample data for nicer plotting
xticks = np.arange(-40, 40.1, 20)  # define where the major ticks are gonna be
xlabel = 'position [µm]'
ymin = 0
ymax = 5
yticks = np.arange(ymin, ymax + 0.001, 1)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(1.5, 3.3))  # create figure and axes
plt.subplots_adjust(wspace=0.4, hspace=0.35)  # adjust space in between plots
# ******************************************************************************************************************************************

# Set up plot parameters for first panel
#######################################################################################################
ax = axes[0]
color = colors_parent[1]
ylabel = None
title = '$\mathrm{\sigma _{avg. normal}(x)}$ [mN/m]'
y = AR1to1d_fullstim_long["MSM_data"]["sigma_avg_normal_x_profile_baseline"] * 1e3  # convert to nN
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
y = AR1to1s_fullstim_long["MSM_data"]["sigma_avg_normal_x_profile_baseline"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False, xmin=-40, xmax=40)

y_sim = singlet_FEM_simulation["feedback1.0"]["sigma_avg_norm_x_profile"][:, 2] * 1e3
x_sim = np.linspace(-22.5, 22.5, y_sim.shape[0])
ax.plot(x_sim, y_sim, color=color)

plt.savefig(figfolder + 'C.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'C.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 3SD, TFM differences fullstim

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

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4, 3))

im = plot_forcemaps_diff(axes[0, 0], Tx_1to1d_fs_diff_crop, Ty_1to1d_fs_diff_crop, T_1to1d_fs_diff_crop, pixelsize, pmax, pmin, n=2,
                         scale=1)
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
fig.savefig(figfolder + 'D.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'D.svg', dpi=300, bbox_inches="tight")
plt.show()
# %% plot figure 3SE, Relative strain energy over time and relative energy increase

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
fig = plt.figure(figsize=(3.8, 2.5))  # create figure and axes
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[:, 1])

plt.subplots_adjust(wspace=0.35, hspace=0.35)  # adjust space in between plots
# ******************************************************************************************************************************************


# Set up plot parameters for first panel
#######################################################################################################
ax = ax1
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
ax = ax2
color = colors_parent[2]
ylabel = 'singlet'
title = None
y = AR1to1s_fullstim_long["TFM_data"]["Es"]
y = (y - np.nanmean(y[0:20], axis=0)) / np.nanmean(y[0:20], axis=(0, 1))  # normalize
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, xmax=60)

ax.plot(sim_relEs_1to1sfs, color=color)

# Set up plot parameters for third panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'REI'  # variable that goes on the y-axis
ax = ax3  # define on which axis the plot goes
colors = [colors_parent[1], colors_parent[2]]  # defines colors
ymin = -0.2  # minimum value on y-axis
ymax = 0.6  # maximum value on y-axis
yticks = np.arange(-0.2, 0.81, 0.2)  # define where to put major ticks on y-axis
stat_annotation_offset = -0.32  # vertical offset of statistical annotation
ylabel = None  # which label to put on y-axis
title = "Relative energy \n increase"  # title of plot
box_pairs = [('AR1to1d_fs', 'AR1to1s_fs')]  # which groups to perform statistical test on

# make plots
make_box_and_swarmplots_with_test(x, y, df_fs, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title,
                                  colors)



plt.suptitle("Global activation", y=1.06)

plt.savefig(figfolder + 'E.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'E.svg', dpi=300, bbox_inches="tight")
plt.show()
