# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 21:56:01 2021

@author: Artur Ruppel

"""
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
from plot_and_filter_functions import *
from scipy.interpolate import interp1d


mpl.rcParams['font.size'] = 8
stressmappixelsize = 1.296 * 1e-6  # in meter

# %% load data for plotting
folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"

tissues_lefthalf_stim = pickle.load(open(folder + "analysed_data/tissues_lefthalf_stim.dat", "rb"))
tissues_tophalf_stim = pickle.load(open(folder + "analysed_data/tissues_tophalf_stim.dat", "rb"))

tissues_lefthalf_FEM_simulation = pickle.load(open(folder + "_FEM_simulations/FEM_tissues_lefthalfstim.dat", "rb"))
tissues_tophalf_FEM_simulation = pickle.load(open(folder + "_FEM_simulations/FEM_tissues_tophalfstim.dat", "rb"))

AR1to2d_halfstim = pickle.load(open(folder + "analysed_data/AR1to2d_halfstim.dat", "rb"))
AR1to1d_halfstim = pickle.load(open(folder + "analysed_data/AR1to1d_halfstim.dat", "rb"))
AR1to1s_halfstim = pickle.load(open(folder + "analysed_data/AR1to1s_halfstim.dat", "rb"))
AR2to1d_halfstim = pickle.load(open(folder + "analysed_data/AR2to1d_halfstim.dat", "rb"))


# define some colors for the plots
colors_parent = ['#026473', '#E3CC69', '#77C8A6', '#D96248']
colors_parent_dark = ['#01353D', '#564910', '#235741', '#A93B23']

figfolder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/_Figure6/"
if not os.path.exists(figfolder):
    os.mkdir(figfolder)



# %% prepare dataframe for boxplots

# initialize empty dictionaries
concatenated_data_lefthalf_stim = {}
concatenated_data_tophalf_stim = {}
concatenated_data = {}

# loop over all keys
for key1 in tissues_lefthalf_stim:  # keys are the same for all dictionaries so I'm just taking one example here
    for key2 in tissues_lefthalf_stim[key1]:
        if tissues_lefthalf_stim[key1][key2].ndim == 1:  # only 1D data can be stored in the data frame
            # concatenate values from different experiments
            concatenated_data_lefthalf_stim[key2] = tissues_lefthalf_stim[key1][key2]
            concatenated_data_tophalf_stim[key2] = tissues_tophalf_stim[key1][key2]

            concatenated_data[key2] = np.concatenate(
                (concatenated_data_lefthalf_stim[key2], concatenated_data_tophalf_stim[key2]))

key2 = 'Es_baseline'
# get number of elements for both condition
n_ls = concatenated_data_lefthalf_stim[key2].shape[0]
n_ts = concatenated_data_tophalf_stim[key2].shape[0]

# create a list of keys with the same dimensions as the data
keysls = ['ls' for i in range(n_ls)]
keysts = ['ts' for i in range(n_ts)]

keys = np.concatenate((keysls, keysts))

keys2 = ['tissue' for i in range(n_ls + n_ts)]

# add keys to dictionary with concatenated data
concatenated_data['keys'] = keys
concatenated_data['keys2'] = keys2

# Creates DataFrame
df = pd.DataFrame(concatenated_data)

# convert to more convenient units for plotting
df_plot_units = df  # all units here are in SI units
df_plot_units['Es_baseline'] *= 1e12  # convert to fJ
df_plot_units['sigma_xx_baseline'] *= 1e3  # convert to mN/m
df_plot_units['sigma_yy_baseline'] *= 1e3  # convert to mN/m
# %% prepare dataframe with doublets and singlets for boxplots

# initialize empty dictionaries
concatenated_data_1to2d = {}
concatenated_data_1to1d = {}
concatenated_data_1to1s = {}
concatenated_data_2to1d = {}
concatenated_data = {}

# loop over all keys
for key1 in AR1to1d_halfstim:  # keys are the same for all dictionaries so I'm just taking one example here
    for key2 in AR1to1d_halfstim[key1]:
        if AR1to1d_halfstim[key1][key2].ndim == 1:  # only 1D data can be stored in the data frame
            # concatenate values from different experiments
            concatenated_data_1to2d[key2] = AR1to2d_halfstim[key1][key2]
            concatenated_data_1to1d[key2] = AR1to1d_halfstim[key1][key2]
            concatenated_data_1to1s[key2] = AR1to1s_halfstim[key1][key2]
            concatenated_data_2to1d[key2] = AR2to1d_halfstim[key1][key2]

            concatenated_data[key2] = np.concatenate(
                (concatenated_data_1to2d[key2], concatenated_data_1to1d[key2], concatenated_data_1to1s[key2], concatenated_data_2to1d[key2]))
key2 = 'Es_baseline'
# get number of elements for both condition
n_1to2d = concatenated_data_1to2d[key2].shape[0]
n_1to1d = concatenated_data_1to1d[key2].shape[0]
n_1to1s = concatenated_data_1to1s[key2].shape[0]
n_2to1d = concatenated_data_2to1d[key2].shape[0]

# create a list of keys with the same dimensions as the data
keys1to2d = ['AR1to2d' for i in range(n_1to2d)]
keys1to1d = ['AR1to1d' for i in range(n_1to1d)]
keys1to1s = ['AR1to1s' for i in range(n_1to1s)]
keys2to1d = ['AR2to1d' for i in range(n_2to1d)]
keys = np.concatenate((keys1to2d, keys1to1d, keys1to1s, keys2to1d))

# add keys to dictionary with concatenated data
concatenated_data['keys2'] = keys

# Creates DataFrame
df_2 = pd.DataFrame(concatenated_data)

# convert to more convenient units for plotting
df['Es_baseline'] *= 1e12  # convert to fJ
df['sigma_xx_baseline'] *= 1e3  # convert to mN/m
df['sigma_yy_baseline'] *= 1e3  # convert to mN/m

df_all = pd.concat([df_2, df])
# %% plot figure 5A, force maps

# prepare data first

# Calculate average maps over first 20 frames and all cells to get average maps
Tx_all = np.concatenate((tissues_tophalf_stim["TFM_data"]["Tx"][:, :, 0:20, :], tissues_lefthalf_stim["TFM_data"]["Tx"][:, :, 0:20, :]), axis=3) * 1e-3     # convert to kPa
Ty_all = np.concatenate((tissues_tophalf_stim["TFM_data"]["Ty"][:, :, 0:20, :], tissues_lefthalf_stim["TFM_data"]["Ty"][:, :, 0:20, :]), axis=3) * 1e-3

Tx_average = np.nanmean(Tx_all, axis=(2, 3))
Ty_average = np.nanmean(Ty_all, axis=(2, 3))

Tx_example = Tx_all[:, :, 0, 3]
Ty_example = Ty_all[:, :, 0, 3]

# set up plot parameters
# *****************************************************************************
n = 4  # every nth arrow will be plotted
pixelsize = 1.296  # in µm
pmax = 2  # kPa
pmin = 0


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(2.5, 3))

im = plot_forcemaps(axes[0], Tx_example, Ty_example, pixelsize, pmax, pmin)
plot_forcemaps(axes[1], Tx_average, Ty_average, pixelsize, pmax, pmin)


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
plt.text(0.3, 0.8, 'n=' + str(1), transform=plt.figure(1).transFigure, color='w')
plt.text(0.3, 0.42, 'n=' + str(n_ts + n_ls), transform=plt.figure(1).transFigure, color='w')

fig.savefig(figfolder + 'A.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'A.svg', dpi=300, bbox_inches="tight")

plt.show()
# %% plot figure 5B, stress maps

# prepare data first
sigma_xx_all = np.concatenate((tissues_tophalf_stim["MSM_data"]["sigma_xx"][:, :, 0:20, :], tissues_lefthalf_stim["MSM_data"]["sigma_xx"][:, :, 0:20, :]), axis=3) * 1e3
sigma_yy_all = np.concatenate((tissues_tophalf_stim["MSM_data"]["sigma_yy"][:, :, 0:20, :], tissues_lefthalf_stim["MSM_data"]["sigma_yy"][:, :, 0:20, :]), axis=3) * 1e3

# Calculate average maps over first 20 frames and all cells to get average maps
sigma_xx_average = np.nanmean(sigma_xx_all, axis=(2, 3))
sigma_yy_average = np.nanmean(sigma_yy_all, axis=(2, 3))

sigma_xx_example = sigma_xx_all[:, :, 0, 3]
sigma_yy_example = sigma_yy_all[:, :, 0, 3]

# convert NaN to 0 to have black background
sigma_xx_average[np.isnan(sigma_xx_average)] = 0
sigma_yy_average[np.isnan(sigma_yy_average)] = 0
sigma_xx_example[np.isnan(sigma_xx_example)] = 0
sigma_yy_example[np.isnan(sigma_yy_example)] = 0


# set up plot parameters
# *****************************************************************************
pixelsize = 1.296  # in µm
pmin = 0
pmax = 10  # mN/m

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4, 3))

im = plot_stressmaps(axes[0, 0], sigma_xx_example, pixelsize, pmax, pmin)
plot_stressmaps(axes[0, 1], sigma_yy_example, pixelsize, pmax, pmin)
plot_stressmaps(axes[1, 0], sigma_xx_average, pixelsize, pmax, pmin)
plot_stressmaps(axes[1, 1], sigma_yy_average, pixelsize, pmax, pmin)

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
plt.suptitle('Cell stresses', y=0.97, x=0.42, size=10)
plt.text(-180, 240, '$\mathrm{\sigma _ {xx}}$', size=10)
plt.text(-20, 240, '$\mathrm{\sigma _ {yy}}$', size=10)

# add annotations
plt.text(0.18, 0.8, 'n=' + str(1), transform=plt.figure(1).transFigure, color='w')
plt.text(0.18, 0.42, 'n=' + str(n_ts + n_ls), transform=plt.figure(1).transFigure, color='w')

plt.text(0.48, 0.8, 'n=' + str(1), transform=plt.figure(1).transFigure, color='w')
plt.text(0.48, 0.42, 'n=' + str(n_ts + n_ls), transform=plt.figure(1).transFigure, color='w')

fig.savefig(figfolder + 'B.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'B.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 5C correlation plot of stress anisotropy and actin anisotropy

# set up global plot parameters
# ******************************************************************************************************************************************
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 2.5))  # create figure and axes
plt.subplots_adjust(wspace=0.45, hspace=0.45)  # adjust space in between plots

ylabeloffset = -7
xlabeloffset = 0
colors = [colors_parent[0], colors_parent[1], colors_parent[2], colors_parent[3], colors_parent_dark[1]]  # defines colors for scatterplot

y = 'actin_anisotropy_coefficient'
x = 'AIC_baseline'
hue = 'keys2'
ymin = -1
ymax = 1
xmin = -1
xmax = 1
yticks = np.arange(-1, 1.1, 0.5)
xticks = np.arange(-1, 1.1, 0.5)
ylabel = "Structural polarization (actin)"  # "'$\mathrm{\sigma_{x, MSM}}$'
xlabel = "Mechanical polarization"  # '$\mathrm{\sigma_{x, CM}}$'

corr, p = make_correlationplotsplots(x, y, hue, df_all, ax, xmin, xmax, ymin, ymax, xticks, yticks, xlabel, ylabel, colors)

# add line with slope 1 for visualisation
# ax.plot([ymin, ymax], [0, 0], linewidth=0.5, linestyle=':', color='grey')
# ax.plot([45, 45], [xmin, xmax], linewidth=0.5, linestyle=':', color='grey')

plt.text(0.21 * xmax + xmin, 1.05 * ymax, 'R = ' + str(corr))
# plt.text(0.52 * xmax, 1.1 * ymax, 'p = ' + '{:0.2e}'.format(p))

plt.savefig(figfolder + 'C.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'C.svg', dpi=300, bbox_inches="tight")
plt.show()
# # %% plot figure 5C, Stress and actin anisotropy coefficient boxplot
#
# # define plot parameters that are valid for the whole figure
# # ******************************************************************************************************************************************
# colors = [colors_parent[0], colors_parent[1], colors_parent[3]]  # defines colors
# sns.set_palette(sns.color_palette(colors))  # sets colors
# xticklabels = ['']  # which labels to put on x-axis
# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(1.2, 2.6))  # create figure instance
# # adjust space in between plots
# plt.subplots_adjust(wspace=0, hspace=0.4)
# # ******************************************************************************************************************************************
#
#
# # Set up plot parameters for first panel
# #######################################################################################################
# x = 'keys2'  # variable by which to group the data
# y = 'AIC_baseline'  # variable that goes on the y-axis
# ymin = 0  # minimum value on y-axis
# ymax = 1  # maximum value on y-axis
# yticks = np.arange(0, 1.01, 0.5)  # define where to put major ticks on y-axis
# ylabel = None  # which label to put on y-axis
# title = 'Mechanical \n polarization'  # title of plot
#
# # make plots
# make_box_and_swarmplots(x, y, df, axes[0], ymin, ymax, yticks, xticklabels, ylabel, title, colors)
#
# title = 'Structural \n polarization (actin)'  # title of plot
# y = 'actin_anisotropy_coefficient'  # variable that goes on the y-axis
# make_box_and_swarmplots(x, y, df, axes[1], ymin, ymax, yticks, xticklabels, ylabel, title, colors)
#
# # # save plot to file
# plt.savefig(figfolder + 'C.png', dpi=300, bbox_inches="tight")
# plt.savefig(figfolder + 'C.svg', dpi=300, bbox_inches="tight")
# plt.show()
# # %% filter data to remove cells that have an unstable baseline
#
# def filter_data_main(data, threshold, title):
#     # concatenate data on which it will be determined which cells will be filtered
#     filterdata = np.stack(
#         (data["TFM_data"]["relEs"][0:20, :], data["MSM_data"]["relsigma_xx"][0:20, :], data["MSM_data"]["relsigma_yy"][0:20, :],
#          data["MSM_data"]["relsigma_xx_left"][0:20, :], data["MSM_data"]["relsigma_xx_right"][0:20, :],
#          data["MSM_data"]["relsigma_yy_left"][0:20, :], data["MSM_data"]["relsigma_yy_right"][0:20, :]))
#
#     # move axis of variable to the last position for consistency
#     filterdata = np.moveaxis(filterdata, 0, -1)
#
#     # maximal allowed slope for linear fit of baseline
#     baselinefilter = create_baseline_filter(filterdata, threshold)
#
#     # remove cells with unstable baselines
#     data["TFM_data"] = apply_filter(data["TFM_data"], baselinefilter)
#     data["MSM_data"] = apply_filter(data["MSM_data"], baselinefilter)
#
#     new_N = np.sum(baselinefilter)
#     print(title + ": " + str(baselinefilter.shape[0] - new_N) + " cells were filtered out")
#
#     return data
#
# # maximal allowed slope for linear fit of baseline
# threshold = 0.1
#
# # tissues_full_stim = filter_data_main(tissues_full_stim, "tissues_full_stim")
# tissues_lefthalf_stim = filter_data_main(tissues_lefthalf_stim, threshold, "tissues_lefthalf_stim")
# tissues_tophalf_stim = filter_data_main(tissues_tophalf_stim, threshold, "tissues_tophalf_stim")
#
# # %% prepare dataframe again after filtering
#
# # initialize empty dictionaries
# concatenated_data_lefthalf_stim = {}
# concatenated_data_tophalf_stim = {}
# concatenated_data = {}
#
# # loop over all keys
# for key1 in tissues_lefthalf_stim:  # keys are the same for all dictionaries so I'm just taking one example here
#     for key2 in tissues_lefthalf_stim[key1]:
#         if tissues_lefthalf_stim[key1][key2].ndim == 1:  # only 1D data can be stored in the data frame
#             # concatenate values from different experiments
#             concatenated_data_lefthalf_stim[key2] = tissues_lefthalf_stim[key1][key2]
#             concatenated_data_tophalf_stim[key2] = tissues_tophalf_stim[key1][key2]
#
#             concatenated_data[key2] = np.concatenate(
#                 (concatenated_data_lefthalf_stim[key2], concatenated_data_tophalf_stim[key2]))
#
# key2 = 'Es_baseline'
# # get number of elements for both condition
# n_ls = concatenated_data_lefthalf_stim[key2].shape[0]
# n_ts = concatenated_data_tophalf_stim[key2].shape[0]
#
# # create a list of keys with the same dimensions as the data
# keysls = ['ls' for i in range(n_ls)]
# keysts = ['ts' for i in range(n_ts)]
#
# keys = np.concatenate((keysls, keysts))
#
# # add keys to dictionary with concatenated data
# concatenated_data['keys'] = keys
#
# # Creates DataFrame
# df = pd.DataFrame(concatenated_data)
#
# # convert to more convenient units for plotting
# df_plot_units = df  # all units here are in SI units
# df_plot_units['Es_baseline'] *= 1e12  # convert to fJ
# df_plot_units['sigma_xx_baseline'] *= 1e3  # convert to mN/m
# df_plot_units['sigma_yy_baseline'] *= 1e3  # convert to mN/m

#%% calculate stress increase ratio of right vs all
sigma_xx_x_profile_increase_tophalf = np.nanmean(tissues_tophalf_stim["MSM_data"]["sigma_xx_x_profile_increase"], axis=1)
sigma_xx_x_profile_increase_tophalf_sem = np.nanstd(tissues_tophalf_stim["MSM_data"]["sigma_xx_x_profile_increase"], axis=1) / np.sqrt(
    np.shape(tissues_tophalf_stim["MSM_data"]["sigma_xx_x_profile_increase"])[1])

sigma_xx_x_profile_increase_lefthalf = np.nanmean(tissues_lefthalf_stim["MSM_data"]["sigma_xx_x_profile_increase"], axis=1)
sigma_xx_x_profile_increase_lefthalf_sem = np.nanstd(tissues_lefthalf_stim["MSM_data"]["sigma_xx_x_profile_increase"], axis=1) / np.sqrt(
    np.shape(tissues_lefthalf_stim["MSM_data"]["sigma_xx_x_profile_increase"])[1])

sigma_yy_x_profile_increase_tophalf = np.nanmean(tissues_tophalf_stim["MSM_data"]["sigma_yy_x_profile_increase"], axis=1)
sigma_yy_x_profile_increase_tophalf_sem = np.nanstd(tissues_tophalf_stim["MSM_data"]["sigma_yy_x_profile_increase"], axis=1) / np.sqrt(
    np.shape(tissues_tophalf_stim["MSM_data"]["sigma_yy_x_profile_increase"])[1])

sigma_yy_x_profile_increase_lefthalf = np.nanmean(tissues_lefthalf_stim["MSM_data"]["sigma_yy_x_profile_increase"], axis=1)
sigma_yy_x_profile_increase_lefthalf_sem = np.nanstd(tissues_lefthalf_stim["MSM_data"]["sigma_yy_x_profile_increase"], axis=1) / np.sqrt(
    np.shape(tissues_lefthalf_stim["MSM_data"]["sigma_yy_x_profile_increase"])[1])


center = int(sigma_xx_x_profile_increase_tophalf.shape[0] / 2)

# calculate error with propagation of uncertainty
SI_xx_right_tophalf = np.nansum(sigma_xx_x_profile_increase_tophalf[center:-1])
SI_xx_right_tophalf_err = np.sqrt(np.nansum(sigma_xx_x_profile_increase_tophalf_sem[center:-1] ** 2))
SI_xx_left_tophalf = np.nansum(sigma_xx_x_profile_increase_tophalf[0:center])
SI_xx_left_tophalf_err = np.sqrt(np.nansum(sigma_xx_x_profile_increase_tophalf_sem[0:center] ** 2))

SI_xx_right_lefthalf = np.nansum(sigma_xx_x_profile_increase_lefthalf[center:-1])
SI_xx_right_lefthalf_err = np.sqrt(np.nansum(sigma_xx_x_profile_increase_lefthalf_sem[center:-1] ** 2))
SI_xx_left_lefthalf = np.nansum(sigma_xx_x_profile_increase_lefthalf[0:center])
SI_xx_left_lefthalf_err = np.sqrt(np.nansum(sigma_xx_x_profile_increase_lefthalf_sem[0:center] ** 2))


SI_yy_right_tophalf = np.nansum(sigma_yy_x_profile_increase_tophalf[center:-1])
SI_yy_right_tophalf_err = np.sqrt(np.nansum(sigma_yy_x_profile_increase_tophalf_sem[center:-1] ** 2))
SI_yy_left_tophalf= np.nansum(sigma_yy_x_profile_increase_tophalf[0:center])
SI_yy_left_tophalf_err = np.sqrt(np.nansum(sigma_yy_x_profile_increase_tophalf_sem[0:center] ** 2))

SI_yy_right_lefthalf = np.nansum(sigma_yy_x_profile_increase_lefthalf[center:-1])
SI_yy_right_lefthalf_err = np.sqrt(np.nansum(sigma_yy_x_profile_increase_lefthalf_sem[center:-1] ** 2))
SI_yy_left_lefthalf = np.nansum(sigma_yy_x_profile_increase_lefthalf[0:center])
SI_yy_left_lefthalf_err = np.sqrt(np.nansum(sigma_yy_x_profile_increase_lefthalf_sem[0:center] ** 2))


# calculate error with propagation of uncertainty
xx_stress_increase_ratio_tophalf = SI_xx_right_tophalf / (SI_xx_left_tophalf + SI_xx_right_tophalf)
xx_stress_increase_ratio_tophalf_err = (SI_xx_right_tophalf_err * SI_xx_left_tophalf + SI_xx_left_tophalf_err * SI_xx_right_tophalf) / ((SI_xx_left_tophalf + SI_xx_right_tophalf) ** 2)

xx_stress_increase_ratio_lefthalf = SI_xx_right_lefthalf / (SI_xx_left_lefthalf + SI_xx_right_lefthalf)
xx_stress_increase_ratio_lefthalf_err = (SI_xx_right_lefthalf_err * SI_xx_left_lefthalf + SI_xx_left_lefthalf_err * SI_xx_right_lefthalf) / ((SI_xx_left_lefthalf + SI_xx_right_lefthalf) ** 2)


yy_stress_increase_ratio_tophalf = SI_yy_right_tophalf / (SI_yy_left_tophalf + SI_yy_right_tophalf)
yy_stress_increase_ratio_tophalf_err = (SI_yy_right_tophalf_err * SI_yy_left_tophalf + SI_yy_left_tophalf_err * SI_yy_right_tophalf) / ((SI_yy_left_tophalf + SI_yy_right_tophalf) ** 2)

yy_stress_increase_ratio_lefthalf = SI_yy_right_lefthalf / (SI_yy_left_lefthalf + SI_yy_right_lefthalf)
yy_stress_increase_ratio_lefthalf_err = (SI_yy_right_lefthalf_err * SI_yy_left_lefthalf + SI_yy_left_lefthalf_err * SI_yy_right_lefthalf) / ((SI_yy_left_lefthalf + SI_yy_right_lefthalf) ** 2)


# %% plot figure 5D, stress map differences

# prepare data first

# concatenate TFM maps from different experiments and calculate average maps over first 20 frames and all cells to get average maps
sigma_xx_tophalf_stim_diff = \
    np.nanmean(tissues_tophalf_stim["MSM_data"]["sigma_xx"][:, :, 32, :] -
               tissues_tophalf_stim["MSM_data"]["sigma_xx"][:, :, 20, :], axis=2) * 1e3  # convert to mN/m
sigma_yy_tophalf_stim_diff = \
    np.nanmean(tissues_tophalf_stim["MSM_data"]["sigma_yy"][:, :, 32, :] -
               tissues_tophalf_stim["MSM_data"]["sigma_yy"][:, :, 20, :], axis=2) * 1e3  # convert to mN/m


sigma_xx_lefthalf_stim_diff = \
    np.nanmean(tissues_lefthalf_stim["MSM_data"]["sigma_xx"][:, :, 32, :] -
               tissues_lefthalf_stim["MSM_data"]["sigma_xx"][:, :, 20, :], axis=2) * 1e3  # convert to mN/m
sigma_yy_lefthalf_stim_diff = \
    np.nanmean(tissues_lefthalf_stim["MSM_data"]["sigma_yy"][:, :, 32, :] -
               tissues_lefthalf_stim["MSM_data"]["sigma_yy"][:, :, 20, :], axis=2) * 1e3  # convert to mN/m


# set up plot parameters
# *****************************************************************************
pixelsize = 1.296  # in µm
sigma_max = 2  # kPa
sigma_min = -2  # kPa

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4, 3))

# adjust space in between plots
plt.subplots_adjust(wspace=0.2, hspace=0)

im = plot_stressmaps(axes[0, 0], sigma_xx_tophalf_stim_diff, pixelsize, sigma_max, sigma_min, cmap="seismic")
plot_stressmaps(axes[0, 1], sigma_yy_tophalf_stim_diff, pixelsize, sigma_max, sigma_min, cmap="seismic")
plot_stressmaps(axes[1, 0], sigma_xx_lefthalf_stim_diff, pixelsize, sigma_max, sigma_min, cmap="seismic")
plot_stressmaps(axes[1, 1], sigma_yy_lefthalf_stim_diff, pixelsize, sigma_max, sigma_min, cmap="seismic")

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
plt.suptitle('$\mathrm{\Delta}$ Cell stresses', y=0.97, x=0.42, size=10)
plt.text(-180, 230, '$\mathrm{\Delta \sigma _ {xx}}$', size=10)
plt.text(-20, 230, '$\mathrm{\Delta \sigma _ {yy}}$', size=10)


# add annotations
plt.text(0.40, 0.85, 'n=' + str(n_ts), transform=plt.figure(1).transFigure, color='black')
plt.text(0.40, 0.40, 'n=' + str(n_ls), transform=plt.figure(1).transFigure, color='black')

# save figure
fig.savefig(figfolder + 'D.png', dpi=300, bbox_inches="tight") ,
fig.savefig(figfolder + 'D.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 5E

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.linspace(-78, 78, 120)
x = x[::2]  # downsample data for nicer plotting
xticks = np.arange(-60, 60.1, 30)  # define where the major ticks are gonna be
xlabel = 'position [µm]'
ymin = -0.6
ymax = 1.6
yticks = np.arange(ymin, ymax + 0.001, 0.4)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(2.7, 2.7))  # create figure and axes
plt.subplots_adjust(wspace=0.4, hspace=0.35)  # adjust space in between plots

# ******************************************************************************************************************************************

# # Set up plot parameters for first and second panel
# #######################################################################################################
color = colors_parent[0]
ylabel = None
ax = axes[0, 0]
title = '$\mathrm{\Delta \sigma _{xx}(x)}$ [mN/m]'
y = tissues_tophalf_stim["MSM_data"]["sigma_xx_x_profile_increase"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False, titleoffset=10)

ax = axes[0, 1]
title = '$\mathrm{\Delta \sigma _{yy}(x)}$ [mN/m]'
y = tissues_tophalf_stim["MSM_data"]["sigma_yy_x_profile_increase"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False, titleoffset=10)

# # Set up plot parameters for third and fourth panel
# #######################################################################################################
color = colors_parent[3]
ylabel = None
title = None

ax = axes[1, 0]
y = tissues_lefthalf_stim["MSM_data"]["sigma_xx_x_profile_increase"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False, titleoffset=20)

ax = axes[1, 1]
y = tissues_lefthalf_stim["MSM_data"]["sigma_yy_x_profile_increase"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False, titleoffset=20)

for ax in axes.flat:
    # add line at y=0 for visualisation
    ax.plot([x[0], x[-1]], [0, 0], linewidth=0.5, linestyle=":", color="grey")

    # add line at x=-10 to show opto stimulation border
    ax.axvline(x=0, ymin=0.0, ymax=1, linewidth=0.5, color="cyan")

plt.savefig(figfolder + 'E.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'E.svg', dpi=300, bbox_inches="tight")
plt.show()




# %% plot figure 5F

def find_x_position_of_point_on_array(x_list, y_list, y_point):
    f = interp1d(y_list, x_list, kind="linear")
    return f(y_point)


xticks = np.arange(-0.5, 1.001, 0.5)
color = colors_parent[1]

# make some calculations on the simulated data first
xx_stress_increase_ratio_sim_tophalf = []
xx_stress_increase_ratio_sim_lefthalf = []
yy_stress_increase_ratio_sim_tophalf = []
yy_stress_increase_ratio_sim_lefthalf = []


feedbacks = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for fb in feedbacks:
    xx_stress_increase_ratio_sim_tophalf.append(tissues_tophalf_FEM_simulation["feedback" + str(fb)]["xx_stress_increase_ratio"])
    yy_stress_increase_ratio_sim_tophalf.append(tissues_tophalf_FEM_simulation["feedback" + str(fb)]["yy_stress_increase_ratio"])
    xx_stress_increase_ratio_sim_lefthalf.append(tissues_lefthalf_FEM_simulation["feedback" + str(fb)]["yy_stress_increase_ratio"])
    yy_stress_increase_ratio_sim_lefthalf.append(tissues_lefthalf_FEM_simulation["feedback" + str(fb)]["xx_stress_increase_ratio"])

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(1.4, 3))
plt.subplots_adjust(hspace=0.4)  # adjust space in between plots

axes[0].plot(feedbacks, xx_stress_increase_ratio_sim_tophalf, color=colors_parent[0])
axes[0].plot(feedbacks, xx_stress_increase_ratio_sim_lefthalf, color=colors_parent[3])

axes[1].plot(feedbacks, yy_stress_increase_ratio_sim_tophalf, color=colors_parent[0])
axes[1].plot(feedbacks, yy_stress_increase_ratio_sim_lefthalf, color=colors_parent[3])

# add data points
# x=0
x = find_x_position_of_point_on_array(feedbacks, xx_stress_increase_ratio_sim_tophalf, xx_stress_increase_ratio_tophalf)
AC_xx_tophalf = x

axes[0].errorbar(x, xx_stress_increase_ratio_tophalf, yerr=xx_stress_increase_ratio_tophalf_err, mfc="w", color=colors_parent[0],
                 marker="v", ms=5, linewidth=0.5, ls="none", markeredgewidth=0.5)


x = find_x_position_of_point_on_array(feedbacks, xx_stress_increase_ratio_sim_lefthalf, xx_stress_increase_ratio_lefthalf)
AC_xx_lefthalf = x

axes[0].errorbar(x, xx_stress_increase_ratio_lefthalf, yerr=xx_stress_increase_ratio_lefthalf_err, mfc="w", color=colors_parent[3],
                 marker="o", ms=5, linewidth=0.5, ls="none", markeredgewidth=0.5)



x = find_x_position_of_point_on_array(feedbacks, yy_stress_increase_ratio_sim_tophalf, yy_stress_increase_ratio_tophalf)
AC_yy_tophalf = x

axes[1].errorbar(x, yy_stress_increase_ratio_tophalf, yerr=yy_stress_increase_ratio_tophalf_err, mfc="w", color=colors_parent[0],
                 marker="v", ms=5, linewidth=0.5, ls="none", markeredgewidth=0.5)

x = find_x_position_of_point_on_array(feedbacks, yy_stress_increase_ratio_sim_lefthalf, yy_stress_increase_ratio_lefthalf)
AC_yy_lefthalf = x

axes[1].errorbar(x, yy_stress_increase_ratio_lefthalf, yerr=yy_stress_increase_ratio_lefthalf_err, mfc="w", color=colors_parent[3],
                 marker="o", ms=5, linewidth=0.5, ls="none", markeredgewidth=0.5)



# set title
axes[0].set_title(label="$\mathrm{\sigma _ {xx}}$")
axes[1].set_title(label="$\mathrm{\sigma _ {yy}}$")
axes[1].set_xlabel(xlabel="Degree of active coupling")

# provide info on tick parameters
for ax in axes.flat:
    ax.minorticks_on()
    ax.tick_params(direction="in", which="minor", length=3, bottom=True, top=False, left=True, right=True)
    ax.tick_params(direction="in", which="major", length=6, bottom=True, top=False, left=True, right=True)
    ax.xaxis.set_ticks(xticks)

    ax.axvline(x=0, ymin=0, ymax=1, linewidth=0.5, color="grey", linestyle="--")


plt.savefig(figfolder + "F.png", dpi=300, bbox_inches="tight")
plt.savefig(figfolder + "F.svg", dpi=300, bbox_inches="tight")
plt.show()

# %%
summarized_data = pd.read_csv("summarized_data.csv")
MP_lefthalf = - np.nanmean(tissues_lefthalf_stim["MSM_data"]["AIC_baseline"])
MP_tophalf = np.nanmean(tissues_tophalf_stim["MSM_data"]["AIC_baseline"])

SP_lefthalf = - np.nanmean(tissues_lefthalf_stim["shape_data"]["actin_anisotropy_coefficient"])
SP_tophalf = np.nanmean(tissues_tophalf_stim["shape_data"]["actin_anisotropy_coefficient"])

d = {'condition': ["lefthalf", "tophalf"],
     'active coupling_x': np.array([AC_xx_lefthalf, AC_xx_tophalf]),
     'active coupling_y': np.array([AC_yy_lefthalf, AC_yy_tophalf]),
     'mechanical polarization': np.array([MP_lefthalf, MP_tophalf]),
     'structural polarization': np.array([SP_lefthalf, SP_tophalf])}

summarized_data_tissues = pd.DataFrame(data=d)
summarized_data = summarized_data.append(summarized_data_tissues)
colors = [colors_parent[0], colors_parent[1], colors_parent[3], colors_parent_dark[0], colors_parent_dark[1]]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(3, 2.8))
plt.subplots_adjust(wspace=0.6, hspace=0.4)  # adjust space in between plots

sns.scatterplot(data=summarized_data, y="active coupling_x", x="mechanical polarization", hue="condition", style="condition",
                palette=colors, legend=False, ax=axes[0, 0])

sns.scatterplot(data=summarized_data, y="active coupling_y", x="mechanical polarization", hue="condition", style="condition",
                palette=colors, legend=False, ax=axes[0, 1])

sns.scatterplot(data=summarized_data, y="active coupling_x", x="structural polarization", hue="condition", style="condition",
                palette=colors, legend=False, ax=axes[1, 0])

sns.scatterplot(data=summarized_data, y="active coupling_y", x="structural polarization", hue="condition", style="condition",
                palette=colors, legend=False, ax=axes[1, 1])

for ax in axes.flat:
    ax.set_ylim(-1, 1.1)
    ax.set_xlim(-1, 1.1)
    ax.minorticks_on()
    ax.tick_params(direction="in", which="minor", length=3, bottom=True, top=False, left=True, right=True)
    ax.tick_params(direction="in", which="major", length=6, bottom=True, top=False, left=True, right=True)
    ax.xaxis.set_ticks(xticks)
    ax.axvline(x=0, ymin=-1, ymax=1, linewidth=0.5, color="grey", linestyle="--")
    ax.axhline(y=0, xmin=-1, xmax=1, linewidth=0.5, color="grey", linestyle="--")

fig.savefig(figfolder + 'G.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'G.svg', dpi=300, bbox_inches="tight")
plt.show()
#  # %% plot figure 5C correlation plot of stress anisotropy and actin anisotropy
# #
# # # set up global plot parameters
# # # ******************************************************************************************************************************************
# # xticklabels = ['1to2', '1to1', '2to1']  # which labels to put on x-axis
# # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 2.5))  # create figure and axes
# # plt.subplots_adjust(wspace=0.45, hspace=0.45)  # adjust space in between plots
# #
# #
# # ylabeloffset = -7
# # xlabeloffset = 0
# # colors = [colors_parent[0], colors_parent[1], colors_parent[3]]  # defines colors for scatterplot
# #
# # y = 'actin_anisotropy_coefficient'
# # x = 'AIC_baseline'
# # hue = 'keys'
# # ymin = -0.5
# # ymax = 0.5
# # xmin = -1
# # xmax = 1
# # yticks = np.arange(-0.5, 0.6, 0.25)
# # xticks = np.arange(-1, 1.1, 0.5)
# # ylabel = "Degree of actin anisotropy"  # "'$\mathrm{\sigma_{x, MSM}}$'
# # xlabel = "Degree of stress anisotropy"  # '$\mathrm{\sigma_{x, CM}}$'
# #
# # corr, p = make_correlationplotsplots(x, y, hue, df, ax, xmin, xmax, ymin, ymax, xticks, yticks, xlabel, ylabel, colors)
# #
# # # add line with slope 1 for visualisation
# # # ax.plot([ymin, ymax], [0, 0], linewidth=0.5, linestyle=':', color='grey')
# # # ax.plot([45, 45], [xmin, xmax], linewidth=0.5, linestyle=':', color='grey')
# #
# # plt.text(0.21 * xmax + xmin, 1.05 * ymax, 'R = ' + str(corr))
# # # plt.text(0.52 * xmax, 1.1 * ymax, 'p = ' + '{:0.2e}'.format(p))
# #
# # plt.savefig(figfolder + 'C.png', dpi=300, bbox_inches="tight")
# # plt.savefig(figfolder + 'C.svg', dpi=300, bbox_inches="tight")
# # plt.show()
# def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True):
#     """
#     Produce a circular histogram of angles on ax.
#
#     Parameters
#     ----------
#     ax : matplotlib.axes._subplots.PolarAxesSubplot
#         axis instance created with subplot_kw=dict(projection='polar').
#
#     x : array
#         Angles to plot, expected in units of radians.
#
#     bins : int, optional
#         Defines the number of equal-width bins in the range. The default is 16.
#
#     density : bool, optional
#         If True plot frequency proportional to area. If False plot frequency
#         proportional to radius. The default is True.
#
#     offset : float, optional
#         Sets the offset for the location of the 0 direction in units of
#         radians. The default is 0.
#
#     gaps : bool, optional
#         Whether to allow gaps between bins. When gaps = False the bins are
#         forced to partition the entire [-pi, pi] range. The default is True.
#
#     Returns
#     -------
#     n : array or list of arrays
#         The number of values in each bin.
#
#     bins : array
#         The edges of the bins.
#
#     patches : `.BarContainer` or list of a single `.Polygon`
#         Container of individual artists used to create the histogram
#         or list of such containers if there are multiple input datasets.
#     """
#     # Wrap angles to [-pi, pi)
#     x = (x+np.pi) % (2*np.pi) - np.pi
#
#     # Force bins to partition entire circle
#     if not gaps:
#         bins = np.linspace(-np.pi, np.pi, num=bins+1)
#
#     # Bin data and record counts
#     n, bins = np.histogram(x, bins=bins)
#
#     # Compute width of each bin
#     widths = np.diff(bins)
#
#     # By default plot frequency proportional to area
#     if density:
#         # Area to assign each bin
#         area = n / x.size
#         # Calculate corresponding bin radius
#         radius = (area/np.pi) ** .5
#     # Otherwise plot frequency proportional to radius
#     else:
#         radius = n
#
#     # Plot data on ax
#     patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
#                      edgecolor='C0', fill=False, linewidth=1)
#
#     # Set the direction of the zero angle
#     ax.set_theta_offset(offset)
#
#     # Remove ylabels for area plots (they are mostly obstructive)
#     if density:
#         ax.set_yticks([])
#
#     return n, bins, patches
#
# # angles0 = np.random.normal(loc=0, scale=1, size=10000)
# # angles1 = np.random.uniform(0, 2*np.pi, size=1000)
# force_angles2= tissues_tophalf_stim["TFM_data"]["force_angles_weighted"] #Angles des forces sommées sur toute l'image pondérés par l'intensité
# fx_top = tissues_tophalf_stim["TFM_data"]["Tx"]* (stressmappixelsize ** 2) #vecteurs forces par pixel
# fy_top = tissues_tophalf_stim["TFM_data"]["Ty"]* (stressmappixelsize ** 2)
# fx_left = tissues_lefthalf_stim["TFM_data"]["Tx"]* (stressmappixelsize ** 2) #vecteurs forces par pixel
# fy_left = tissues_lefthalf_stim["TFM_data"]["Ty"]* (stressmappixelsize ** 2)
# fx_all = np.concatenate((fx_top,fx_left),axis=3)
# fy_all = np.concatenate((fy_top,fy_left),axis=3)
#
# f_norm=np.sqrt(fy_all[:,:,0:20,:]**2 + fx_all[:,:,0:20,:]**2)
# f_norm_mean = np.nanmean(f_norm[:,:,0:20,:], axis=2)
# weight_factors=np.rint(f_norm_mean*(1/np.min(f_norm_mean)))
# weight_factors_list=np.reshape(weight_factors,(120*120*107,1))
# force_angles1=-np.abs(np.arctan(np.divide(fy_left[:,:,0:20,:], fx_left[:,:,0:20,:])))
# # force_angles1=np.arctan2(fy_all[:,:,0:20,:], fx_all[:,:,0:20,:])
# force_angles_mean = np.nanmean(force_angles1[:,:,0:20,:], axis=2)
# force_angles_mean_list=np.reshape(force_angles_mean,(120*120*60,1))
#
# force_angle_final=[]
# for i in range (0, len(force_angles_mean_list)):
#     force_angle_final.extend([round(float(force_angles_mean_list[i]),8)]*int(weight_factors_list[i]))
# force_angle_final=np.array(force_angle_final)
#
# actin_angles_top = tissues_tophalf_stim["shape_data"]["actin_angles"]*np.pi/180
# actin_angles_left = tissues_lefthalf_stim["shape_data"]["actin_angles"]*np.pi/180
# actin_angles_all = np.concatenate((actin_angles_top,actin_angles_left),axis=0)-np.pi/2
#
# a=np.arange(0,2*np.pi,0.1)
# # Construct figure and axis to plot on
# fig, axes = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))
#
# circular_hist(axes[0], a, bins=63, density=True)
# circular_hist(axes[1], force_angles_mean_list, bins=20, density=True)
#
# plt.savefig(figfolder + 'Cbis.png', dpi=300, bbox_inches="tight")
# plt.savefig(figfolder + 'Cbis.svg', dpi=300, bbox_inches="tight")
#
# plt.show()