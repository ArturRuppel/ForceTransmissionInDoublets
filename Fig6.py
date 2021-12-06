# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 21:56:01 2021

@author: Artur Ruppel

"""
import os
import pickle
import pandas as pd
from scipy.stats import zscore
from plot_and_filter_functions import *
from scipy.interpolate import interp1d


mpl.rcParams['font.size'] = 8

# %% load data for plotting
folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"

tissues_tophalf_stim = pickle.load(open(folder + "analysed_data/tissues_tophalf_stim.dat", "rb"))
tissues_lefthalf_stim = pickle.load(open(folder + "analysed_data/tissues_lefthalf_stim.dat", "rb"))

tissues_tophalf_FEM_simulation = pickle.load(open(folder + "_FEM_simulations/FEM_tissues_up.dat", "rb"))
tissues_lefthalf_FEM_simulation = pickle.load(open(folder + "_FEM_simulations/FEM_tissues_side.dat", "rb"))


# define some colors for the plots
colors_parent = ['#026473', '#E3CC69', '#77C8A6', '#D96248']
colors_parent_dark = ['#01353D', '#564910', '#235741', '#A93B23']

figfolder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/_figure6/"
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

# add keys to dictionary with concatenated data
concatenated_data['keys'] = keys

# Creates DataFrame
df = pd.DataFrame(concatenated_data)

# convert to more convenient units for plotting
df_plot_units = df  # all units here are in SI units
df_plot_units['Es_baseline'] *= 1e12  # convert to fJ
df_plot_units['sigma_xx_baseline'] *= 1e3  # convert to mN/m
df_plot_units['sigma_yy_baseline'] *= 1e3  # convert to mN/m

# %% plot figure 5A, force maps

# prepare data first

# Calculate average maps over first 20 frames and all cells to get average maps
Tx_tophalf_average = np.nanmean(tissues_tophalf_stim["TFM_data"]["Ty"][:, :, 0:20, :], axis=(2, 3)) * 1e-3
Ty_tophalf_average = -np.nanmean(tissues_tophalf_stim["TFM_data"]["Tx"][:, :, 0:20, :], axis=(2, 3)) * 1e-3

Tx_lefthalf_average = np.nanmean(tissues_lefthalf_stim["TFM_data"]["Tx"][:, :, 0:20, :], axis=(2, 3)) * 1e-3
Ty_lefthalf_average = np.nanmean(tissues_lefthalf_stim["TFM_data"]["Ty"][:, :, 0:20, :], axis=(2, 3)) * 1e-3

# calculate amplitudes
T_tophalf_average = np.sqrt(Tx_tophalf_average ** 2 + Ty_tophalf_average ** 2)
T_lefthalf_average = np.sqrt(Tx_lefthalf_average ** 2 + Ty_lefthalf_average ** 2)


# set up plot parameters
# *****************************************************************************
n = 4  # every nth arrow will be plotted
pixelsize = 1.296  # in µm
pmax = 2  # kPa
pmin = 0


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(2.5, 3))

im = plot_forcemaps(axes[0], Tx_tophalf_average, Ty_tophalf_average, pixelsize, pmax, pmin)
plot_forcemaps(axes[1], Tx_lefthalf_average, Ty_lefthalf_average, pixelsize, pmax, pmin)


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
plt.text(0.3, 0.8, 'n=' + str(n_ts), transform=plt.figure(1).transFigure, color='w')
plt.text(0.3, 0.42, 'n=' + str(n_ls), transform=plt.figure(1).transFigure, color='w')

fig.savefig(figfolder + 'A.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'A.svg', dpi=300, bbox_inches="tight")

plt.show()
# %% plot figure 5B, stress maps

# prepare data first

# Calculate average maps over first 20 frames and all cells to get average maps
sigma_xx_tophalf_average = np.nanmean(tissues_tophalf_stim["MSM_data"]["sigma_xx"][:, :, 0:20, :], axis=(2, 3)) * 1e3  # convert to mN/m
sigma_yy_tophalf_average= np.nanmean(tissues_tophalf_stim["MSM_data"]["sigma_yy"][:, :, 0:20, :], axis=(2, 3)) * 1e3

sigma_xx_lefthalf_average = np.nanmean(tissues_lefthalf_stim["MSM_data"]["sigma_xx"][:, :, 0:20, :], axis=(2, 3)) * 1e3
sigma_yy_lefthalf_average = np.nanmean(tissues_lefthalf_stim["MSM_data"]["sigma_yy"][:, :, 0:20, :], axis=(2, 3)) * 1e3


# convert NaN to 0 to have black background
sigma_xx_lefthalf_average[np.isnan(sigma_xx_lefthalf_average)] = 0
sigma_yy_lefthalf_average[np.isnan(sigma_yy_lefthalf_average)] = 0

sigma_xx_tophalf_average[np.isnan(sigma_xx_tophalf_average)] = 0
sigma_yy_tophalf_average[np.isnan(sigma_yy_tophalf_average)] = 0


# set up plot parameters
# *****************************************************************************
n = 4  # every nth arrow will be plotted
pixelsize = 1.296  # in µm
pmin = 0
pmax = 10  # mN/m

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4, 3))

im = plot_stressmaps(axes[0, 0], sigma_xx_tophalf_average, pixelsize, pmax, pmin)
plot_stressmaps(axes[0, 1], sigma_yy_tophalf_average, pixelsize, pmax, pmin)
plot_stressmaps(axes[1, 0], sigma_xx_lefthalf_average, pixelsize, pmax, pmin)
plot_stressmaps(axes[1, 1], sigma_yy_lefthalf_average, pixelsize, pmax, pmin)

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
plt.text(0.18, 0.8, 'n=' + str(n_ts), transform=plt.figure(1).transFigure, color='w')
plt.text(0.18, 0.42, 'n=' + str(n_ls), transform=plt.figure(1).transFigure, color='w')

plt.text(0.48, 0.8, 'n=' + str(n_ts), transform=plt.figure(1).transFigure, color='w')
plt.text(0.48, 0.42, 'n=' + str(n_ls), transform=plt.figure(1).transFigure, color='w')

fig.savefig(figfolder + 'B.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'B.svg', dpi=300, bbox_inches="tight")
plt.show()

# # %% plot figure 5C correlation plot of stress anisotropy and actin anisotropy
#
# # set up global plot parameters
# # ******************************************************************************************************************************************
# xticklabels = ['1to2', '1to1', '2to1']  # which labels to put on x-axis
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 2.5))  # create figure and axes
# plt.subplots_adjust(wspace=0.45, hspace=0.45)  # adjust space in between plots
#
#
# ylabeloffset = -7
# xlabeloffset = 0
# colors = [colors_parent[0], colors_parent[1], colors_parent[3]]  # defines colors for scatterplot
#
# y = 'actin_anisotropy_coefficient'
# x = 'AIC_baseline'
# hue = 'keys'
# ymin = -0.5
# ymax = 0.5
# xmin = -1
# xmax = 1
# yticks = np.arange(-0.5, 0.6, 0.25)
# xticks = np.arange(-1, 1.1, 0.5)
# ylabel = "Degree of actin anisotropy"  # "'$\mathrm{\sigma_{x, MSM}}$'
# xlabel = "Degree of stress anisotropy"  # '$\mathrm{\sigma_{x, CM}}$'
#
# corr, p = make_correlationplotsplots(x, y, hue, df, ax, xmin, xmax, ymin, ymax, xticks, yticks, xlabel, ylabel, colors)
#
# # add line with slope 1 for visualisation
# # ax.plot([ymin, ymax], [0, 0], linewidth=0.5, linestyle=':', color='grey')
# # ax.plot([45, 45], [xmin, xmax], linewidth=0.5, linestyle=':', color='grey')
#
# plt.text(0.21 * xmax + xmin, 1.05 * ymax, 'R = ' + str(corr))
# # plt.text(0.52 * xmax, 1.1 * ymax, 'p = ' + '{:0.2e}'.format(p))
#
# plt.savefig(figfolder + 'C.png', dpi=300, bbox_inches="tight")
# plt.savefig(figfolder + 'C.svg', dpi=300, bbox_inches="tight")
# plt.show()
# %% filter data to remove cells that have an unstable baseline

def filter_data_main(data, threshold, title):
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

    new_N = np.sum(baselinefilter)
    print(title + ": " + str(baselinefilter.shape[0] - new_N) + " cells were filtered out")

    return data

# maximal allowed slope for linear fit of baseline
threshold = 0.1

# tissues_full_stim = filter_data_main(tissues_full_stim, "tissues_full_stim")
tissues_lefthalf_stim = filter_data_main(tissues_lefthalf_stim, threshold, "tissues_lefthalf_stim")
tissues_tophalf_stim = filter_data_main(tissues_tophalf_stim, threshold, "tissues_tophalf_stim")

# %% prepare dataframe again after filtering

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

# add keys to dictionary with concatenated data
concatenated_data['keys'] = keys

# Creates DataFrame
df = pd.DataFrame(concatenated_data)

# convert to more convenient units for plotting
df_plot_units = df  # all units here are in SI units
df_plot_units['Es_baseline'] *= 1e12  # convert to fJ
df_plot_units['sigma_xx_baseline'] *= 1e3  # convert to mN/m
df_plot_units['sigma_yy_baseline'] *= 1e3  # convert to mN/m

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
sigma_max = 1  # kPa
sigma_min = -1  # kPa

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
plt.text(0.40, 0.85, 'n=' + str(n_ls), transform=plt.figure(1).transFigure, color='black')
plt.text(0.40, 0.40, 'n=' + str(n_ts), transform=plt.figure(1).transFigure, color='black')

# save figure
fig.savefig(figfolder + 'D.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'D.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 5E

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.linspace(-78, 78, 120)
x = x[::2]  # downsample data for nicer plotting
xticks = np.arange(-60, 60.1, 30)  # define where the major ticks are gonna be
xlabel = 'position [µm]'
ymin = -0.2
ymax = 1.4
yticks = np.arange(ymin, ymax + 0.001, 0.4)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(3, 3))  # create figure and axes
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


feedbacks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for fb in feedbacks:
    xx_stress_increase_ratio_sim_tophalf.append(tissues_tophalf_FEM_simulation["feedback" + str(fb)]["xx_stress_increase_ratio"])
    yy_stress_increase_ratio_sim_tophalf.append(tissues_tophalf_FEM_simulation["feedback" + str(fb)]["yy_stress_increase_ratio"])
    xx_stress_increase_ratio_sim_lefthalf.append(tissues_lefthalf_FEM_simulation["feedback" + str(fb)]["xx_stress_increase_ratio"])
    yy_stress_increase_ratio_sim_lefthalf.append(tissues_lefthalf_FEM_simulation["feedback" + str(fb)]["yy_stress_increase_ratio"])

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(1.4, 3))
plt.subplots_adjust(hspace=0.4)  # adjust space in between plots

axes[0].plot(feedbacks, xx_stress_increase_ratio_sim_tophalf, color=colors_parent[0])
axes[0].plot(feedbacks, xx_stress_increase_ratio_sim_lefthalf, color=colors_parent[3])

axes[1].plot(feedbacks, yy_stress_increase_ratio_sim_tophalf, color=colors_parent[0])
axes[1].plot(feedbacks, yy_stress_increase_ratio_sim_lefthalf, color=colors_parent[3])

# add data points
x = find_x_position_of_point_on_array(feedbacks, xx_stress_increase_ratio_sim_tophalf, xx_stress_increase_ratio_tophalf)
axes[0].errorbar(x, xx_stress_increase_ratio_tophalf, yerr=xx_stress_increase_ratio_tophalf_err, mfc="w", color=colors_parent[0],
                 marker="v", ms=5, linewidth=0.5, ls="none", markeredgewidth=0.5)


x = find_x_position_of_point_on_array(feedbacks, xx_stress_increase_ratio_sim_lefthalf, xx_stress_increase_ratio_lefthalf)

axes[0].errorbar(x, xx_stress_increase_ratio_lefthalf, yerr=xx_stress_increase_ratio_lefthalf_err, mfc="w", color=colors_parent[3],
                 marker="o", ms=5, linewidth=0.5, ls="none", markeredgewidth=0.5)



# x = find_x_position_of_point_on_array(feedbacks, yy_stress_increase_ratio_sim_tophalf, yy_stress_increase_ratio_tophalf)
x = -0.08

axes[1].errorbar(x, yy_stress_increase_ratio_tophalf, yerr=yy_stress_increase_ratio_tophalf_err, mfc="w", color=colors_parent[0],
                 marker="v", ms=5, linewidth=0.5, ls="none", markeredgewidth=0.5)

x = find_x_position_of_point_on_array(feedbacks, yy_stress_increase_ratio_sim_lefthalf, yy_stress_increase_ratio_lefthalf)

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

# plt.ylabel("Normalized xx-stress increase of \n non-activated area")
# plt.suptitle("Stress coupling", y=0.95)
plt.savefig(figfolder + "F.png", dpi=300, bbox_inches="tight")
plt.savefig(figfolder + "F.svg", dpi=300, bbox_inches="tight")
plt.show()

