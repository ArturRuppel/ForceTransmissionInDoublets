"""

@author: Artur Ruppel

"""
import os
import pickle
import pandas as pd
from scipy.stats import zscore
from plot_and_filter_functions import *
from scipy.interpolate import interp1d

# define some colors for the plots
colors_parent = ['#026473', '#E3CC69', '#77C8A6', '#D96248']
colors_parent_dark = ['#01353D', '#564910', '#235741', '#A93B23']

folder = "C:/Users/aruppel/Documents/_forcetransmission_in_cell_doublets_raw/"
figfolder = folder + "_Figure5/"
if not os.path.exists(figfolder):
    os.mkdir(figfolder)

pixelsize = 0.864  # in µm
# %% load data for plotting
AR1to2d_halfstim = pickle.load(open(folder + "analysed_data/AR1to2d_halfstim.dat", "rb"))
AR1to1d_halfstim = pickle.load(open(folder + "analysed_data/AR1to1d_halfstim.dat", "rb"))
AR2to1d_halfstim = pickle.load(open(folder + "analysed_data/AR2to1d_halfstim.dat", "rb"))

AR1to2_FEM_simulation = pickle.load(open(folder + "_FEM_simulations/FEM_1to2.dat", "rb"))
AR1to1_FEM_simulation = pickle.load(open(folder + "_FEM_simulations/FEM_singlets.dat", "rb"))
AR2to1_FEM_simulation = pickle.load(open(folder + "_FEM_simulations/FEM_2to1.dat", "rb"))

# %% prepare dataframe for boxplots

# initialize empty dictionaries
concatenated_data_1to2d = {}
concatenated_data_1to1d = {}
concatenated_data_2to1d = {}
concatenated_data = {}

# loop over all keys
for key1 in AR1to1d_halfstim:  # keys are the same for all dictionaries so I'm just taking one example here
    for key2 in AR1to1d_halfstim[key1]:
        if AR1to1d_halfstim[key1][key2].ndim == 1:  # only 1D data can be stored in the data frame
            # concatenate values from different experiments
            concatenated_data_1to2d[key2] = AR1to2d_halfstim[key1][key2]
            concatenated_data_1to1d[key2] = AR1to1d_halfstim[key1][key2]
            concatenated_data_2to1d[key2] = AR2to1d_halfstim[key1][key2]

            concatenated_data[key2] = np.concatenate(
                (concatenated_data_1to2d[key2], concatenated_data_1to1d[key2], concatenated_data_2to1d[key2]))
key2 = 'Es_baseline'
# get number of elements for both condition
n_1to2d = concatenated_data_1to2d[key2].shape[0]
n_1to1d = concatenated_data_1to1d[key2].shape[0]
n_2to1d = concatenated_data_2to1d[key2].shape[0]

# create a list of keys with the same dimensions as the data
keys1to2d = ['AR1to2d' for i in range(n_1to2d)]
keys1to1d = ['AR1to1d' for i in range(n_1to1d)]
keys2to1d = ['AR2to1d' for i in range(n_2to1d)]
keys = np.concatenate((keys1to2d, keys1to1d, keys2to1d))

# add keys to dictionary with concatenated data
concatenated_data['keys'] = keys

# Creates DataFrame
df = pd.DataFrame(concatenated_data)

# convert to more convenient units for plotting
df['Es_baseline'] *= 1e12  # convert to fJ
df['spreadingsize_baseline'] *= 1e12  # convert to µm²
df['sigma_xx_baseline'] *= 1e3  # convert to mN/m
df['sigma_yy_baseline'] *= 1e3  # convert to mN/m

# %% plot figure 5A, force maps

# prepare data first

# Calculate average maps over first 20 frames and all cells to get average maps
Tx_1to2d_average = np.nanmean(AR1to2d_halfstim["TFM_data"]["Tx"][:, :, 0:20, :], axis=(2, 3))
Ty_1to2d_average = np.nanmean(AR1to2d_halfstim["TFM_data"]["Ty"][:, :, 0:20, :], axis=(2, 3))

Tx_1to1d_average = np.nanmean(AR1to1d_halfstim["TFM_data"]["Tx"][:, :, 0:20, :], axis=(2, 3))
Ty_1to1d_average = np.nanmean(AR1to1d_halfstim["TFM_data"]["Ty"][:, :, 0:20, :], axis=(2, 3))

Tx_2to1d_average = np.nanmean(AR2to1d_halfstim["TFM_data"]["Tx"][:, :, 0:20, :], axis=(2, 3))
Ty_2to1d_average = np.nanmean(AR2to1d_halfstim["TFM_data"]["Ty"][:, :, 0:20, :], axis=(2, 3))

# calculate amplitudes
T_1to2d_average = np.sqrt(Tx_1to2d_average ** 2 + Ty_1to2d_average ** 2)
T_1to1d_average = np.sqrt(Tx_1to1d_average ** 2 + Ty_1to1d_average ** 2)
T_2to1d_average = np.sqrt(Tx_2to1d_average ** 2 + Ty_2to1d_average ** 2)

# crop maps
crop_start = 2
crop_end = 90

Tx_1to2d_average_crop = Tx_1to2d_average[crop_start:crop_end, crop_start:crop_end] * 1e-3  # convert to kPa
Ty_1to2d_average_crop = Ty_1to2d_average[crop_start:crop_end, crop_start:crop_end] * 1e-3
T_1to2d_average_crop = T_1to2d_average[crop_start:crop_end, crop_start:crop_end] * 1e-3

Tx_1to1d_average_crop = Tx_1to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e-3  # convert to kPa
Ty_1to1d_average_crop = Ty_1to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e-3
T_1to1d_average_crop = T_1to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e-3

Tx_2to1d_average_crop = Tx_2to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e-3  # convert to kPa
Ty_2to1d_average_crop = Ty_2to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e-3
T_2to1d_average_crop = T_2to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e-3

# set up plot parameters
# *****************************************************************************
n = 4  # every nth arrow will be plotted
pixelsize = 0.864  # in µm
pmin = 0
pmax = 2  # kPa

#
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(2.5, 4))

im = plot_forcemaps(axes[0], Tx_1to2d_average_crop, Ty_1to2d_average_crop, pixelsize, pmax, pmin)
plot_forcemaps(axes[1], Tx_1to1d_average_crop, Ty_1to1d_average_crop, pixelsize, pmax, pmin)
plot_forcemaps(axes[2], Tx_2to1d_average_crop, Ty_2to1d_average_crop, pixelsize, pmax, pmin)

# draw pattern
draw_pattern_1to2(axes[0])
draw_pattern_1to1(axes[1])
draw_pattern_2to1(axes[2])

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
plt.suptitle('Traction forces', y=0.92, x=0.54)

# add annotations
plt.text(0.48, 0.853, 'n=' + str(n_1to2d), transform=plt.figure(1).transFigure, color='w')
plt.text(0.48, 0.598, 'n=' + str(n_1to1d), transform=plt.figure(1).transFigure, color='w')
plt.text(0.48, 0.343, 'n=' + str(n_2to1d), transform=plt.figure(1).transFigure, color='w')

fig.savefig(figfolder + 'A.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'A.svg', dpi=300, bbox_inches="tight")

plt.show()
# %% plot figure 5B, stress maps

# prepare data first

# Calculate average maps over first 20 frames and all cells to get average maps
sigma_xx_1to2d_average = np.nanmean(AR1to2d_halfstim["MSM_data"]["sigma_xx"][:, :, 0:20, :], axis=(2, 3))
sigma_yy_1to2d_average = np.nanmean(AR1to2d_halfstim["MSM_data"]["sigma_yy"][:, :, 0:20, :], axis=(2, 3))

sigma_xx_1to1d_average = np.nanmean(AR1to1d_halfstim["MSM_data"]["sigma_xx"][:, :, 0:20, :], axis=(2, 3))
sigma_yy_1to1d_average = np.nanmean(AR1to1d_halfstim["MSM_data"]["sigma_yy"][:, :, 0:20, :], axis=(2, 3))

sigma_xx_2to1d_average = np.nanmean(AR2to1d_halfstim["MSM_data"]["sigma_xx"][:, :, 0:20, :], axis=(2, 3))
sigma_yy_2to1d_average = np.nanmean(AR2to1d_halfstim["MSM_data"]["sigma_yy"][:, :, 0:20, :], axis=(2, 3))

# convert NaN to 0 to have black background
sigma_xx_1to2d_average[np.isnan(sigma_xx_1to2d_average)] = 0
sigma_yy_1to2d_average[np.isnan(sigma_yy_1to2d_average)] = 0

sigma_xx_1to1d_average[np.isnan(sigma_xx_1to1d_average)] = 0
sigma_yy_1to1d_average[np.isnan(sigma_yy_1to1d_average)] = 0

sigma_xx_2to1d_average[np.isnan(sigma_xx_2to1d_average)] = 0
sigma_yy_2to1d_average[np.isnan(sigma_yy_2to1d_average)] = 0

# crop maps
crop_start = 2
crop_end = 90

sigma_xx_1to2d_average_crop = sigma_xx_1to2d_average[crop_start:crop_end, crop_start:crop_end] * 1e3  # convert to mN/m
sigma_yy_1to2d_average_crop = sigma_yy_1to2d_average[crop_start:crop_end, crop_start:crop_end] * 1e3

sigma_xx_1to1d_average_crop = sigma_xx_1to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e3
sigma_yy_1to1d_average_crop = sigma_yy_1to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e3

sigma_xx_2to1d_average_crop = sigma_xx_2to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e3
sigma_yy_2to1d_average_crop = sigma_yy_2to1d_average[crop_start:crop_end, crop_start:crop_end] * 1e3

# set up plot parameters
# *****************************************************************************
n = 4  # every nth arrow will be plotted
pixelsize = 0.864  # in µm
pmin = 0
pmax = 10  # mN/m

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(3.32, 4))

im = plot_stressmaps(axes[0, 0], sigma_xx_1to2d_average_crop, pixelsize, pmax, pmin)
plot_stressmaps(axes[1, 0], sigma_xx_1to1d_average_crop, pixelsize, pmax, pmin)
plot_stressmaps(axes[2, 0], sigma_xx_2to1d_average_crop, pixelsize, pmax, pmin)
plot_stressmaps(axes[0, 1], sigma_yy_1to2d_average_crop, pixelsize, pmax, pmin)
plot_stressmaps(axes[1, 1], sigma_yy_1to1d_average_crop, pixelsize, pmax, pmin)
plot_stressmaps(axes[2, 1], sigma_yy_2to1d_average_crop, pixelsize, pmax, pmin)

# adjust space in between plots
plt.subplots_adjust(wspace=0, hspace=0)

# remove axes
for ax in axes.flat:
    ax.axis('off')
    aspectratio = 1.0
    ratio_default = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.set_aspect(ratio_default * aspectratio)

# draw pattern
draw_pattern_1to2(axes[0, 0])
draw_pattern_1to2(axes[0, 1])
draw_pattern_1to1(axes[1, 0])
draw_pattern_1to1(axes[1, 1])
draw_pattern_2to1(axes[2, 0])
draw_pattern_2to1(axes[2, 1])

# add colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist())
cbar.ax.set_title('mN/m')

# add title
plt.suptitle('Cell stresses', y=0.97, x=0.43, size=10)
plt.text(-80, 195, '$\mathrm{\sigma _ {xx}}$', size=10)
plt.text(-7, 195, '$\mathrm{\sigma _ {yy}}$', size=10)

# add annotations
plt.text(0.25, 0.853, 'n=' + str(n_1to2d), transform=plt.figure(1).transFigure, color='w')
plt.text(0.25, 0.598, 'n=' + str(n_1to1d), transform=plt.figure(1).transFigure, color='w')
plt.text(0.25, 0.343, 'n=' + str(n_2to1d), transform=plt.figure(1).transFigure, color='w')

plt.text(0.55, 0.853, 'n=' + str(n_1to2d), transform=plt.figure(1).transFigure, color='w')
plt.text(0.55, 0.598, 'n=' + str(n_1to1d), transform=plt.figure(1).transFigure, color='w')
plt.text(0.55, 0.343, 'n=' + str(n_2to1d), transform=plt.figure(1).transFigure, color='w')

fig.savefig(figfolder + 'B.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'B.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 5C correlation plot of stress anisotropy and actin anisotropy

# set up global plot parameters
# ******************************************************************************************************************************************
xticklabels = ['1to2', '1to1', '2to1']  # which labels to put on x-axis
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 2.5))  # create figure and axes
plt.subplots_adjust(wspace=0.45, hspace=0.45)  # adjust space in between plots

ylabeloffset = -7
xlabeloffset = 0
colors = [colors_parent[0], colors_parent[1], colors_parent[3]]  # defines colors for scatterplot

y = 'actin_anisotropy_coefficient'
x = 'AIC_baseline'
hue = 'keys'
ymin = -1
ymax = 1
xmin = -1
xmax = 1
yticks = np.arange(-1, 1.1, 0.5)
xticks = np.arange(-1, 1.1, 0.5)
ylabel = "Structural polarization (actin)"  # "'$\mathrm{\sigma_{x, MSM}}$'
xlabel = "Mechanical polarization"  # '$\mathrm{\sigma_{x, CM}}$'

corr, p = make_correlationplotsplots(x, y, hue, df, ax, xmin, xmax, ymin, ymax, xticks, yticks, xlabel, ylabel, colors)

# add line with slope 1 for visualisation
# ax.plot([ymin, ymax], [0, 0], linewidth=0.5, linestyle=':', color='grey')
# ax.plot([45, 45], [xmin, xmax], linewidth=0.5, linestyle=':', color='grey')

plt.text(0.21 * xmax + xmin, 1.05 * ymax, 'R = ' + str(corr))
# plt.text(0.52 * xmax, 1.1 * ymax, 'p = ' + '{:0.2e}'.format(p))

plt.savefig(figfolder + 'C.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'C.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 5C_alt, Stress anisotropy coefficient boxplot

# define plot parameters that are valid for the whole figure
# ******************************************************************************************************************************************
colors = [colors_parent[0], colors_parent[1], colors_parent[3]]  # defines colors
sns.set_palette(sns.color_palette(colors))  # sets colors
box_pairs = [('AR1to1d', 'AR1to2d'), ('AR2to1d', 'AR1to2d'), ('AR2to1d', 'AR1to1d')]  # which groups to perform statistical test on
xticklabels = ['1to2', '1to1', '2to1']  # which labels to put on x-axis
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2, 4))  # create figure instance
# ******************************************************************************************************************************************


# Set up plot parameters for first panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'AIC_baseline'  # variable that goes on the y-axis
ymin = -1  # minimum value on y-axis
ymax = 1.5  # maximum value on y-axis
yticks = np.arange(-1, 1.5, 0.5)  # define where to put major ticks on y-axis
stat_annotation_offset = 0.1
# vertical offset of statistical annotation
ylabel = None  # which label to put on y-axis
title = 'Stress anisotropy coefficient'  # title of plot

# make plots
make_box_and_swarmplots_with_test(x, y, df, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title, colors)

# # save plot to file
plt.savefig(figfolder + 'C_alt.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'C_alt.svg', dpi=300, bbox_inches="tight")
plt.show()


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

AR1to2d_halfstim = filter_data_main(AR1to2d_halfstim, threshold, "AR1to2d_halfstim")
AR1to1d_halfstim = filter_data_main(AR1to1d_halfstim, threshold, "AR1to1d_halfstim")
AR2to1d_halfstim = filter_data_main(AR2to1d_halfstim, threshold, "AR2to1d_halfstim")

# %% prepare dataframe again after filtering

# initialize empty dictionaries
concatenated_data_1to2d = {}
concatenated_data_1to1d = {}
concatenated_data_2to1d = {}
concatenated_data = {}

# loop over all keys
for key1 in AR1to1d_halfstim:  # keys are the same for all dictionaries so I'm just taking one example here
    for key2 in AR1to1d_halfstim[key1]:
        if AR1to1d_halfstim[key1][key2].ndim == 1:  # only 1D data can be stored in the data frame
            # concatenate values from different experiments
            concatenated_data_1to2d[key2] = AR1to2d_halfstim[key1][key2]
            concatenated_data_1to1d[key2] = AR1to1d_halfstim[key1][key2]
            concatenated_data_2to1d[key2] = AR2to1d_halfstim[key1][key2]

            concatenated_data[key2] = np.concatenate(
                (concatenated_data_1to2d[key2], concatenated_data_1to1d[key2], concatenated_data_2to1d[key2]))
key2 = 'Es_baseline'
# get number of elements for both condition
n_1to2d = concatenated_data_1to2d[key2].shape[0]
n_1to1d = concatenated_data_1to1d[key2].shape[0]
n_2to1d = concatenated_data_2to1d[key2].shape[0]

# create a list of keys with the same dimensions as the data
keys1to2d = ['AR1to2d' for i in range(n_1to2d)]
keys1to1d = ['AR1to1d' for i in range(n_1to1d)]
keys2to1d = ['AR2to1d' for i in range(n_2to1d)]
keys = np.concatenate((keys1to2d, keys1to1d, keys2to1d))

# add keys to dictionary with concatenated data
concatenated_data['keys'] = keys

# Creates DataFrame
df = pd.DataFrame(concatenated_data)

# # remove outliers
# filter_outlier1 = np.abs(zscore(df["attenuation_position"])) < 1
# filter_outlier2 = np.abs(zscore(df["attenuation_length"])) < 1
# filter_outlier = np.all((filter_outlier1, filter_outlier2), axis=0)
# df = df[filter_outlier]

stressmappixelsize = 0.864  # in µm
# convert to more convenient units for plotting
df['Es_baseline'] *= 1e12  # convert to fJ
df['spreadingsize_baseline'] *= 1e12  # convert to µm²
df['sigma_xx_baseline'] *= 1e3  # convert to mN/m
df['sigma_yy_baseline'] *= 1e3  # convert to mN/m
# df["left_asymptote"] *= 1e3  # convert to mN/m
# df["right_asymptote"] *= 1e3  # convert to mN/m
df['cell_width_center_baseline'] *= stressmappixelsize / 8  # convert from pixel to µm
# df["attenuation_position"] *= stressmappixelsize  # convert to µm
# df["attenuation_length"] *= stressmappixelsize  # convert to µm

# %% calculate stress increase ratio of right vs all
sigma_xx_x_profile_increase_1to2 = np.nanmean(AR1to2d_halfstim["MSM_data"]["sigma_xx_x_profile_increase"], axis=1)
sigma_xx_x_profile_increase_1to2_sem = np.nanstd(AR1to2d_halfstim["MSM_data"]["sigma_xx_x_profile_increase"], axis=1) / np.sqrt(
    np.shape(AR1to2d_halfstim["MSM_data"]["sigma_xx_x_profile_increase"])[1])

sigma_xx_x_profile_increase_1to1 = np.nanmean(AR1to1d_halfstim["MSM_data"]["sigma_xx_x_profile_increase"], axis=1)
sigma_xx_x_profile_increase_1to1_sem = np.nanstd(AR1to1d_halfstim["MSM_data"]["sigma_xx_x_profile_increase"], axis=1) / np.sqrt(
    np.shape(AR1to1d_halfstim["MSM_data"]["sigma_xx_x_profile_increase"])[1])

sigma_xx_x_profile_increase_2to1 = np.nanmean(AR2to1d_halfstim["MSM_data"]["sigma_xx_x_profile_increase"], axis=1)
sigma_xx_x_profile_increase_2to1_sem = np.nanstd(AR2to1d_halfstim["MSM_data"]["sigma_xx_x_profile_increase"], axis=1) / np.sqrt(
    np.shape(AR2to1d_halfstim["MSM_data"]["sigma_xx_x_profile_increase"])[1])

sigma_yy_x_profile_increase_1to2 = np.nanmean(AR1to2d_halfstim["MSM_data"]["sigma_yy_x_profile_increase"], axis=1)
sigma_yy_x_profile_increase_1to2_sem = np.nanstd(AR1to2d_halfstim["MSM_data"]["sigma_yy_x_profile_increase"], axis=1) / np.sqrt(
    np.shape(AR1to2d_halfstim["MSM_data"]["sigma_yy_x_profile_increase"])[1])

sigma_yy_x_profile_increase_1to1 = np.nanmean(AR1to1d_halfstim["MSM_data"]["sigma_yy_x_profile_increase"], axis=1)
sigma_yy_x_profile_increase_1to1_sem = np.nanstd(AR1to1d_halfstim["MSM_data"]["sigma_yy_x_profile_increase"], axis=1) / np.sqrt(
    np.shape(AR1to1d_halfstim["MSM_data"]["sigma_yy_x_profile_increase"])[1])

sigma_yy_x_profile_increase_2to1 = np.nanmean(AR2to1d_halfstim["MSM_data"]["sigma_yy_x_profile_increase"], axis=1)
sigma_yy_x_profile_increase_2to1_sem = np.nanstd(AR2to1d_halfstim["MSM_data"]["sigma_yy_x_profile_increase"], axis=1) / np.sqrt(
    np.shape(AR2to1d_halfstim["MSM_data"]["sigma_yy_x_profile_increase"])[1])

center = int(sigma_xx_x_profile_increase_1to1.shape[0] / 2)

# calculate error with propagation of uncertainty
SI_xx_right_1to2 = np.nansum(sigma_xx_x_profile_increase_1to2[center:-1])
SI_xx_right_1to2_err = np.sqrt(np.nansum(sigma_xx_x_profile_increase_1to2_sem[center:-1] ** 2))
SI_xx_left_1to2 = np.nansum(sigma_xx_x_profile_increase_1to2[0:center])
SI_xx_left_1to2_err = np.sqrt(np.nansum(sigma_xx_x_profile_increase_1to2_sem[0:center] ** 2))

SI_xx_right_1to1 = np.nansum(sigma_xx_x_profile_increase_1to1[center:-1])
SI_xx_right_1to1_err = np.sqrt(np.nansum(sigma_xx_x_profile_increase_1to1_sem[center:-1] ** 2))
SI_xx_left_1to1 = np.nansum(sigma_xx_x_profile_increase_1to1[0:center])
SI_xx_left_1to1_err = np.sqrt(np.nansum(sigma_xx_x_profile_increase_1to1_sem[0:center] ** 2))

SI_xx_right_2to1 = np.nansum(sigma_xx_x_profile_increase_2to1[center:-1])
SI_xx_right_2to1_err = np.sqrt(np.nansum(sigma_xx_x_profile_increase_2to1_sem[center:-1] ** 2))
SI_xx_left_2to1 = np.nansum(sigma_xx_x_profile_increase_2to1[0:center])
SI_xx_left_2to1_err = np.sqrt(np.nansum(sigma_xx_x_profile_increase_2to1_sem[0:center] ** 2))

SI_yy_right_1to2 = np.nansum(sigma_yy_x_profile_increase_1to2[center:-1])
SI_yy_right_1to2_err = np.sqrt(np.nansum(sigma_yy_x_profile_increase_1to2_sem[center:-1] ** 2))
SI_yy_left_1to2 = np.nansum(sigma_yy_x_profile_increase_1to2[0:center])
SI_yy_left_1to2_err = np.sqrt(np.nansum(sigma_yy_x_profile_increase_1to2_sem[0:center] ** 2))

SI_yy_right_1to1 = np.nansum(sigma_yy_x_profile_increase_1to1[center:-1])
SI_yy_right_1to1_err = np.sqrt(np.nansum(sigma_yy_x_profile_increase_1to1_sem[center:-1] ** 2))
SI_yy_left_1to1 = np.nansum(sigma_yy_x_profile_increase_1to1[0:center])
SI_yy_left_1to1_err = np.sqrt(np.nansum(sigma_yy_x_profile_increase_1to1_sem[0:center] ** 2))

SI_yy_right_2to1 = np.nansum(sigma_yy_x_profile_increase_2to1[center:-1])
SI_yy_right_2to1_err = np.sqrt(np.nansum(sigma_yy_x_profile_increase_2to1_sem[center:-1] ** 2))
SI_yy_left_2to1 = np.nansum(sigma_yy_x_profile_increase_2to1[0:center])
SI_yy_left_2to1_err = np.sqrt(np.nansum(sigma_yy_x_profile_increase_2to1_sem[0:center] ** 2))

# calculate error with propagation of uncertainty
xx_stress_increase_ratio_1to2 = SI_xx_right_1to2 / (SI_xx_left_1to2 + SI_xx_right_1to2)
xx_stress_increase_ratio_1to2_err = (SI_xx_right_1to2_err * SI_xx_left_1to2 + SI_xx_left_1to2_err * SI_xx_right_1to2) / (
            (SI_xx_left_1to2 + SI_xx_right_1to2) ** 2)

xx_stress_increase_ratio_1to1 = SI_xx_right_1to1 / (SI_xx_left_1to1 + SI_xx_right_1to1)
xx_stress_increase_ratio_1to1_err = (SI_xx_right_1to1_err * SI_xx_left_1to1 + SI_xx_left_1to1_err * SI_xx_right_1to1) / (
            (SI_xx_left_1to1 + SI_xx_right_1to1) ** 2)

xx_stress_increase_ratio_2to1 = SI_xx_right_2to1 / (SI_xx_left_2to1 + SI_xx_right_2to1)
xx_stress_increase_ratio_2to1_err = (SI_xx_right_2to1_err * SI_xx_left_2to1 + SI_xx_left_2to1_err * SI_xx_right_2to1) / (
            (SI_xx_left_2to1 + SI_xx_right_2to1) ** 2)

yy_stress_increase_ratio_1to2 = SI_yy_right_1to2 / (SI_yy_left_1to2 + SI_yy_right_1to2)
yy_stress_increase_ratio_1to2_err = (SI_yy_right_1to2_err * SI_yy_left_1to2 + SI_yy_left_1to2_err * SI_yy_right_1to2) / (
            (SI_yy_left_1to2 + SI_yy_right_1to2) ** 2)

yy_stress_increase_ratio_1to1 = SI_yy_right_1to1 / (SI_yy_left_1to1 + SI_yy_right_1to1)
yy_stress_increase_ratio_1to1_err = (SI_yy_right_1to1_err * SI_yy_left_1to1 + SI_yy_left_1to1_err * SI_yy_right_1to1) / (
            (SI_yy_left_1to1 + SI_yy_right_1to1) ** 2)

yy_stress_increase_ratio_2to1 = SI_yy_right_2to1 / (SI_yy_left_2to1 + SI_yy_right_2to1)
yy_stress_increase_ratio_2to1_err = (SI_yy_right_2to1_err * SI_yy_left_2to1 + SI_yy_left_2to1_err * SI_yy_right_2to1) / (
            (SI_yy_left_2to1 + SI_yy_right_2to1) ** 2)

# %% plot figure 5D, stress map differences

# prepare data first

# concatenate TFM maps from different experiments and calculate average maps over first 20 frames and all cells to get average maps
sigma_xx_1to2d_diff = np.nanmean(
    AR1to2d_halfstim["MSM_data"]["sigma_xx"][:, :, 32, :] - AR1to2d_halfstim["MSM_data"]["sigma_xx"][:, :, 20, :], axis=2)
sigma_xx_1to1d_diff = np.nanmean(
    AR1to1d_halfstim["MSM_data"]["sigma_xx"][:, :, 32, :] - AR1to1d_halfstim["MSM_data"]["sigma_xx"][:, :, 20, :], axis=2)
sigma_xx_2to1d_diff = np.nanmean(
    AR2to1d_halfstim["MSM_data"]["sigma_xx"][:, :, 32, :] - AR2to1d_halfstim["MSM_data"]["sigma_xx"][:, :, 20, :], axis=2)

sigma_yy_1to2d_diff = np.nanmean(
    AR1to2d_halfstim["MSM_data"]["sigma_yy"][:, :, 32, :] - AR1to2d_halfstim["MSM_data"]["sigma_yy"][:, :, 20, :], axis=2)
sigma_yy_1to1d_diff = np.nanmean(
    AR1to1d_halfstim["MSM_data"]["sigma_yy"][:, :, 32, :] - AR1to1d_halfstim["MSM_data"]["sigma_yy"][:, :, 20, :], axis=2)
sigma_yy_2to1d_diff = np.nanmean(
    AR2to1d_halfstim["MSM_data"]["sigma_yy"][:, :, 32, :] - AR2to1d_halfstim["MSM_data"]["sigma_yy"][:, :, 20, :], axis=2)

# crop maps
crop_start = 2
crop_end = 90

sigma_xx_1to2d_diff_crop = sigma_xx_1to2d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3  # convert to mN/m
sigma_xx_1to1d_diff_crop = sigma_xx_1to1d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3  # convert to mN/m
sigma_xx_2to1d_diff_crop = sigma_xx_2to1d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3  # convert to mN/m

sigma_yy_1to2d_diff_crop = sigma_yy_1to2d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3  # convert to mN/m
sigma_yy_1to1d_diff_crop = sigma_yy_1to1d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3  # convert to mN/m
sigma_yy_2to1d_diff_crop = sigma_yy_2to1d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3  # convert to mN/m

# set up plot parameters
# *****************************************************************************

pixelsize = 0.864  # in µm
sigma_max = 1  # mN/m
sigma_min = -1  # mN/m

# create x- and y-axis for plotting maps
x_end = np.shape(sigma_xx_1to1d_diff_crop)[1]
y_end = np.shape(sigma_xx_1to1d_diff_crop)[0]
extent = [0, x_end * pixelsize, 0, y_end * pixelsize]

# create mesh for vectorplot
xq, yq = np.meshgrid(np.linspace(0, extent[1], x_end), np.linspace(0, extent[3], y_end))

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(3, 4))

im = plot_stressmaps(axes[0, 0], sigma_xx_1to2d_diff_crop, pixelsize, sigma_max, sigma_min, cmap="seismic")
plot_stressmaps(axes[1, 0], sigma_xx_1to1d_diff_crop, pixelsize, sigma_max, sigma_min, cmap="seismic")
plot_stressmaps(axes[2, 0], sigma_xx_2to1d_diff_crop, pixelsize, sigma_max, sigma_min, cmap="seismic")
plot_stressmaps(axes[0, 1], sigma_yy_1to2d_diff_crop, pixelsize, sigma_max, sigma_min, cmap="seismic")
plot_stressmaps(axes[1, 1], sigma_yy_1to1d_diff_crop, pixelsize, sigma_max, sigma_min, cmap="seismic")
plot_stressmaps(axes[2, 1], sigma_yy_2to1d_diff_crop, pixelsize, sigma_max, sigma_min, cmap="seismic")

# adjust space in between plots
plt.subplots_adjust(wspace=0, hspace=0)

# remove axes
for ax in axes.flat:
    ax.axis('off')
    aspectratio = 1.0
    ratio_default = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.set_aspect(ratio_default * aspectratio)

# draw pattern
draw_pattern_1to2(axes[0, 0])
draw_pattern_1to2(axes[0, 1])
draw_pattern_1to1(axes[1, 0])
draw_pattern_1to1(axes[1, 1])
draw_pattern_2to1(axes[2, 0])
draw_pattern_2to1(axes[2, 1])

# add colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist())
cbar.ax.set_title('mN/m')

# add title
# plt.suptitle('$\mathrm{\Delta \sigma _{avg. normal}(x,y)}$', y=0.98, x=0.5)
# plt.suptitle('$\mathrm{abc}$', y=0.98, x=0.5)
# plt.text(-20, 230, '$\mathrm{\Delta}$ mean stresses')
plt.text(-85, 215, '$\mathrm{\Delta \sigma _ {xx}}$', size=10)
plt.text(-15, 215, '$\mathrm{\Delta \sigma _ {yy}}$', size=10)

# add annotations
plt.text(0.38, 0.853, 'n=' + str(n_1to2d), transform=plt.figure(1).transFigure, color='black')
plt.text(0.38, 0.598, 'n=' + str(n_1to1d), transform=plt.figure(1).transFigure, color='black')
plt.text(0.38, 0.343, 'n=' + str(n_2to1d), transform=plt.figure(1).transFigure, color='black')

# save figure
fig.savefig(figfolder + 'D.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'D.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% prepare data for figure 5E

# set up global plot parameters
# ******************************************************************************************************************************************
x = np.linspace(-40, 40, 92)
x = x[::2]  # downsample data for nicer plotting
xticks = np.arange(-40, 40.1, 20)  # define where the major ticks are gonna be
xlabel = 'position [µm]'
ymin = -0.1
ymax = 0.5
yticks = np.arange(ymin, ymax + 0.001, 0.1)
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(2.5, 4))  # create figure and axes
plt.subplots_adjust(wspace=0.4, hspace=0.35)  # adjust space in between plots
# ******************************************************************************************************************************************

# # Set up plot parameters for first and second panel
# #######################################################################################################
color = colors_parent[0]
ylabel = None
ax = axes[0, 0]
title = '$\mathrm{\Delta \sigma _{xx}(x)}$ [mN/m]'
y = AR1to2d_halfstim["MSM_data"]["sigma_xx_x_profile_increase"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False, titleoffset=10)

ax = axes[0, 1]
title = '$\mathrm{\Delta \sigma _{yy}(x)}$ [mN/m]'
y = AR1to2d_halfstim["MSM_data"]["sigma_yy_x_profile_increase"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False, titleoffset=10)

# # Set up plot parameters for third and fourth panel
# #######################################################################################################
color = colors_parent[1]
ylabel = None
title = None

ax = axes[1, 0]
y = AR1to1d_halfstim["MSM_data"]["sigma_xx_x_profile_increase"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False, titleoffset=20)

ax = axes[1, 1]
y = AR1to1d_halfstim["MSM_data"]["sigma_yy_x_profile_increase"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False, titleoffset=20)

# # Set up plot parameters for fourth and sixth panel
# # #######################################################################################################
color = colors_parent[3]
ylabel = None
ax = axes[2, 0]
y = AR2to1d_halfstim["MSM_data"]["sigma_xx_x_profile_increase"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False, titleoffset=20)

ax = axes[2, 1]
y = AR2to1d_halfstim["MSM_data"]["sigma_yy_x_profile_increase"] * 1e3  # convert to nN
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False, titleoffset=20)

for ax in axes.flat:
    # add line at y=0 for visualisation
    ax.plot([x[0], x[-1]], [0, 0], linewidth=0.5, linestyle=":", color="grey")

    # add line at x=-10 to show opto stimulation border
    ax.axvline(x=-10, ymin=0.0, ymax=1, linewidth=0.5, color="cyan")

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
xx_stress_increase_ratio_sim_1to2 = []
xx_stress_increase_ratio_sim_1to1 = []
xx_stress_increase_ratio_sim_2to1 = []
yy_stress_increase_ratio_sim_1to2 = []
yy_stress_increase_ratio_sim_1to1 = []
yy_stress_increase_ratio_sim_2to1 = []

feedbacks = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for fb in feedbacks:
    xx_stress_increase_ratio_sim_1to2.append(AR1to2_FEM_simulation["feedback" + str(fb)]["xx_stress_increase_ratio"])
    xx_stress_increase_ratio_sim_1to1.append(AR1to1_FEM_simulation["feedback" + str(fb)]["xx_stress_increase_ratio"])
    xx_stress_increase_ratio_sim_2to1.append(AR2to1_FEM_simulation["feedback" + str(fb)]["xx_stress_increase_ratio"])
    yy_stress_increase_ratio_sim_1to2.append(AR1to2_FEM_simulation["feedback" + str(fb)]["yy_stress_increase_ratio"])
    yy_stress_increase_ratio_sim_1to1.append(AR1to1_FEM_simulation["feedback" + str(fb)]["yy_stress_increase_ratio"])
    yy_stress_increase_ratio_sim_2to1.append(AR2to1_FEM_simulation["feedback" + str(fb)]["yy_stress_increase_ratio"])

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(1.8, 4))
plt.subplots_adjust(hspace=0.4)  # adjust space in between plots

axes[0].plot(feedbacks, xx_stress_increase_ratio_sim_1to2, color=colors_parent[0])
axes[0].plot(feedbacks, xx_stress_increase_ratio_sim_1to1, color=colors_parent[1])
axes[0].plot(feedbacks, xx_stress_increase_ratio_sim_2to1, color=colors_parent[3])

axes[1].plot(feedbacks, yy_stress_increase_ratio_sim_1to2, color=colors_parent[0])
axes[1].plot(feedbacks, yy_stress_increase_ratio_sim_1to1, color=colors_parent[1])
axes[1].plot(feedbacks, yy_stress_increase_ratio_sim_2to1, color=colors_parent[3])

# add data points

x = find_x_position_of_point_on_array(feedbacks, xx_stress_increase_ratio_sim_1to2, xx_stress_increase_ratio_1to2)
AC_xx_1to2 = x

axes[0].errorbar(x, xx_stress_increase_ratio_1to2, yerr=xx_stress_increase_ratio_1to2_err, mfc="w", color=colors_parent[0],
                 marker="v", ms=5, linewidth=0.5, ls="none", markeredgewidth=0.5)

x = find_x_position_of_point_on_array(feedbacks, xx_stress_increase_ratio_sim_1to1, xx_stress_increase_ratio_1to1)
AC_xx_1to1 = x

axes[0].errorbar(x, xx_stress_increase_ratio_1to1, yerr=xx_stress_increase_ratio_1to1_err, mfc="w", color=colors_parent[1],
                 marker="s", ms=5, linewidth=0.5, ls="none", markeredgewidth=0.5)

# x = find_x_position_of_point_on_array(feedbacks, xx_stress_increase_ratio_sim_2to1, xx_stress_increase_ratio_2to1)
x = 1.0
AC_xx_2to1 = x

axes[0].errorbar(x, xx_stress_increase_ratio_2to1, yerr=xx_stress_increase_ratio_2to1_err, mfc="w", color=colors_parent[3],
                 marker="o", ms=5, linewidth=0.5, ls="none", markeredgewidth=0.5)
#

x = find_x_position_of_point_on_array(feedbacks, yy_stress_increase_ratio_sim_1to2, yy_stress_increase_ratio_1to2)
AC_yy_1to2 = x

axes[1].errorbar(x, yy_stress_increase_ratio_1to2, yerr=yy_stress_increase_ratio_1to2_err, mfc="w", color=colors_parent[0],
                 marker="v", ms=5, linewidth=0.5, ls="none", markeredgewidth=0.5)

x = find_x_position_of_point_on_array(feedbacks, yy_stress_increase_ratio_sim_1to1, yy_stress_increase_ratio_1to1)
AC_yy_1to1 = x

axes[1].errorbar(x, yy_stress_increase_ratio_1to1, yerr=yy_stress_increase_ratio_1to1_err, mfc="w", color=colors_parent[1],
                 marker="s", ms=5, linewidth=0.5, ls="none", markeredgewidth=0.5)

x = find_x_position_of_point_on_array(feedbacks, yy_stress_increase_ratio_sim_2to1, yy_stress_increase_ratio_2to1)
AC_yy_2to1 = x

axes[1].errorbar(x, yy_stress_increase_ratio_2to1, yerr=yy_stress_increase_ratio_2to1_err, mfc="w", color=colors_parent[3],
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
# %% load data for plotting
MP_1to2 = df["AIC_baseline"][keys == "AR1to2d"].mean()
MP_1to1 = df["AIC_baseline"][keys == "AR1to1d"].mean()
MP_2to1 = df["AIC_baseline"][keys == "AR2to1d"].mean()

SP_1to2 = df["actin_anisotropy_coefficient"][keys == "AR1to2d"].mean()
SP_1to1 = df["actin_anisotropy_coefficient"][keys == "AR1to1d"].mean()
SP_2to1 = df["actin_anisotropy_coefficient"][keys == "AR2to1d"].mean()

d = {'condition': ["1to2", "1to1", "2to1"], 'active coupling_x': np.array([AC_xx_1to2, AC_xx_1to1, AC_xx_2to1]),
     'active coupling_y': np.array([AC_yy_1to2, AC_yy_1to1, AC_yy_2to1]),
     'mechanical polarization': np.array([MP_1to2, MP_1to1, MP_2to1]),
     'structural polarization': np.array([SP_1to2, SP_1to1, SP_2to1])}

summarized_data = pd.DataFrame(data=d)
summarized_data.to_csv("summarized_data.csv", index=False)
colors = [colors_parent[0], colors_parent[1], colors_parent[3]]

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
