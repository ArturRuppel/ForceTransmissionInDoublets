# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 21:56:01 2021

@author: Artur Ruppel

"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib as mpl
import pickle
import seaborn as sns
import pandas as pd
from scipy.stats import normaltest, shapiro, pearsonr
from statannot import add_stat_annotation
import os
import pylustrator

# mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 8

# define some colors for the plots
colors_parent = ['#026473', '#E3CC69', '#77C8A6', '#D96248']
colors_parent_dark = ['#01353D', '#564910', '#235741', '#A93B23']


def test_if_gaussian(data1, data2, title):
    # test if data follows Gaussian distribution
    stat, p_n1 = normaltest(data1)
    stat, p_s1 = shapiro(data1)
    stat, p_n2 = normaltest(data2)
    stat, p_s2 = shapiro(data2)
    print('#############################################')
    # depending on the result of the Gaussian distribution test, perform either unpaired t-test or Mann-Whitney U test
    if (p_n1 > 0.05 and p_s1 > 0.05 and p_n2 > 0.05 and p_s2 > 0.05):
        gaussian = True
        print(title + ': Probably Gaussian.')
    else:
        gaussian = False
        print(title + ': Probably not Gaussian.')

    return gaussian


# %% load data for plotting
folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"

# AR1to1d_fullstim_long =   pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_long.dat", "rb"))
# AR1to1s_fullstim_long =   pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_long.dat", "rb"))
# AR1to1d_fullstim_short =  pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_short.dat", "rb"))
# AR1to1s_fullstim_short =  pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_short.dat", "rb"))
AR1to2d_halfstim = pickle.load(open(folder + "analysed_data/AR1to2d_halfstim.dat", "rb"))
AR1to1d_halfstim = pickle.load(open(folder + "analysed_data/AR1to1d_halfstim.dat", "rb"))
AR1to1s_halfstim = pickle.load(open(folder + "analysed_data/AR1to1s_halfstim.dat", "rb"))
AR2to1d_halfstim = pickle.load(open(folder + "analysed_data/AR2to1d_halfstim.dat", "rb"))

figfolder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/_Figure6/"
if not os.path.exists(figfolder):
    os.mkdir(figfolder)

# %% prepare dataframe for boxplots

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

            concatenated_data[key2] = np.concatenate((concatenated_data_1to2d[key2], concatenated_data_1to1d[key2],
                                                      concatenated_data_1to1s[key2], concatenated_data_2to1d[key2]))

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
concatenated_data['keys'] = keys

# Creates DataFrame
df = pd.DataFrame(concatenated_data)

# # convert to more convenient units for plotting
# df_plot_units = df # all units here are in SI units
# df_plot_units['Es_baseline'] *= 1e12 # convert to fJ
# df_plot_units['spreadingsize_baseline'] *= 1e12 # convert to µm²
# df_plot_units['sigma_xx_baseline'] *= 1e3 # convert to mN/m
# df_plot_units['sigma_yy_baseline'] *= 1e3 # convert to mN/m

# %% prepare dataframe for boxplots
n_1to2d = AR1to2d_halfstim['MSM_data']['RSI_xx_left'].shape[0]
n_1to1d = AR1to1d_halfstim['MSM_data']['RSI_xx_left'].shape[0]
n_1to1s = AR1to1s_halfstim['MSM_data']['RSI_xx_left'].shape[0]
n_2to1d = AR2to1d_halfstim['MSM_data']['RSI_xx_left'].shape[0]

RSI_data_1to2d = {}
RSI_data_1to1d = {}
RSI_data_1to1s = {}
RSI_data_2to1d = {}

RSI_data_1to2d['sigma'] = np.concatenate((AR1to2d_halfstim['MSM_data']['RSI_xx_left'],
                                          AR1to2d_halfstim['MSM_data']['RSI_xx_right'],
                                          AR1to2d_halfstim['MSM_data']['RSI_yy_left'],
                                          AR1to2d_halfstim['MSM_data']['RSI_yy_right']))
RSI_data_1to1d['sigma'] = np.concatenate((AR1to1d_halfstim['MSM_data']['RSI_xx_left'],
                                          AR1to1d_halfstim['MSM_data']['RSI_xx_right'],
                                          AR1to1d_halfstim['MSM_data']['RSI_yy_left'],
                                          AR1to1d_halfstim['MSM_data']['RSI_yy_right']))
RSI_data_1to1s['sigma'] = np.concatenate((AR1to1s_halfstim['MSM_data']['RSI_xx_left'],
                                          AR1to1s_halfstim['MSM_data']['RSI_xx_right'],
                                          AR1to1s_halfstim['MSM_data']['RSI_yy_left'],
                                          AR1to1s_halfstim['MSM_data']['RSI_yy_right']))
RSI_data_2to1d['sigma'] = np.concatenate((AR2to1d_halfstim['MSM_data']['RSI_xx_left'],
                                          AR2to1d_halfstim['MSM_data']['RSI_xx_right'],
                                          AR2to1d_halfstim['MSM_data']['RSI_yy_left'],
                                          AR2to1d_halfstim['MSM_data']['RSI_yy_right']))

keys1to2d = np.concatenate((['RSI_xx_left' for i in range(n_1to2d)], ['RSI_xx_right' for i in range(n_1to2d)],
                            ['RSI_yy_left' for i in range(n_1to2d)], ['RSI_yy_right' for i in range(n_1to2d)]))
keys1to1d = np.concatenate((['RSI_xx_left' for i in range(n_1to1d)], ['RSI_xx_right' for i in range(n_1to1d)],
                            ['RSI_yy_left' for i in range(n_1to1d)], ['RSI_yy_right' for i in range(n_1to1d)]))
keys1to1s = np.concatenate((['RSI_xx_left' for i in range(n_1to1s)], ['RSI_xx_right' for i in range(n_1to1s)],
                            ['RSI_yy_left' for i in range(n_1to1s)], ['RSI_yy_right' for i in range(n_1to1s)]))
keys2to1d = np.concatenate((['RSI_xx_left' for i in range(n_2to1d)], ['RSI_xx_right' for i in range(n_2to1d)],
                            ['RSI_yy_left' for i in range(n_2to1d)], ['RSI_yy_right' for i in range(n_2to1d)]))

RSI_data_1to2d['keys'] = keys1to2d
RSI_data_1to1d['keys'] = keys1to1d
RSI_data_1to1s['keys'] = keys1to1s
RSI_data_2to1d['keys'] = keys2to1d

df1to2d = pd.DataFrame(RSI_data_1to2d)
df1to1d = pd.DataFrame(RSI_data_1to1d)
df1to1s = pd.DataFrame(RSI_data_1to1s)
df2to1d = pd.DataFrame(RSI_data_2to1d)

# %% plot figure 6A, stress map differences

# prepare data first

# concatenate TFM maps from different experiments and calculate average maps over first 20 frames and all cells to get average maps
sigmaxx_1to2d_diff = np.nanmean(
    AR1to2d_halfstim["MSM_data"]["sigma_xx"][:, :, 33, :] - AR1to2d_halfstim["MSM_data"]["sigma_xx"][:, :, 20, :],
    axis=2)
sigmayy_1to2d_diff = np.nanmean(
    AR1to2d_halfstim["MSM_data"]["sigma_yy"][:, :, 33, :] - AR1to2d_halfstim["MSM_data"]["sigma_yy"][:, :, 20, :],
    axis=2)

sigmaxx_1to1d_diff = np.nanmean(
    AR1to1d_halfstim["MSM_data"]["sigma_xx"][:, :, 33, :] - AR1to1d_halfstim["MSM_data"]["sigma_xx"][:, :, 20, :],
    axis=2)
sigmayy_1to1d_diff = np.nanmean(
    AR1to1d_halfstim["MSM_data"]["sigma_yy"][:, :, 33, :] - AR1to1d_halfstim["MSM_data"]["sigma_yy"][:, :, 20, :],
    axis=2)

sigmaxx_1to1s_diff = np.nanmean(
    AR1to1s_halfstim["MSM_data"]["sigma_xx"][:, :, 33, :] - AR1to1s_halfstim["MSM_data"]["sigma_xx"][:, :, 20, :],
    axis=2)
sigmayy_1to1s_diff = np.nanmean(
    AR1to1s_halfstim["MSM_data"]["sigma_yy"][:, :, 33, :] - AR1to1s_halfstim["MSM_data"]["sigma_yy"][:, :, 20, :],
    axis=2)

sigmaxx_2to1d_diff = np.nanmean(
    AR2to1d_halfstim["MSM_data"]["sigma_xx"][:, :, 33, :] - AR2to1d_halfstim["MSM_data"]["sigma_xx"][:, :, 20, :],
    axis=2)
sigmayy_2to1d_diff = np.nanmean(
    AR2to1d_halfstim["MSM_data"]["sigma_yy"][:, :, 33, :] - AR2to1d_halfstim["MSM_data"]["sigma_yy"][:, :, 20, :],
    axis=2)

# crop maps 
crop_start = 2
crop_end = 90

sigmaxx_1to2d_diff_crop = sigmaxx_1to2d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3  # convert to mN/m
sigmayy_1to2d_diff_crop = sigmayy_1to2d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3
sigmaxx_1to1d_diff_crop = sigmaxx_1to1d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3  # convert to mN/m
sigmayy_1to1d_diff_crop = sigmayy_1to1d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3
sigmaxx_1to1s_diff_crop = sigmaxx_1to1s_diff[crop_start:crop_end, crop_start:crop_end] * 1e3
sigmayy_1to1s_diff_crop = sigmayy_1to1s_diff[crop_start:crop_end, crop_start:crop_end] * 1e3
sigmaxx_2to1d_diff_crop = sigmaxx_2to1d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3  # convert to mN/m
sigmayy_2to1d_diff_crop = sigmayy_2to1d_diff[crop_start:crop_end, crop_start:crop_end] * 1e3

# set up plot parameters
# *****************************************************************************

pixelsize = 0.864  # in µm
sigma_max = 1  # kPa
sigma_min = -1  # kPa

# create x- and y-axis for plotting maps 
x_end = np.shape(sigmaxx_1to1d_diff_crop)[1]
y_end = np.shape(sigmaxx_1to1d_diff_crop)[0]
extent = [0, x_end * pixelsize, 0, y_end * pixelsize]

# create mesh for vectorplot    
xq, yq = np.meshgrid(np.linspace(0, extent[1], x_end), np.linspace(0, extent[3], y_end))

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(3, 5))

im = axes[0, 0].imshow(sigmaxx_1to2d_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent,
                       vmin=sigma_min, vmax=sigma_max, aspect='auto')
axes[0, 1].imshow(sigmayy_1to2d_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent,
                  vmin=sigma_min, vmax=sigma_max, aspect='auto')

axes[1, 0].imshow(sigmaxx_1to1d_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent,
                  vmin=sigma_min, vmax=sigma_max, aspect='auto')
axes[1, 1].imshow(sigmayy_1to1d_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent,
                  vmin=sigma_min, vmax=sigma_max, aspect='auto')

axes[2, 0].imshow(sigmaxx_1to1s_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent,
                  vmin=sigma_min, vmax=sigma_max, aspect='auto')
axes[2, 1].imshow(sigmayy_1to1s_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent,
                  vmin=sigma_min, vmax=sigma_max, aspect='auto')

axes[3, 0].imshow(sigmaxx_2to1d_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent,
                  vmin=sigma_min, vmax=sigma_max, aspect='auto')
axes[3, 1].imshow(sigmayy_2to1d_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent,
                  vmin=sigma_min, vmax=sigma_max, aspect='auto')

# adjust space in between plots
plt.subplots_adjust(wspace=0, hspace=0)

# axes[0,0].set_xlabel("lol")
# # add annotations
# plt.text(-50,120,'sigmaxx',color = 'k')
# plt.text(20,120,'sigmayy',color = 'k')
# plt.text(-40.5,119,'n=1',color = 'white')
# plt.text(23,55.5,'n=101',color = 'white')
# plt.text(23.5,119,'n=66',color = 'white')

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
# plt.suptitle('$\mathrm{\Delta}$ Stresses, local activation',y=0.94, x=0.44)

# make some annotations
# #% start: automatic generated code from pylustrator
# plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}

# plt.figure(1).texts[0].set_position([0.440000, 0.98])
# plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[1].new
# plt.figure(1).texts[1].set_position([0.239612, 0.840355])
# plt.figure(1).texts[1].set_text("n=40")
# plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[2].new
# plt.figure(1).texts[2].set_position([0.2, 0.889135])
# plt.figure(1).texts[2].set_text("xx-Stress")
# plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[3].new
# plt.figure(1).texts[3].set_position([0.239612, 0.471175])
# plt.figure(1).texts[3].set_text("n=16")
# plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[4].new
# plt.figure(1).texts[4].set_position([0.5, 0.889135])
# plt.figure(1).texts[4].set_text("yy-Stress")
# plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[5].new
# plt.figure(1).texts[5].set_position([0.553555, 0.840355])
# plt.figure(1).texts[5].set_text("n=28")
# plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[6].new
# plt.figure(1).texts[6].set_position([0.555402, 0.471175])
# plt.figure(1).texts[6].set_text("n=36")
# #% end: automatic generated code from pylustrator

# save figure
fig.savefig(figfolder + 'A.png', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 6B, Normalized stress over time halfstim

# define plot parameters
fig = plt.figure(2, figsize=(5, 6))  # figuresize in inches
gs = gridspec.GridSpec(4, 3)  # sets up subplotgrid rows by columns
gs.update(wspace=0.3, hspace=0.5)  # adjusts space in between the boxes in the grid
linewidth_bp = 0.5  # linewidth of boxplot borders
width = 0.7  # width of boxplots
dotsize = 1.5  # size of datapoints in swarmplot
linewidth_sw = 0.3  # linewidth of boxplot borders
alpha_sw = 1  # transparency of dots in swarmplot
alpha_bp = 0.8  # transparency of boxplots
ylabeloffset = 1  # adjusts distance of ylabel to the plot
xlabeloffset = 1  # adjusts distance of ylabel to the plot
titleoffset = 5  # adjusts distance of title to the plot
optolinewidth = 0.1  # adjusts the linewidth of the annotations that represent the optogenetic activation

##############################################################################
# Generate first panel
##############################################################################

ymin = -2
ymax = 2
xticks = np.arange(0, 61, 20)
yticks = np.arange(-2, 2, 0.5)

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[0, 0])

# set plot variables
x = np.arange(60)
y = AR1to2d_halfstim["MSM_data"]["normsigma_xx_left"]
x = x[::2]  # downsample data for nicer plotting
y = y[::2, :]
y_mean = np.nanmean(y, axis=1)
y_std = np.nanstd(y, axis=1)
y_sem = y_std / np.sqrt(np.shape(y)[1])

# create first plot
fig_ax.errorbar(x, y_mean, yerr=y_sem, mfc='w', color=colors_parent[0], marker='o', ms=2, linewidth=0.5, ls='none',
                markeredgewidth=0.5)
# set plot variables
x = np.arange(60)
y = AR1to2d_halfstim["MSM_data"]["normsigma_xx_right"]
x = x[::2]  # downsample data for nicer plotting
y = y[::2, :]
y_mean = np.nanmean(y, axis=1)
y_std = np.nanstd(y, axis=1)
y_sem = y_std / np.sqrt(np.shape(y)[1])

# create second plot
fig_ax.errorbar(x, y_mean, yerr=y_sem, mfc='w', color=colors_parent_dark[0], marker='o', ms=2, linewidth=0.5, ls='none',
                markeredgewidth=0.5)

# set labels
fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
fig_ax.set_ylabel(ylabel='1to2d', labelpad=ylabeloffset)
# fig_ax.set_title(label='relative $\mathrm{E_s}$', pad=titleoffset)
fig_ax.set_title(label='xx-Stress', pad=titleoffset)

# add anotations for opto pulses
for i in np.arange(10):
    plt.axline((20 + i, ymin), (20 + i, ymax), linewidth=optolinewidth, color="cyan")


plt.xticks(xticks)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in', which='minor', length=3, bottom=True, top=False, left=True, right=True)
plt.tick_params(direction='in', which='major', length=6, bottom=True, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
# Generate second panel
##############################################################################

ymin = -2
ymax = 2
xticks = np.arange(0, 61, 20)
yticks = np.arange(-2, 2, 0.5)

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[0, 1])

# set plot variables
x = np.arange(60)
y = AR1to2d_halfstim["MSM_data"]["normsigma_yy_left"]
x = x[::2]  # downsample data for nicer plotting
y = y[::2, :]
y_mean = np.nanmean(y, axis=1)
y_std = np.nanstd(y, axis=1)
y_sem = y_std / np.sqrt(np.shape(y)[1])

# create first plot
fig_ax.errorbar(x, y_mean, yerr=y_sem, mfc='w', color=colors_parent[0], marker='o', ms=2, linewidth=0.5, ls='none',
                markeredgewidth=0.5)
# set plot variables
x = np.arange(60)
y = AR1to2d_halfstim["MSM_data"]["normsigma_yy_right"]
x = x[::2]  # downsample data for nicer plotting
y = y[::2, :]
y_mean = np.nanmean(y, axis=1)
y_std = np.nanstd(y, axis=1)
y_sem = y_std / np.sqrt(np.shape(y)[1])

# create second plot
fig_ax.errorbar(x, y_mean, yerr=y_sem, mfc='w', color=colors_parent_dark[0], marker='o', ms=2, linewidth=0.5, ls='none',
                markeredgewidth=0.5)

# set labels
fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
# fig_ax.set_ylabel(ylabel='doublet', labelpad=ylabeloffset)
# fig_ax.set_title(label='normative $\mathrm{E_s}$', pad=titleoffset)
fig_ax.set_title(label='yy-Stress', pad=titleoffset)
fig_ax.set()

# add anotations for opto pulses
for i in np.arange(10):
    plt.axline((20 + i, ymin), (20 + i, ymax), linewidth=optolinewidth, color="cyan")

# Define where you want ticks
xticks = np.arange(0, 61, 20)
yticks = np.arange(0.8, 1.31, 0.1)


# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in', which='minor', length=3, bottom=True, top=False, left=True, right=True)
plt.tick_params(direction='in', which='major', length=6, bottom=True, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
# Generate third panel
##############################################################################

ymin = -2
ymax = 2
xticks = np.arange(0, 61, 20)
yticks = np.arange(-2, 2, 0.5)

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[1, 0])

# set plot variables
x = np.arange(60)
y = AR1to1d_halfstim["MSM_data"]["normsigma_xx_left"]
x = x[::2]  # downsample data for nicer plotting
y = y[::2, :]
y_mean = np.nanmean(y, axis=1)
y_std = np.nanstd(y, axis=1)
y_sem = y_std / np.sqrt(np.shape(y)[1])

# create first plot
fig_ax.errorbar(x, y_mean, yerr=y_sem, mfc='w', color=colors_parent[1], marker='o', ms=2, linewidth=0.5, ls='none',
                markeredgewidth=0.5)
# set plot variables
x = np.arange(60)
y = AR1to1d_halfstim["MSM_data"]["normsigma_xx_right"]
x = x[::2]  # downsample data for nicer plotting
y = y[::2, :]
y_mean = np.nanmean(y, axis=1)
y_std = np.nanstd(y, axis=1)
y_sem = y_std / np.sqrt(np.shape(y)[1])

# create second plot
fig_ax.errorbar(x, y_mean, yerr=y_sem, mfc='w', color=colors_parent_dark[1], marker='o', ms=2, linewidth=0.5, ls='none',
                markeredgewidth=0.5)

# set labels
fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
fig_ax.set_ylabel(ylabel='1to1d', labelpad=ylabeloffset)
# fig_ax.set_title(label='normative $\mathrm{E_s}$', pad=titleoffset)
# fig_ax.set_title(label='xx-Stress', pad=titleoffset)

# add anotations for opto pulses
for i in np.arange(10):
    plt.axline((20 + i, ymin), (20 + i, ymax), linewidth=optolinewidth, color="cyan")


plt.xticks(xticks)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in', which='minor', length=3, bottom=True, top=False, left=True, right=True)
plt.tick_params(direction='in', which='major', length=6, bottom=True, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
# Generate fourth panel
##############################################################################

ymin = -2
ymax = 2
xticks = np.arange(0, 61, 20)
yticks = np.arange(-2, 2, 0.5)

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[1, 1])

# set plot variables
x = np.arange(60)
y = AR1to1d_halfstim["MSM_data"]["normsigma_yy_left"]
x = x[::2]  # downsample data for nicer plotting
y = y[::2, :]
y_mean = np.nanmean(y, axis=1)
y_std = np.nanstd(y, axis=1)
y_sem = y_std / np.sqrt(np.shape(y)[1])

# create first plot
fig_ax.errorbar(x, y_mean, yerr=y_sem, mfc='w', color=colors_parent[1], marker='o', ms=2, linewidth=0.5, ls='none',
                markeredgewidth=0.5)
# set plot variables
x = np.arange(60)
y = AR1to1d_halfstim["MSM_data"]["normsigma_yy_right"]
x = x[::2]  # downsample data for nicer plotting
y = y[::2, :]
y_mean = np.nanmean(y, axis=1)
y_std = np.nanstd(y, axis=1)
y_sem = y_std / np.sqrt(np.shape(y)[1])

# create second plot
fig_ax.errorbar(x, y_mean, yerr=y_sem, mfc='w', color=colors_parent_dark[1], marker='o', ms=2, linewidth=0.5, ls='none',
                markeredgewidth=0.5)

# set labels
fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
# fig_ax.set_ylabel(ylabel='doublet', labelpad=ylabeloffset)
# fig_ax.set_title(label='normative $\mathrm{E_s}$', pad=titleoffset)
# fig_ax.set_title(label='yy-Stress', pad=titleoffset)
fig_ax.set()

# add anotations for opto pulses
for i in np.arange(10):
    plt.axline((20 + i, ymin), (20 + i, ymax), linewidth=optolinewidth, color="cyan")


plt.xticks(xticks)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in', which='minor', length=3, bottom=True, top=False, left=True, right=True)
plt.tick_params(direction='in', which='major', length=6, bottom=True, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
# Generate fifth panel
##############################################################################

ymin = -2
ymax = 2
xticks = np.arange(0, 61, 20)
yticks = np.arange(-2, 2, 0.5)

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[2, 0])

# set plot variables
x = np.arange(60)
y = AR1to1s_halfstim["MSM_data"]["normsigma_xx_left"]
x = x[::2]  # downsample data for nicer plotting
y = y[::2, :]
y_mean = np.nanmean(y, axis=1)
y_std = np.nanstd(y, axis=1)
y_sem = y_std / np.sqrt(np.shape(y)[1])

# create first plot
fig_ax.errorbar(x, y_mean, yerr=y_sem, mfc='w', color=colors_parent[2], marker='o', ms=2, linewidth=0.5, ls='none',
                markeredgewidth=0.5)
# set plot variables
x = np.arange(60)
y = AR1to1s_halfstim["MSM_data"]["normsigma_xx_right"]
x = x[::2]  # downsample data for nicer plotting
y = y[::2, :]
y_mean = np.nanmean(y, axis=1)
y_std = np.nanstd(y, axis=1)
y_sem = y_std / np.sqrt(np.shape(y)[1])

# create second plot
fig_ax.errorbar(x, y_mean, yerr=y_sem, mfc='w', color=colors_parent_dark[2], marker='o', ms=2, linewidth=0.5, ls='none',
                markeredgewidth=0.5)

# set labels
fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
fig_ax.set_ylabel(ylabel='1to1s', labelpad=ylabeloffset)
# fig_ax.set_title(label='normative $\mathrm{E_s}$', pad=titleoffset)
fig_ax.set()

# add anotations for opto pulses
for i in np.arange(10):
    plt.axline((20 + i, ymin), (20 + i, ymax), linewidth=optolinewidth, color="cyan")

# Define where you want ticks
xticks = np.arange(0, 61, 20)
yticks = np.arange(0.8, 1.31, 0.1)


# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in', which='minor', length=3, bottom=True, top=False, left=True, right=True)
plt.tick_params(direction='in', which='major', length=6, bottom=True, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
# Generate sixth panel
##############################################################################

ymin = -2
ymax = 2
xticks = np.arange(0, 61, 20)
yticks = np.arange(-2, 2, 0.5)

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[2, 1])

# set plot variables
x = np.arange(60)
y = AR1to1s_halfstim["MSM_data"]["normsigma_yy_left"]
x = x[::2]  # downsample data for nicer plotting
y = y[::2, :]
y_mean = np.nanmean(y, axis=1)
y_std = np.nanstd(y, axis=1)
y_sem = y_std / np.sqrt(np.shape(y)[1])

# create first plot
fig_ax.errorbar(x, y_mean, yerr=y_sem, mfc='w', color=colors_parent[2], marker='o', ms=2, linewidth=0.5, ls='none',
                markeredgewidth=0.5)
# set plot variables
x = np.arange(60)
y = AR1to1s_halfstim["MSM_data"]["normsigma_yy_right"]
x = x[::2]  # downsample data for nicer plotting
y = y[::2, :]
y_mean = np.nanmean(y, axis=1)
y_std = np.nanstd(y, axis=1)
y_sem = y_std / np.sqrt(np.shape(y)[1])

# create second plot
fig_ax.errorbar(x, y_mean, yerr=y_sem, mfc='w', color=colors_parent_dark[2], marker='o', ms=2, linewidth=0.5, ls='none',
                markeredgewidth=0.5)

# set labels
fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
# fig_ax.set_ylabel(ylabel='singlet', labelpad=ylabeloffset)
# fig_ax.set_title(label='normative $\mathrm{E_s}$', pad=titleoffset)
fig_ax.set()

# add anotations for opto pulses
for i in np.arange(10):
    plt.axline((20 + i, ymin), (20 + i, ymax), linewidth=optolinewidth, color="cyan")

# Define where you want ticks
xticks = np.arange(0, 61, 20)
yticks = np.arange(0.8, 1.31, 0.1)


# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in', which='minor', length=3, bottom=True, top=False, left=True, right=True)
plt.tick_params(direction='in', which='major', length=6, bottom=True, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
# Generate seventh panel
##############################################################################

ymin = -2
ymax = 2
xticks = np.arange(0, 61, 20)
yticks = np.arange(-2, 2, 0.5)

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[3, 0])

# set plot variables
x = np.arange(60)
y = AR2to1d_halfstim["MSM_data"]["normsigma_xx_left"]
x = x[::2]  # downsample data for nicer plotting
y = y[::2, :]
y_mean = np.nanmean(y, axis=1)
y_std = np.nanstd(y, axis=1)
y_sem = y_std / np.sqrt(np.shape(y)[1])

# create first plot
fig_ax.errorbar(x, y_mean, yerr=y_sem, mfc='w', color=colors_parent[3], marker='o', ms=2, linewidth=0.5, ls='none',
                markeredgewidth=0.5)
# set plot variables
x = np.arange(60)
y = AR2to1d_halfstim["MSM_data"]["normsigma_xx_right"]
x = x[::2]  # downsample data for nicer plotting
y = y[::2, :]
y_mean = np.nanmean(y, axis=1)
y_std = np.nanstd(y, axis=1)
y_sem = y_std / np.sqrt(np.shape(y)[1])

# create second plot
fig_ax.errorbar(x, y_mean, yerr=y_sem, mfc='w', color=colors_parent_dark[3], marker='o', ms=2, linewidth=0.5, ls='none',
                markeredgewidth=0.5)

# set labels
fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
fig_ax.set_ylabel(ylabel='2to1d', labelpad=ylabeloffset)
# fig_ax.set_title(label='normative $\mathrm{E_s}$', pad=titleoffset)
# fig_ax.set_title(label='xx-Stress', pad=titleoffset)

# add anotations for opto pulses
for i in np.arange(10):
    plt.axline((20 + i, ymin), (20 + i, ymax), linewidth=optolinewidth, color="cyan")


plt.xticks(xticks)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in', which='minor', length=3, bottom=True, top=False, left=True, right=True)
plt.tick_params(direction='in', which='major', length=6, bottom=True, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
# Generate eighth panel
##############################################################################

ymin = -2
ymax = 2
xticks = np.arange(0, 61, 20)
yticks = np.arange(-2, 2, 0.5)

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[3, 1])

# set plot variables
x = np.arange(60)
y = AR2to1d_halfstim["MSM_data"]["normsigma_yy_left"]
x = x[::2]  # downsample data for nicer plotting
y = y[::2, :]
y_mean = np.nanmean(y, axis=1)
y_std = np.nanstd(y, axis=1)
y_sem = y_std / np.sqrt(np.shape(y)[1])

# create first plot
fig_ax.errorbar(x, y_mean, yerr=y_sem, mfc='w', color=colors_parent[3], marker='o', ms=2, linewidth=0.5, ls='none',
                markeredgewidth=0.5)
# set plot variables
x = np.arange(60)
y = AR2to1d_halfstim["MSM_data"]["normsigma_yy_right"]
x = x[::2]  # downsample data for nicer plotting
y = y[::2, :]
y_mean = np.nanmean(y, axis=1)
y_std = np.nanstd(y, axis=1)
y_sem = y_std / np.sqrt(np.shape(y)[1])

# create second plot
fig_ax.errorbar(x, y_mean, yerr=y_sem, mfc='w', color=colors_parent_dark[3], marker='o', ms=2, linewidth=0.5, ls='none',
                markeredgewidth=0.5)

# set labels
fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
# fig_ax.set_ylabel(ylabel='doublet', labelpad=ylabeloffset)
# fig_ax.set_title(label='normative $\mathrm{E_s}$', pad=titleoffset)
# fig_ax.set_title(label='yy-Stress', pad=titleoffset)
fig_ax.set()

# add anotations for opto pulses
for i in np.arange(10):
    plt.axline((20 + i, ymin), (20 + i, ymax), linewidth=optolinewidth, color="cyan")


plt.xticks(xticks)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in', which='minor', length=3, bottom=True, top=False, left=True, right=True)
plt.tick_params(direction='in', which='major', length=6, bottom=True, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
# Generate ninth panel
##############################################################################
colors = [colors_parent[0], colors_parent_dark[0], colors_parent[0], colors_parent_dark[0]];  # defines colors
sns.set_palette(sns.color_palette(colors))  # sets colors
ylabeloffset = -1

# extract data from dataframe to test if their distribution is gaussian
# data_1to1d = df_fs[df_fs["keys"]=="AR1to1d_fs"]["RSI_xx"].to_numpy()
# data_1to1s = df_fs[df_fs["keys"]=="AR1to1s_fs"]["RSI_xx"].to_numpy()
# if test_if_gaussian(data_1to1d,data_1to1s,'REI'):
#     test = 't-test_ind'
# else:
test = 'Mann-Whitney'

ymin = -0.2
ymax = 0.6
stat_annotation_offset = 0

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[0, 2])

# create box- and swarmplots
sns.swarmplot(x='keys', y='sigma', data=df1to2d, ax=fig_ax, alpha=alpha_sw, linewidth=linewidth_sw, zorder=0,
              size=dotsize)
bp = sns.boxplot(x='keys', y='sigma', data=df1to2d, ax=fig_ax, linewidth=linewidth_bp, notch=True, showfliers=False,
                 width=width)

# order = ['AR1to1d_fs', 'AR1to1s_fs']
# add_stat_annotation(bp, data=df_fs, x=x, y=y, order=order, box_pairs=[('AR1to1d_fs', 'AR1to1s_fs')], 
#                     line_offset_to_box=stat_annotation_offset, test=test, text_format='star', loc='inside', verbose=2)

# make boxplots transparent
for patch in bp.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, alpha_bp))

plt.setp(bp.artists, edgecolor='k')
plt.setp(bp.lines, color='k')

# set labels
fig_ax.set_xticklabels(
    ['left \n         $\mathrm{\sigma _ {xx}}$', 'right', 'left \n         $\mathrm{\sigma _ {yy}}$', 'right'])
fig_ax.set_xlabel(xlabel=None)
fig_ax.set_ylabel(ylabel=None, labelpad=ylabeloffset)
fig_ax.set_title(label='normative stress \n increase', pad=titleoffset)
fig_ax.set()

# Define where you want ticks
yticks = np.arange(-0.2, 0.61, 0.2)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in', which='minor', length=3, bottom=False, top=False, left=True, right=True)
plt.tick_params(direction='in', which='major', length=6, bottom=False, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

# ##############################################################################
# #Generate tenth panel
# ##############################################################################
colors = [colors_parent[1], colors_parent_dark[1], colors_parent[1], colors_parent_dark[1]];  # defines colors
sns.set_palette(sns.color_palette(colors))  # sets colors
ylabeloffset = -1

# extract data from dataframe to test if their distribution is gaussian
# data_1to1d = df_fs[df_fs["keys"]=="AR1to1d_fs"]["RSI_xx"].to_numpy()
# data_1to1s = df_fs[df_fs["keys"]=="AR1to1s_fs"]["RSI_xx"].to_numpy()
# if test_if_gaussian(data_1to1d,data_1to1s,'REI'):
#     test = 't-test_ind'
# else:
test = 'Mann-Whitney'

ymin = -0.2
ymax = 0.6
stat_annotation_offset = 0

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[1, 2])

# create box- and swarmplots
sns.swarmplot(x='keys', y='sigma', data=df1to1d, ax=fig_ax, alpha=alpha_sw, linewidth=linewidth_sw, zorder=0,
              size=dotsize)
bp = sns.boxplot(x='keys', y='sigma', data=df1to1d, ax=fig_ax, linewidth=linewidth_bp, notch=True, showfliers=False,
                 width=width)

# order = ['AR1to1d_fs', 'AR1to1s_fs']
# add_stat_annotation(bp, data=df_fs, x=x, y=y, order=order, box_pairs=[('AR1to1d_fs', 'AR1to1s_fs')], 
#                     line_offset_to_box=stat_annotation_offset, test=test, text_format='star', loc='inside', verbose=2)

# make boxplots transparent
for patch in bp.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, alpha_bp))

plt.setp(bp.artists, edgecolor='k')
plt.setp(bp.lines, color='k')

# set labels
fig_ax.set_xticklabels(
    ['left \n         $\mathrm{\sigma _ {xx}}$', 'right', 'left \n         $\mathrm{\sigma _ {yy}}$', 'right'])
fig_ax.set_xlabel(xlabel=None)
fig_ax.set_ylabel(ylabel=None, labelpad=ylabeloffset)
# fig_ax.set_title(label='Relative stress \n increase', pad=titleoffset)
fig_ax.set()

# Define where you want ticks
yticks = np.arange(-0.2, 0.61, 0.2)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in', which='minor', length=3, bottom=False, top=False, left=True, right=True)
plt.tick_params(direction='in', which='major', length=6, bottom=False, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

# ##############################################################################
# #Generate eleventh panel
# ##############################################################################
colors = [colors_parent[2], colors_parent_dark[2], colors_parent[2], colors_parent_dark[2]];  # defines colors
sns.set_palette(sns.color_palette(colors))  # sets colors
ylabeloffset = -1

# extract data from dataframe to test if their distribution is gaussian
# data_1to1d = df_fs[df_fs["keys"]=="AR1to1d_fs"]["RSI_xx"].to_numpy()
# data_1to1s = df_fs[df_fs["keys"]=="AR1to1s_fs"]["RSI_xx"].to_numpy()
# if test_if_gaussian(data_1to1d,data_1to1s,'REI'):
#     test = 't-test_ind'
# else:
test = 'Mann-Whitney'

ymin = -0.2
ymax = 0.6
stat_annotation_offset = 0

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[2, 2])

# create box- and swarmplots
sns.swarmplot(x='keys', y='sigma', data=df1to1s, ax=fig_ax, alpha=alpha_sw, linewidth=linewidth_sw, zorder=0,
              size=dotsize)
bp = sns.boxplot(x='keys', y='sigma', data=df1to1s, ax=fig_ax, linewidth=linewidth_bp, notch=True, showfliers=False,
                 width=width)

# order = ['AR1to1d_fs', 'AR1to1s_fs']
# add_stat_annotation(bp, data=df_fs, x=x, y=y, order=order, box_pairs=[('AR1to1d_fs', 'AR1to1s_fs')], 
#                     line_offset_to_box=stat_annotation_offset, test=test, text_format='star', loc='inside', verbose=2)

# make boxplots transparent
for patch in bp.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, alpha_bp))

plt.setp(bp.artists, edgecolor='k')
plt.setp(bp.lines, color='k')

# set labels
fig_ax.set_xticklabels(
    ['left \n         $\mathrm{\sigma _ {xx}}$', 'right', 'left \n         $\mathrm{\sigma _ {yy}}$', 'right'])
fig_ax.set_xlabel(xlabel=None)
fig_ax.set_ylabel(ylabel=None, labelpad=ylabeloffset)
# fig_ax.set_title(label='Relative stress \n increase', pad=titleoffset)
fig_ax.set()

# Define where you want ticks
yticks = np.arange(-0.2, 0.61, 0.2)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in', which='minor', length=3, bottom=False, top=False, left=True, right=True)
plt.tick_params(direction='in', which='major', length=6, bottom=False, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

# ##############################################################################
# #Generate twelveth panel
# ##############################################################################
colors = [colors_parent[3], colors_parent_dark[3], colors_parent[3], colors_parent_dark[3]];  # defines colors
sns.set_palette(sns.color_palette(colors))  # sets colors
ylabeloffset = -1

# extract data from dataframe to test if their distribution is gaussian
# data_1to1d = df_fs[df_fs["keys"]=="AR1to1d_fs"]["RSI_xx"].to_numpy()
# data_1to1s = df_fs[df_fs["keys"]=="AR1to1s_fs"]["RSI_xx"].to_numpy()
# if test_if_gaussian(data_1to1d,data_1to1s,'REI'):
#     test = 't-test_ind'
# else:
test = 'Mann-Whitney'

ymin = -0.2
ymax = 0.6
stat_annotation_offset = 0

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[3, 2])

# create box- and swarmplots
sns.swarmplot(x='keys', y='sigma', data=df2to1d, ax=fig_ax, alpha=alpha_sw, linewidth=linewidth_sw, zorder=0,
              size=dotsize)
bp = sns.boxplot(x='keys', y='sigma', data=df2to1d, ax=fig_ax, linewidth=linewidth_bp, notch=True, showfliers=False,
                 width=width)

# order = ['AR1to1d_fs', 'AR1to1s_fs']
# add_stat_annotation(bp, data=df_fs, x=x, y=y, order=order, box_pairs=[('AR1to1d_fs', 'AR1to1s_fs')], 
#                     line_offset_to_box=stat_annotation_offset, test=test, text_format='star', loc='inside', verbose=2)

# make boxplots transparent
for patch in bp.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, alpha_bp))

plt.setp(bp.artists, edgecolor='k')
plt.setp(bp.lines, color='k')

# set labels
fig_ax.set_xticklabels(
    ['left \n         $\mathrm{\sigma _ {xx}}$', 'right', 'left \n         $\mathrm{\sigma _ {yy}}$', 'right'])
fig_ax.set_xlabel(xlabel=None)
fig_ax.set_ylabel(ylabel=None, labelpad=ylabeloffset)
# fig_ax.set_title(label='Relative stress \n increase', pad=titleoffset)
fig_ax.set()

# Define where you want ticks
yticks = np.arange(-0.2, 0.61, 0.2)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in', which='minor', length=3, bottom=False, top=False, left=True, right=True)
plt.tick_params(direction='in', which='major', length=6, bottom=False, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

plt.savefig(figfolder + 'B.png', dpi=300, bbox_inches="tight")

plt.show()
# %% plot figure 6C, Correlation plots

# set plot parameters
ylabeloffset = -5
colors = colors_parent;  # defines colors for scatterplot
colors_regplot = ['#000000', '#000000', '#000000', '#000000'];  # defines colors for linear regression plot

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4, 1.8))

############# first panel ####################
sns.set_palette(sns.color_palette(colors))
sns.scatterplot(data=df, x='AIC_baseline', y='RSI_xx_left', hue='keys', ax=axes[0])
sns.set_palette(sns.color_palette(colors_regplot))  # sets colors
sns.regplot(data=df, x='AIC_baseline', y='RSI_xx_left', scatter=False, ax=axes[0])

# set labels
axes[0].set_xlabel(xlabel='Anisotropy coefficient')
axes[0].set_ylabel(ylabel='$\mathrm{RSI_{xx, left}}$', labelpad=ylabeloffset)

# remove legend
axes[0].get_legend().remove()

# set limits
axes[0].set_ylim(ymin=-0.2, ymax=0.2)

# Define where you want ticks
plt.sca(axes[0])
yticks = np.arange(-0.2, 0.21, 0.1)
plt.yticks(yticks)
xticks = np.arange(-1, 1.1, 0.5)
plt.xticks(xticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in', which='minor', length=3, bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in', which='major', length=6, bottom=True, top=True, left=True, right=True)

corr, p = pearsonr(df['AIC_baseline'], df['RSI_xx_left'])
corr = np.round(corr, decimals=3)
p = np.round(p, decimals=3)

plt.text(-0.9, -0.14, 'Pearson R = ' + str(corr))
plt.text(-0.9, -0.17, 'p = ' + str(p))

############# second panel ####################
sns.set_palette(sns.color_palette(colors))
sns.scatterplot(data=df, x='AIC_baseline', y='RSI_yy_left', hue='keys', ax=axes[1])
sns.set_palette(sns.color_palette(colors_regplot))  # sets colors
sns.regplot(data=df, x='AIC_baseline', y='RSI_yy_left', scatter=False, ax=axes[1])

# set labels
axes[1].set_xlabel(xlabel='Anisotropy coefficient')
axes[1].set_ylabel(ylabel='$\mathrm{RSI_{yy, left}}$', labelpad=ylabeloffset)

# remove legend
axes[1].get_legend().remove()

# set limits
axes[1].set_ylim(ymin=-0.2, ymax=0.2)

# Define where you want ticks
plt.sca(axes[1])
yticks = np.arange(-0.2, 0.21, 0.1)
plt.yticks(yticks)
xticks = np.arange(-1, 1.1, 0.5)
plt.xticks(xticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in', which='minor', length=3, bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in', which='major', length=6, bottom=True, top=True, left=True, right=True)

corr, p = pearsonr(df['AIC_baseline'], df['RSI_yy_left'])
corr = np.round(corr, decimals=3)
p = np.round(p, decimals=3)

plt.text(-0.9, -0.14, 'Pearson R = ' + str(corr))
plt.text(-0.9, -0.17, 'p = ' + str(p))

plt.subplots_adjust(wspace=0.4, hspace=0.4)

plt.savefig(figfolder + 'C.png', dpi=300, bbox_inches="tight")
plt.show()
# %% plot figure 6D, Correlation plots

# set plot parameters
ylabeloffset = -5
colors = colors_parent;  # defines colors for scatterplot
colors_regplot = ['#000000', '#000000', '#000000', '#000000'];  # defines colors for linear regression plot

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4, 1.8))

############# first panel ####################
sns.set_palette(sns.color_palette(colors))
sns.scatterplot(data=df, x='AIC_baseline', y='RSI_xx_right', hue='keys', ax=axes[0])
sns.set_palette(sns.color_palette(colors_regplot))  # sets colors
sns.regplot(data=df, x='AIC_baseline', y='RSI_xx_right', scatter=False, ax=axes[0])

# set labels
axes[0].set_xlabel(xlabel='Anisotropy coefficient')
axes[0].set_ylabel(ylabel='$\mathrm{RSI_{xx, right}}$', labelpad=ylabeloffset)

# remove legend
axes[0].get_legend().remove()

# set limits
axes[0].set_ylim(ymin=-0.2, ymax=0.2)

# Define where you want ticks
plt.sca(axes[0])
yticks = np.arange(-0.2, 0.21, 0.1)
plt.yticks(yticks)
xticks = np.arange(-1, 1.1, 0.5)
plt.xticks(xticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in', which='minor', length=3, bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in', which='major', length=6, bottom=True, top=True, left=True, right=True)

corr, p = pearsonr(df['AIC_baseline'], df['RSI_xx_right'])

corr = np.round(corr, decimals=3)
p = np.round(p, decimals=3)

plt.text(-0.9, -0.14, 'Pearson R = ' + str(corr))
plt.text(-0.9, -0.17, 'p = ' + str(p))

############# second panel ####################
sns.set_palette(sns.color_palette(colors))
sns.scatterplot(data=df, x='AIC_baseline', y='RSI_yy_right', hue='keys', ax=axes[1])
sns.set_palette(sns.color_palette(colors_regplot))  # sets colors
sns.regplot(data=df, x='AIC_baseline', y='RSI_yy_right', scatter=False, ax=axes[1])

# set labels
axes[1].set_xlabel(xlabel='Anisotropy coefficient')
axes[1].set_ylabel(ylabel='$\mathrm{RSI_{yy, right}}$', labelpad=ylabeloffset)

# remove legend
axes[1].get_legend().remove()

# set limits
axes[1].set_ylim(ymin=-0.2, ymax=0.2)

# Define where you want ticks
plt.sca(axes[1])
yticks = np.arange(-0.2, 0.21, 0.1)
plt.yticks(yticks)
xticks = np.arange(-1, 1.1, 0.5)
plt.xticks(xticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in', which='minor', length=3, bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in', which='major', length=6, bottom=True, top=True, left=True, right=True)

corr, p = pearsonr(df['AIC_baseline'], df['RSI_yy_right'])

corr = np.round(corr, decimals=3)
p = np.round(p, decimals=3)

plt.text(-0.9, -0.14, 'Pearson R = ' + str(corr))
plt.text(-0.9, -0.17, 'p = ' + str(p))

plt.subplots_adjust(wspace=0.4, hspace=0.4)

plt.savefig(figfolder + 'D.png', dpi=300, bbox_inches="tight")
plt.show()
# %% plot figure 6plus, Correlation plots

# set plot parameters
ylabeloffset = -5
colors = colors_parent;  # defines colors for scatterplot
colors_regplot = ['#000000', '#000000', '#000000', '#000000'];  # defines colors for linear regression plot

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4, 1.8))

############# first panel ####################
sns.set_palette(sns.color_palette(colors))
sns.scatterplot(data=df, x='RSI_xx_left', y='RSI_xx_right', hue='keys', ax=axes[0])
sns.set_palette(sns.color_palette(colors_regplot))  # sets colors
sns.regplot(data=df, x='RSI_xx_left', y='RSI_xx_right', scatter=False, ax=axes[0])

# set labels
axes[0].set_xlabel(xlabel='$\mathrm{RSI_{xx, left}$')
axes[0].set_ylabel(ylabel='$\mathrm{RSI_{xx, right}}$', labelpad=ylabeloffset)

# remove legend
axes[0].get_legend().remove()

# set limits
axes[0].set_ylim(ymin=-0.2, ymax=0.2)

# Define where you want ticks
plt.sca(axes[0])
yticks = np.arange(-0.2, 0.21, 0.1)
plt.yticks(yticks)
xticks = np.arange(-1, 1.1, 0.5)
plt.xticks(xticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in', which='minor', length=3, bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in', which='major', length=6, bottom=True, top=True, left=True, right=True)

corr, p = pearsonr(df['RSI_xx_left'], df['RSI_xx_right'])

corr = np.round(corr, decimals=3)
p = np.round(p, decimals=3)

plt.text(-0.9, -0.14, 'Pearson R = ' + str(corr))
plt.text(-0.9, -0.17, 'p = ' + str(p))

############# second panel ####################
sns.set_palette(sns.color_palette(colors))
sns.scatterplot(data=df, x='RSI_yy_left', y='RSI_yy_right', hue='keys', ax=axes[1])
sns.set_palette(sns.color_palette(colors_regplot))  # sets colors
sns.regplot(data=df, x='RSI_yy_left', y='RSI_yy_right', scatter=False, ax=axes[1])

# set labels
axes[1].set_xlabel(xlabel='$\mathrm{RSI_{yy, left}}$')
axes[1].set_ylabel(ylabel='$\mathrm{RSI_{yy, right}}$', labelpad=ylabeloffset)

# remove legend
axes[1].get_legend().remove()

# set limits
axes[1].set_ylim(ymin=-0.2, ymax=0.2)

# Define where you want ticks
plt.sca(axes[1])
yticks = np.arange(-0.2, 0.21, 0.1)
plt.yticks(yticks)
xticks = np.arange(-1, 1.1, 0.5)
plt.xticks(xticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in', which='minor', length=3, bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in', which='major', length=6, bottom=True, top=True, left=True, right=True)

corr, p = pearsonr(df['RSI_yy_left'], df['RSI_yy_right'])

corr = np.round(corr, decimals=3)
p = np.round(p, decimals=3)

plt.text(-0.9, -0.14, 'Pearson R = ' + str(corr))
plt.text(-0.9, -0.17, 'p = ' + str(p))

plt.subplots_adjust(wspace=0.4, hspace=0.4)

plt.savefig(figfolder + 'sup.png', dpi=300, bbox_inches="tight")
plt.show()