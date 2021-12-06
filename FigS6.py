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

tissues_lefthalf_stim = pickle.load(open(folder + "analysed_data/tissues_lefthalf_stim.dat", "rb"))
tissues_tophalf_stim = pickle.load(open(folder + "analysed_data/tissues_tophalf_stim.dat", "rb"))

tissues_lefthalf_FEM_simulation = pickle.load(open(folder + "_FEM_simulations/FEM_tissues_side.dat", "rb"))
tissues_tophalf_FEM_simulation = pickle.load(open(folder + "_FEM_simulations/FEM_tissues_up.dat", "rb"))


# define some colors for the plots
colors_parent = ['#026473', '#E3CC69', '#77C8A6', '#D96248']
colors_parent_dark = ['#01353D', '#564910', '#235741', '#A93B23']

figfolder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/_figureS6/"
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

Tx_lefthalf_sim = np.nanmean(tissues_lefthalf_FEM_simulation["feedback0.0"]["t_x"][:, :, 0:20], axis=2) * 1e-3
Ty_lefthalf_sim = np.nanmean(tissues_lefthalf_FEM_simulation["feedback0.0"]["t_y"][:, :, 0:20], axis=2) * 1e-3

Tx_tophalf_sim = np.nanmean(tissues_tophalf_FEM_simulation["feedback0.0"]["t_x"][:, :, 0:20], axis=2) * 1e-3
Ty_tophalf_sim = np.nanmean(tissues_tophalf_FEM_simulation["feedback0.0"]["t_y"][:, :, 0:20], axis=2) * 1e-3

# calculate amplitudes
T_lefthalf_sim = np.sqrt(Tx_lefthalf_sim ** 2 + Ty_lefthalf_sim ** 2)
T_tophalf_sim = np.sqrt(Tx_tophalf_sim ** 2 + Ty_tophalf_sim ** 2)

# # pad simulation maps to make shapes square
paddingdistance = int((Tx_lefthalf_sim.shape[1] - Tx_lefthalf_sim.shape[0]) / 2)
Tx_lefthalf_sim = np.pad(Tx_lefthalf_sim, ((paddingdistance, paddingdistance), (0, 0)), "constant", constant_values=(0, 0))
Ty_lefthalf_sim = np.pad(Ty_lefthalf_sim, ((paddingdistance, paddingdistance), (0, 0)), "constant", constant_values=(0, 0))
T_lefthalf_sim = np.pad(T_lefthalf_sim, ((paddingdistance, paddingdistance), (0, 0)), "constant", constant_values=(0, 0))

Tx_tophalf_sim = np.pad(Tx_tophalf_sim, ((0, 0), (paddingdistance, paddingdistance)), "constant", constant_values=(0, 0))
Ty_tophalf_sim = np.pad(Ty_tophalf_sim, ((0, 0), (paddingdistance, paddingdistance)), "constant", constant_values=(0, 0))
T_tophalf_sim = np.pad(T_tophalf_sim, ((0, 0), (paddingdistance, paddingdistance)), "constant", constant_values=(0, 0))


# set up plot parameters
# *****************************************************************************
n = 4  # every nth arrow will be plotted
pixelsize = 1.296  # in µm
pmax = 2  # kPa

# create x- and y-axis for plotting maps
x_end = np.shape(T_lefthalf_sim)[1]
y_end = np.shape(T_lefthalf_sim)[0]
extent = [0, x_end * pixelsize, 0, y_end * pixelsize]

# create mesh for vectorplot
xq, yq = np.meshgrid(np.linspace(0, extent[1], x_end), np.linspace(0, extent[3], y_end))

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(2.5, 3))

im = axes[0].imshow(T_lefthalf_sim, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0,
                    vmax=pmax, aspect='auto')
axes[0].quiver(xq[::n, ::n], yq[::n, ::n], Tx_lefthalf_sim[::n, ::n], Ty_lefthalf_sim[::n, ::n],
               angles='xy', scale=10, units='width', color="r")
# axes[0,0].set_title('n=1', pad=-400, color='r')

axes[1].imshow(T_tophalf_sim, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0,
               vmax=pmax, aspect='auto')
axes[1].quiver(xq[::n, ::n], yq[::n, ::n], Tx_tophalf_sim[::n, ::n], Ty_tophalf_sim[::n, ::n],
               angles='xy', scale=10, units='width', color="r")


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

fig.savefig(figfolder + 'A.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'A.svg', dpi=300, bbox_inches="tight")

plt.show()
# %% plot figure 5B, stress maps

# prepare data first

# Calculate average maps over first 20 frames and all cells to get average maps
sigma_xx_lefthalf_sim = np.nanmean(tissues_lefthalf_FEM_simulation["feedback0.0"]["sigma_xx"][:, :, 0:20], axis=2) * 1e3
sigma_yy_lefthalf_sim = np.nanmean(tissues_lefthalf_FEM_simulation["feedback0.0"]["sigma_yy"][:, :, 0:20], axis=2) * 1e3

sigma_xx_tophalf_sim = np.nanmean(tissues_tophalf_FEM_simulation["feedback0.0"]["sigma_xx"][:, :, 0:20], axis=2) * 1e3
sigma_yy_tophalf_sim= np.nanmean(tissues_tophalf_FEM_simulation["feedback0.0"]["sigma_yy"][:, :, 0:20], axis=2) * 1e3

# # pad simulation maps to make shapes square
paddingdistance = int((sigma_xx_lefthalf_sim.shape[1] - sigma_xx_lefthalf_sim.shape[0]) / 2)
sigma_xx_lefthalf_sim = np.pad(sigma_xx_lefthalf_sim, ((paddingdistance, paddingdistance), (0, 0)), "constant", constant_values=(0, 0))
sigma_yy_lefthalf_sim = np.pad(sigma_yy_lefthalf_sim, ((paddingdistance, paddingdistance), (0, 0)), "constant", constant_values=(0, 0))

sigma_xx_tophalf_sim = np.pad(sigma_xx_tophalf_sim, ((0, 0), (paddingdistance, paddingdistance)), "constant", constant_values=(0, 0))
sigma_yy_tophalf_sim = np.pad(sigma_yy_tophalf_sim, ((0, 0), (paddingdistance, paddingdistance)), "constant", constant_values=(0, 0))


# # convert NaN to 0 to have black background
# sigma_xx_lefthalf_sim[np.isnan(sigma_xx_lefthalf_sim)] = 0
# sigma_yy_lefthalf_sim[np.isnan(sigma_yy_lefthalf_sim)] = 0
#
# sigma_xx_tophalf_sim[np.isnan(sigma_xx_tophalf_sim)] = 0
# sigma_yy_tophalf_sim[np.isnan(sigma_yy_tophalf_sim)] = 0


# set up plot parameters
# *****************************************************************************
n = 4  # every nth arrow will be plotted
pixelsize = 1.296  # in µm
pmax = 10  # mN/m

# create x- and y-axis for plotting maps
x_end = np.shape(sigma_xx_lefthalf_sim)[1]
y_end = np.shape(sigma_yy_lefthalf_sim)[0]
extent = [0, x_end * pixelsize, 0, y_end * pixelsize]

# create mesh for vectorplot
xq, yq = np.meshgrid(np.linspace(0, extent[1], x_end), np.linspace(0, extent[3], y_end))

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4, 3))

im = axes[0, 0].imshow(sigma_xx_lefthalf_sim, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent,
                       vmin=0, vmax=pmax, aspect='auto')
axes[1, 0].imshow(sigma_xx_tophalf_sim, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent,
                  vmin=0, vmax=pmax, aspect='auto')

axes[0, 1].imshow(sigma_yy_lefthalf_sim, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent,
                  vmin=0, vmax=pmax, aspect='auto')
axes[1, 1].imshow(sigma_yy_tophalf_sim, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent,
                  vmin=0, vmax=pmax, aspect='auto')


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
plt.text(-110, 316, '$\mathrm{\sigma _ {xx}}$', size=10)
plt.text(55, 316, '$\mathrm{\sigma _ {yy}}$', size=10)

# add annotations
plt.text(0.18, 0.8, 'n=' + str(n_ls), transform=plt.figure(1).transFigure, color='w')
plt.text(0.18, 0.42, 'n=' + str(n_ts), transform=plt.figure(1).transFigure, color='w')

plt.text(0.48, 0.8, 'n=' + str(n_ls), transform=plt.figure(1).transFigure, color='w')
plt.text(0.48, 0.42, 'n=' + str(n_ts), transform=plt.figure(1).transFigure, color='w')

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

# %% plot figure 5D, stress map differences

# prepare data first

# concatenate TFM maps from different experiments and calculate average maps over first 20 frames and all cells to get average maps
sigma_xx_lefthalf_sim_diff = \
                (tissues_lefthalf_FEM_simulation["feedback0.0"]["sigma_xx"][:, :, 32] -
                 tissues_lefthalf_FEM_simulation["feedback0.0"]["sigma_xx"][:, :, 20]) * 1e3  # convert to mN/m
sigma_yy_lefthalf_sim_diff = \
                (tissues_lefthalf_FEM_simulation["feedback0.0"]["sigma_yy"][:, :, 32] -
                 tissues_lefthalf_FEM_simulation["feedback0.0"]["sigma_yy"][:, :, 20]) * 1e3 # convert to mN/m

sigma_xx_tophalf_sim_diff = \
                (tissues_tophalf_FEM_simulation["feedback0.0"]["sigma_xx"][:, :, 32] -
                 tissues_tophalf_FEM_simulation["feedback0.0"]["sigma_xx"][:, :, 20]) * 1e3  # convert to mN/m
sigma_yy_tophalf_sim_diff = \
                (tissues_tophalf_FEM_simulation["feedback0.0"]["sigma_yy"][:, :, 32] -
                 tissues_tophalf_FEM_simulation["feedback0.0"]["sigma_yy"][:, :, 20]) * 1e3


# pad simulation maps to make shapes square
paddingdistance = int((sigma_xx_lefthalf_sim_diff.shape[1] - sigma_xx_lefthalf_sim_diff.shape[0]) / 2)
sigma_xx_lefthalf_sim_diff = np.pad(sigma_xx_lefthalf_sim_diff, ((paddingdistance, paddingdistance), (0, 0)), "constant", constant_values=(0, 0))
sigma_yy_lefthalf_sim_diff = np.pad(sigma_yy_lefthalf_sim_diff, ((paddingdistance, paddingdistance), (0, 0)), "constant", constant_values=(0, 0))

sigma_xx_tophalf_sim_diff = np.pad(sigma_xx_tophalf_sim_diff, ((0, 0), (paddingdistance, paddingdistance)), "constant", constant_values=(0, 0))
sigma_yy_tophalf_sim_diff = np.pad(sigma_yy_tophalf_sim_diff, ((0, 0), (paddingdistance, paddingdistance)), "constant", constant_values=(0, 0))


# set up plot parameters
# *****************************************************************************

pixelsize = 1.296  # in µm
sigma_max = 1  # kPa
sigma_min = -1  # kPa

# create x- and y-axis for plotting maps
x_end = np.shape(sigma_xx_lefthalf_sim_diff)[1]
y_end = np.shape(sigma_yy_lefthalf_sim_diff)[0]
extent = [0, x_end * pixelsize, 0, y_end * pixelsize]

# create mesh for vectorplot
xq, yq = np.meshgrid(np.linspace(0, extent[1], x_end), np.linspace(0, extent[3], y_end))

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4, 3))

im = axes[0][0].imshow(sigma_xx_lefthalf_sim_diff, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent,
                       vmin=sigma_min, vmax=sigma_max, aspect='auto')
axes[0][1].imshow(sigma_yy_lefthalf_sim_diff, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent,
                  vmin=sigma_min, vmax=sigma_max, aspect='auto')
axes[1][0].imshow(sigma_xx_tophalf_sim_diff, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent,
                  vmin=sigma_min, vmax=sigma_max, aspect='auto')
axes[1][1].imshow(sigma_yy_tophalf_sim_diff, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent,
                  vmin=sigma_min, vmax=sigma_max, aspect='auto')

# adjust space in between plots
plt.subplots_adjust(wspace=0.2, hspace=0)


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
plt.text(-110, 316, '$\mathrm{\Delta \sigma _ {xx}}$', size=10)
plt.text(55, 316, '$\mathrm{\Delta \sigma _ {yy}}$', size=10)


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

color = colors_parent[0]

# panel 1
ax = axes[0, 0]
c = 0
for key in tissues_lefthalf_FEM_simulation:
    c += 1
    y_sim = tissues_lefthalf_FEM_simulation[key]["sigma_xx_x_profile_increase"] * 1e3  # convert to nN
    x_sim = np.linspace(-75, 75, y_sim.shape[0])
    ax.plot(x_sim, y_sim, color=adjust_lightness(color, c / 10), alpha=1, linewidth=0.7)

# panel 2
ax = axes[0, 1]
c = 0
for key in tissues_lefthalf_FEM_simulation:
    c += 1
    y_sim = tissues_lefthalf_FEM_simulation[key]["sigma_yy_x_profile_increase"] * 1e3  # convert to nN
    x_sim = np.linspace(-75, 75, y_sim.shape[0])
    ax.plot(x_sim, y_sim, color=adjust_lightness(color, c / 10), alpha=1, linewidth=0.7)

color = colors_parent[3]

# panel 3
ax = axes[1, 0]
c = 0
for key in tissues_tophalf_FEM_simulation:
    c += 1
    y_sim = tissues_tophalf_FEM_simulation[key]["sigma_xx_x_profile_increase"] * 1e3  # convert to nN
    x_sim = np.linspace(-20, 20, y_sim.shape[0])
    ax.plot(x_sim, y_sim, color=adjust_lightness(color, c / 10), alpha=1, linewidth=0.7)

# panel 4
ax = axes[1, 1]
c = 0
for key in tissues_tophalf_FEM_simulation:
    c += 1
    y_sim = tissues_tophalf_FEM_simulation[key]["sigma_yy_x_profile_increase"] * 1e3  # convert to nN
    x_sim = np.linspace(-20, 20, y_sim.shape[0])
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
    ax.axvline(x=0, ymin=0.0, ymax=1, linewidth=0.5, color="cyan")

fig.savefig(figfolder + "E.png", dpi=300, bbox_inches="tight")
fig.savefig(figfolder + "E.svg", dpi=300, bbox_inches="tight")
plt.show()




