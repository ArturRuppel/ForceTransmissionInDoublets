"""

@author: Artur Ruppel

"""
import os
import pickle
import pandas as pd
from scipy.stats import zscore
from plot_and_filter_functions import *
from scipy.interpolate import interp1d
import scipy.io

pixelsize = 0.864  # in µm

def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True, color='blue'):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='black', fill=True, color=color, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches

# %% load data for plotting
colors_parent = ['#026473', '#E3CC69', '#77C8A6', '#D96248']
colors_parent_dark = ['#01353D', '#564910', '#235741', '#A93B23']

folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"
AR1to2d_halfstim = pickle.load(open(folder + "analysed_data/AR1to2d_halfstim.dat", "rb"))
AR1to1d_halfstim = pickle.load(open(folder + "analysed_data/AR1to1d_halfstim.dat", "rb"))
AR1to1s_halfstim = pickle.load(open(folder + "analysed_data/AR1to1s_halfstim.dat", "rb"))
AR2to1d_halfstim = pickle.load(open(folder + "analysed_data/AR2to1d_halfstim.dat", "rb"))

t_x = (AR1to2d_halfstim["TFM_data"]["Tx"][:, :, 0, :].reshape(-1))
t_y = (AR1to2d_halfstim["TFM_data"]["Ty"][:, :, 0, :].reshape(-1))
force_angles_1to2 = np.arctan2(t_y, t_x)

t_x = (AR1to1d_halfstim["TFM_data"]["Tx"][:, :, 0, :].reshape(-1))
t_y = (AR1to1d_halfstim["TFM_data"]["Ty"][:, :, 0, :].reshape(-1))
force_angles_1to1 = np.arctan2(t_y, t_x)

t_x = (AR2to1d_halfstim["TFM_data"]["Tx"][:, :, 0, :].reshape(-1))
t_y = (AR2to1d_halfstim["TFM_data"]["Ty"][:, :, 0, :].reshape(-1))
force_angles_2to1 = np.arctan2(t_y, t_x)

t_x = (AR1to1s_halfstim["TFM_data"]["Tx"][:, :, 0, :].reshape(-1))
t_y = (AR1to1s_halfstim["TFM_data"]["Ty"][:, :, 0, :].reshape(-1))
force_angles_1to1s = np.arctan2(t_y, t_x)
# # %% make plots
# # Construct figure and axis to plot on
# fig, ax = plt.subplots(1, 3, subplot_kw=dict(projection='polar'))
# plt.subplots_adjust(wspace=0.5, hspace=0)
#
#
# # Visualise by area of bins
# circular_hist(ax[0], force_angles_1to2, color=colors_parent[0])
# circular_hist(ax[1], force_angles_1to1, color=colors_parent[1])
# circular_hist(ax[2], force_angles_2to1, color=colors_parent[3])
#
# fig.savefig('force_histograms.png', dpi=300, bbox_inches="tight")
#
# plt.show()
#
# fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))
# plt.subplots_adjust(wspace=0.5, hspace=0)
#
#
# # Visualise by area of bins
# circular_hist(ax[0], force_angles_1to1, color=colors_parent[1])
# circular_hist(ax[1], force_angles_1to1s, color=colors_parent[2])
#
# fig.savefig('force_histograms2.png', dpi=300, bbox_inches="tight")
#
# plt.show()
#
# # %% load data for plotting
# actin_angles_1to2 = scipy.io.loadmat("D:/2020_OPTO H2000 stimulate left half doublets and singlets/TFM_doublets/AR1to2/actin_angles_all.mat")["angles_all"]
# actin_angles_1to1 = scipy.io.loadmat("D:/2020_OPTO H2000 stimulate left half doublets and singlets/TFM_doublets/AR1to1/actin_angles_all.mat")["angles_all"]
# actin_angles_1to1s = scipy.io.loadmat("D:/2020_OPTO H2000 stimulate left half doublets and singlets/TFM_singlets/actin_angles_all.mat")["angles_all"]
# actin_angles_2to1 = scipy.io.loadmat("D:/2020_OPTO H2000 stimulate left half doublets and singlets/TFM_doublets/AR2to1/actin_angles_all.mat")["angles_all"]
#
#
# fig, ax = plt.subplots(1, 3, subplot_kw=dict(projection='polar'))
# plt.subplots_adjust(wspace=0.5, hspace=0)
#
#
# # Visualise by area of bins
# circular_hist(ax[0], actin_angles_1to2, color=colors_parent[0])
# circular_hist(ax[1], actin_angles_1to1, color=colors_parent[1])
# circular_hist(ax[2], actin_angles_2to1, color=colors_parent[3])
#
# fig.savefig('actin_histogram1.png', dpi=300, bbox_inches="tight")
#
# plt.show()
#
#
# fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))
# plt.subplots_adjust(wspace=0.5, hspace=0)
#
# # Visualise by area of bins
# circular_hist(ax[0], actin_angles_1to1, color=colors_parent[1])
# circular_hist(ax[1], actin_angles_1to1s, color=colors_parent[2])
#
# fig.savefig('actin_histogram2.png', dpi=300, bbox_inches="tight")
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

# %%
# prepare data first
for t in range(60):

    # concatenate TFM maps from different experiments and calculate average maps over first 20 frames and all cells to get average maps
    Tx_1to2 = np.nanmean(AR1to2d_halfstim["TFM_data"]["Tx"][:, :, t, :], axis=2)
    Ty_1to2 = np.nanmean(AR1to2d_halfstim["TFM_data"]["Ty"][:, :, t, :], axis=2)

    Tx_1to1 = np.nanmean(AR1to1d_halfstim["TFM_data"]["Tx"][:, :, t, :], axis=2)
    Ty_1to1 = np.nanmean(AR1to1d_halfstim["TFM_data"]["Ty"][:, :, t, :], axis=2)

    Tx_2to1 = np.nanmean(AR2to1d_halfstim["TFM_data"]["Tx"][:, :, t, :], axis=2)
    Ty_2to1 = np.nanmean(AR2to1d_halfstim["TFM_data"]["Ty"][:, :, t, :], axis=2)



    # calculate amplitudes
    T_1to2 = np.sqrt(Tx_1to2 ** 2 + Ty_1to2 ** 2)
    T_1to1 = np.sqrt(Tx_1to1 ** 2 + Ty_1to1 ** 2)
    T_2to1 = np.sqrt(Tx_2to1 ** 2 + Ty_2to1 ** 2)

    # crop maps
    crop_start = 8
    crop_end = 84

    Tx_1to2_crop = Tx_1to2[crop_start:crop_end, crop_start:crop_end] * 1e-3  # convert to kPa
    Ty_1to2_crop = Ty_1to2[crop_start:crop_end, crop_start:crop_end] * 1e-3
    T_1to2_crop = T_1to2[crop_start:crop_end, crop_start:crop_end] * 1e-3

    Tx_1to1_crop = Tx_1to1[crop_start:crop_end, crop_start:crop_end] * 1e-3  # convert to kPa
    Ty_1to1_crop = Ty_1to1[crop_start:crop_end, crop_start:crop_end] * 1e-3
    T_1to1_crop = T_1to1[crop_start:crop_end, crop_start:crop_end] * 1e-3

    Tx_2to1_crop = Tx_2to1[crop_start:crop_end, crop_start:crop_end] * 1e-3  # convert to kPa
    Ty_2to1_crop = Ty_2to1[crop_start:crop_end, crop_start:crop_end] * 1e-3
    T_2to1_crop = T_2to1[crop_start:crop_end, crop_start:crop_end] * 1e-3

    # set up plot parameters
    # ******************************************************************************************************************************************
    n = 4                           # every nth arrow will be plotted
    pixelsize = 0.864               # in µm
    pmin = 0
    pmax = 1.5                        # in kPa
    axtitle = 'kPa'                 # unit of colorbar
    suptitle = 'Traction forces'    # title of plot
    x_end = np.shape(T_1to1)[1]   # create x- and y-axis for plotting maps
    y_end = np.shape(T_1to1)[0]
    extent = [0, x_end * pixelsize, 0, y_end * pixelsize]
    xq, yq = np.meshgrid(np.linspace(0, extent[1], x_end), np.linspace(0, extent[3], y_end))  # create mesh for vectorplot
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 2))    # create figure and axes
    plt.subplots_adjust(wspace=-0.15, hspace=-0.06)      # adjust space in between plots
    # ******************************************************************************************************************************************

    im = plot_forcemaps(axes[0], Tx_1to2_crop, Ty_1to2_crop, pixelsize, pmax, pmin)
    plot_forcemaps(axes[1], Tx_1to1_crop, Ty_1to1_crop, pixelsize, pmax, pmin)
    plot_forcemaps(axes[2], Tx_2to1_crop, Ty_2to1_crop, pixelsize, pmax, pmin)



    # remove axes
    for ax in axes.flat:
        ax.axis('off')
        aspectratio = 1.0
        ratio_default = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.set_aspect(ratio_default * aspectratio)

        if t >= 20 and t < 30:
            rect = mpl.patches.Rectangle((-31, -30.5), 21.5, 61, edgecolor='cyan', facecolor='none', lw=3)
            ax.add_patch(rect)

    # add colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist())
    cbar.ax.set_title(axtitle)

    plt.show()

    fig.savefig(str(t) + 'forces.png', dpi=300, bbox_inches="tight")
    plt.show()

#%%
x = np.arange(60)
# downsample data for nicer plotting
xticks = np.arange(0, 61, 20)  # define where the major ticks are gonna be
ymin = -0.1
ymax = 0.2
yticks = np.arange(ymin, ymax + 0.01, 0.1)
xlabel = 'time [min]'
ylabel = None
title = None

for t in np.arange(60):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 2))  # create figure and axes

    if t >= 20 and t<30:
        for i in np.arange(t-19):
            ax[0].axline((20 + i, ymin), (20 + i, ymax), linewidth=0.2, color="cyan")
            ax[1].axline((20 + i, ymin), (20 + i, ymax), linewidth=0.2, color="cyan")
            ax[2].axline((20 + i, ymin), (20 + i, ymax), linewidth=0.2, color="cyan")

    if t >= 30:
        for i in np.arange(10):
            ax[0].axline((20 + i, ymin), (20 + i, ymax), linewidth=0.2, color="cyan")
            ax[1].axline((20 + i, ymin), (20 + i, ymax), linewidth=0.2, color="cyan")
            ax[2].axline((20 + i, ymin), (20 + i, ymax), linewidth=0.2, color="cyan")

    colors = [colors_parent[0], colors_parent_dark[0]]
    y1 = AR1to2d_halfstim["TFM_data"]["relEs_left"]
    y2 = AR1to2d_halfstim["TFM_data"]["relEs_right"]


    # make plots
    plot_two_values_over_time(x[0:t], y1[0:t,:], y2[0:t,:], xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax[0], colors)


    colors = [colors_parent[1], colors_parent_dark[1]]
    y1 = AR1to1d_halfstim["TFM_data"]["relEs_left"]
    y2 = AR1to1d_halfstim["TFM_data"]["relEs_right"]


    # make plots
    plot_two_values_over_time(x[0:t], y1[0:t,:], y2[0:t,:], xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax[1], colors)

    colors = [colors_parent[3], colors_parent_dark[3]]
    y1 = AR2to1d_halfstim["TFM_data"]["relEs_left"]
    y2 = AR2to1d_halfstim["TFM_data"]["relEs_right"]

    # make plots
    plot_two_values_over_time(x[0:t], y1[0:t,:], y2[0:t,:], xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax[2], colors)

    # ax[0].plot(strain_energy[0:t+1] / np.nanmean(strain_energy[0:20]))

    plt.suptitle("Relative strain energy")
    ax[0].set_xlim([0, 60])
    ax[1].set_xlim([0, 60])
    ax[2].set_xlim([0, 60])
    plt.xlabel("time [min]")
    plt.savefig("relative_strain_energy_"+str(t)+".png", dpi=300, bbox_inches="tight")