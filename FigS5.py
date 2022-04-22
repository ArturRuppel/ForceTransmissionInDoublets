"""

@author: Artur Ruppel

"""
import os
import pickle
import pandas as pd
from scipy.stats import zscore
from plot_and_filter_functions import *
from scipy.interpolate import interp1d


pixelsize = 0.864  # in µm
# %% load data for plotting
folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"
AR1to2d_halfstim = pickle.load(open(folder + "analysed_data/AR1to2d_halfstim.dat", "rb"))
AR1to1d_halfstim = pickle.load(open(folder + "analysed_data/AR1to1d_halfstim.dat", "rb"))
AR2to1d_halfstim = pickle.load(open(folder + "analysed_data/AR2to1d_halfstim.dat", "rb"))

AR1to2_CM_simulation = pickle.load(open(folder + "_contour_simulations/CM_1to2_simulation.dat", "rb"))
AR1to1_CM_simulation = pickle.load(open(folder + "_contour_simulations/CM_1to1_simulation.dat", "rb"))
AR2to1_CM_simulation = pickle.load(open(folder + "_contour_simulations/CM_2to1_simulation.dat", "rb"))
# tissues_lefthalf_stim = pickle.load(open(folder + "analysed_data/tissues_lefthalf_stim.dat", "rb"))
# tissues_tophalf_stim = pickle.load(open(folder + "analysed_data/tissues_tophalf_stim.dat", "rb"))
#
#
#
# AR1to2_FEM_simulation = pickle.load(open(folder + "_FEM_simulations/FEM_1to2.dat", "rb"))
# AR1to1_FEM_simulation = pickle.load(open(folder + "_FEM_simulations/FEM_singlets.dat", "rb"))
# AR2to1_FEM_simulation = pickle.load(open(folder + "_FEM_simulations/FEM_2to1.dat", "rb"))
# tissues_lefthalf_FEM_simulation = pickle.load(open(folder + "_FEM_simulations/FEM_tissues_lefthalfstim.dat", "rb"))
# tissues_tophalf_FEM_simulation = pickle.load(open(folder + "_FEM_simulations/FEM_tissues_tophalfstim.dat", "rb"))


# define some colors for the plots
colors_parent = ['#026473', '#E3CC69', '#77C8A6', '#D96248']
colors_parent_dark = ['#01353D', '#564910', '#235741', '#A93B23']

figfolder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/_FigureS5/"
if not os.path.exists(figfolder):
    os.mkdir(figfolder)

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

# %% plot figure B, contour strain after photoactivation

# set up global plot parameters
# ******************************************************************************************************************************************
ymin = -0.05
ymax = 0
xticks = np.arange(-15, 15.1, 15)  # define where the major ticks are gonna be
yticks = np.arange(ymin, ymax + 0.001, 0.01)
xlabel = "position [µm]"
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(4.5, 1.3))  # create figure and axes
plt.subplots_adjust(wspace=0.5, hspace=0.35)  # adjust space in between plots
# ******************************************************************************************************************************************

# Set up plot parameters for first panel
#######################################################################################################
ax = axes[0]
color = colors_parent[0]
ylabel = None
title = "Contour strain \n measurement"
x = np.linspace(-25, 25, 50)
x = x[::2]  # downsample data for nicer plotting
y = AR1to2d_halfstim["shape_data"]["contour_strain"]
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False, xmin=-25, xmax=25)

# Set up plot parameters for first panel
#######################################################################################################
ax = axes[1]
color = colors_parent[1]
ylabel = None
title = "Contour strain \n measurement"
x = np.linspace(-17.5, 17.5, 50)
x = x[::2]  # downsample data for nicer plotting
y = AR1to1d_halfstim["shape_data"]["contour_strain"]
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False, xmin=-25, xmax=25)

# Set up plot parameters for first panel
#######################################################################################################
ax = axes[2]
color = colors_parent[3]
ylabel = None
title = "Contour strain \n measurement"
x = np.linspace(-12.5, 12.5, 50)
x = x[::2]  # downsample data for nicer plotting
y = AR2to1d_halfstim["shape_data"]["contour_strain"]
y = y[::2, :]

# make plots
plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, color, optolinewidth=False, xmin=-25, xmax=25)


plt.savefig(figfolder + "B.png", dpi=300, bbox_inches="tight")
plt.savefig(figfolder + "B.svg", dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure C, active coupling of contour

# make some calculations on the simulated data first
strain_ratio_1to2_sim = []
strain_ratio_1to1_sim = []
strain_ratio_2to1_sim = []
xticks = np.arange(-0.5, 1.01, 0.5)

feedbacks = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for fb in feedbacks:
    epsilon = AR1to2_CM_simulation["EA300"]["FB" + str(fb)]["epsilon_yy"]
    center = int(epsilon.shape[0] / 2)
    epsilon_ratio = - np.nansum(epsilon[center:-1]) / (np.nansum(np.abs(epsilon)))
    strain_ratio_1to2_sim.append(epsilon_ratio)

    epsilon = AR1to1_CM_simulation["EA300"]["FB" + str(fb)]["epsilon_yy"]
    center = int(epsilon.shape[0] / 2)
    epsilon_ratio = - np.nansum(epsilon[center:-1]) / (np.nansum(np.abs(epsilon)))
    strain_ratio_1to1_sim.append(epsilon_ratio)

    epsilon = AR2to1_CM_simulation["EA300"]["FB" + str(fb)]["epsilon_yy"]
    center = int(epsilon.shape[0] / 2)
    epsilon_ratio = - np.nansum(epsilon[center:-1]) / (np.nansum(np.abs(epsilon)))
    strain_ratio_2to1_sim.append(epsilon_ratio)


def find_x_position_of_point_on_array(x_list, y_list, y_point):
    f = interp1d(y_list, x_list, kind="cubic")
    return f(y_point)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(1.3, 1.3))

ax.plot(feedbacks, strain_ratio_1to2_sim, color=colors_parent[0])
ax.plot(feedbacks, strain_ratio_1to1_sim, color=colors_parent[1])
ax.plot(feedbacks, strain_ratio_2to1_sim, color=colors_parent[3])


# add data points
contour_strain_1to2 = np.nanmean(AR1to2d_halfstim["shape_data"]["contour_strain"], axis=1)
contour_strain_1to2_sem = np.nanstd(AR1to2d_halfstim["shape_data"]["contour_strain"], axis=1) / np.sqrt(
    np.shape(AR1to2d_halfstim["shape_data"]["contour_strain"])[1])

contour_strain_1to1 = np.nanmean(AR1to1d_halfstim["shape_data"]["contour_strain"], axis=1)
contour_strain_1to1_sem = np.nanstd(AR1to1d_halfstim["shape_data"]["contour_strain"], axis=1) / np.sqrt(
    np.shape(AR1to1d_halfstim["shape_data"]["contour_strain"])[1])

contour_strain_2to1 = np.nanmean(AR2to1d_halfstim["shape_data"]["contour_strain"], axis=1)
contour_strain_2to1_sem = np.nanstd(AR2to1d_halfstim["shape_data"]["contour_strain"], axis=1) / np.sqrt(
    np.shape(AR2to1d_halfstim["shape_data"]["contour_strain"])[1])

center = int(contour_strain_1to1.shape[0] / 2)

strain_right_1to2 = np.nansum(contour_strain_1to2[center:-1])
strain_right_1to2_err = np.sqrt(np.nansum(contour_strain_1to2_sem[center:-1] ** 2))
strain_left_1to2 = np.nansum(contour_strain_1to2[0:center])
strain_left_1to2_err = np.sqrt(np.nansum(contour_strain_1to2_sem[0:center] ** 2))

strain_right_1to1 = np.nansum(contour_strain_1to1[center:-1])
strain_right_1to1_err = np.sqrt(np.nansum(contour_strain_1to1_sem[center:-1] ** 2))
strain_left_1to1 = np.nansum(contour_strain_1to1[0:center])
strain_left_1to1_err = np.sqrt(np.nansum(contour_strain_1to1_sem[0:center] ** 2))

strain_right_2to1 = np.nansum(contour_strain_2to1[center:-1])
strain_right_2to1_err = np.sqrt(np.nansum(contour_strain_2to1_sem[center:-1] ** 2))
strain_left_2to1 = np.nansum(contour_strain_2to1[0:center])
strain_left_2to1_err = np.sqrt(np.nansum(contour_strain_2to1_sem[0:center] ** 2))


# calculate error with propagation of uncertainty
strain_ratio_1to2 = strain_right_1to2 / (strain_left_1to2 + strain_right_1to2)
strain_ratio_1to2_err = (strain_right_1to2_err * strain_left_1to2 + strain_left_1to2_err * strain_right_1to2) / ((strain_left_1to2 + strain_right_1to2) ** 2)

strain_ratio_1to1 = strain_right_1to1 / (strain_left_1to1 + strain_right_1to1)
strain_ratio_1to1_err = (strain_right_1to1_err * strain_left_1to1 + strain_left_1to1_err * strain_right_1to1) / ((strain_left_1to1 + strain_right_1to1) ** 2)

strain_ratio_2to1 = strain_right_2to1 / (strain_left_2to1 + strain_right_2to1)
strain_ratio_2to1_err = (strain_right_2to1_err * strain_left_2to1 + strain_left_2to1_err * strain_right_2to1) / ((strain_left_2to1 + strain_right_2to1) ** 2)


x = find_x_position_of_point_on_array(feedbacks, strain_ratio_1to2_sim, strain_ratio_1to2)
ax.errorbar(x, strain_ratio_1to2, yerr=strain_ratio_1to2_err, mfc="w", color=colors_parent[0], marker="s", ms=5, linewidth=0.5, ls="none",
            markeredgewidth=0.5)

x = find_x_position_of_point_on_array(feedbacks, strain_ratio_1to1_sim, strain_ratio_1to1)
ax.errorbar(x, strain_ratio_1to1, yerr=strain_ratio_1to1_err, mfc="w", color=colors_parent[1], marker="s", ms=5, linewidth=0.5, ls="none",
            markeredgewidth=0.5)

x = find_x_position_of_point_on_array(feedbacks, strain_ratio_2to1_sim, strain_ratio_2to1)
ax.errorbar(x, strain_ratio_2to1, yerr=strain_ratio_2to1_err, mfc="w", color=colors_parent[3], marker="s", ms=5, linewidth=0.5, ls="none",
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
plt.savefig(figfolder + "C.png", dpi=300, bbox_inches="tight")
plt.savefig(figfolder + "C.svg", dpi=300, bbox_inches="tight")
plt.show()