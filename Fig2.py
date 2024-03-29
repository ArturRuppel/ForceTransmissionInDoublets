# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 21:56:01 2021

@author: Artur Ruppel

"""
import os
import pickle
import matplotlib.image as mpimg
import pandas as pd
from plot_and_filter_functions import *

# define some colors for the plots
colors_parent = ['#026473', '#E3CC69', '#77C8A6', '#D96248']
mpl.rcParams['font.size'] = 8

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

folder = "C:/Users/aruppel/Documents/_forcetransmission_in_cell_doublets_raw/"
figfolder = folder + "_Figure2/"
if not os.path.exists(figfolder):
    os.mkdir(figfolder)
    
# %% load data for plotting
AR1to1d_fullstim_long = pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_long.dat", "rb"))
AR1to1d_fullstim_short = pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_short.dat", "rb"))
AR1to1d_halfstim = pickle.load(open(folder + "analysed_data/AR1to1d_halfstim.dat", "rb"))

AR1to1s_fullstim_long = pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_long.dat", "rb"))
AR1to1s_fullstim_short = pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_short.dat", "rb"))
AR1to1s_halfstim = pickle.load(open(folder + "analysed_data/AR1to1s_halfstim.dat", "rb"))

# load contour model analysis data
AR1to1d_fullstim_long_CM = pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_long_CM.dat", "rb"))
AR1to1d_fullstim_short_CM = pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_short_CM.dat", "rb"))
AR1to1d_halfstim_CM = pickle.load(open(folder + "analysed_data/AR1to1d_halfstim_CM.dat", "rb"))

AR1to1s_fullstim_long_CM = pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_long_CM.dat", "rb"))
AR1to1s_fullstim_short_CM = pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_short_CM.dat", "rb"))
AR1to1s_halfstim_CM = pickle.load(open(folder + "analysed_data/AR1to1s_halfstim_CM.dat", "rb"))


# %% set up pandas data frame to use with seaborn for box- and swarmplots

# initialize empty dictionaries
concatenated_data_1to1d = {}
concatenated_data_1to1s = {}
concatenated_data = {}

# loop over all keys for contour model analysis data
for key1 in AR1to1d_fullstim_long_CM:
    if key1 == "ellipse_data_approx":
        continue
    for key2 in AR1to1d_fullstim_long_CM[key1]:
        if AR1to1d_fullstim_long_CM[key1][key2].ndim == 1:  # only 1D data can be stored in the data frame

            # concatenate values from different experiments
            concatenated_data_1to1d[key2] = np.concatenate(
                (AR1to1d_halfstim_CM[key1][key2], AR1to1d_fullstim_short_CM[key1][key2], AR1to1d_fullstim_long_CM[key1][key2]))
            concatenated_data_1to1s[key2] = np.concatenate(
                (AR1to1s_halfstim_CM[key1][key2], AR1to1s_fullstim_short_CM[key1][key2], AR1to1s_fullstim_long_CM[key1][key2]))

            # concatenate doublet and singlet data to create pandas dataframe
            concatenated_data[key2] = np.concatenate((concatenated_data_1to1d[key2], concatenated_data_1to1s[key2]))

# loop over all keys for TFM and MSM data
for key1 in AR1to1d_fullstim_long:
    for key2 in AR1to1d_fullstim_long[key1]:
        if AR1to1d_fullstim_long[key1][key2].ndim == 1:  # only 1D data can be stored in the data frame

            # concatenate values from different experiments
            concatenated_data_1to1d[key2] = np.concatenate(
                (AR1to1d_halfstim[key1][key2], AR1to1d_fullstim_short[key1][key2], AR1to1d_fullstim_long[key1][key2]))
            concatenated_data_1to1s[key2] = np.concatenate(
                (AR1to1s_halfstim[key1][key2], AR1to1s_fullstim_short[key1][key2], AR1to1s_fullstim_long[key1][key2]))

            # concatenate doublet and singlet data to create pandas dataframe
            concatenated_data[key2] = np.concatenate((concatenated_data_1to1d[key2], concatenated_data_1to1s[key2]))

# get number of elements for both conditions
n_doublets = concatenated_data_1to1d[key2].shape[0]
n_singlets = concatenated_data_1to1s[key2].shape[0]

# create a list of keys with the same dimensions as the data
keys1to1d = ['AR1to1d' for i in range(n_doublets)]
keys1to1s = ['AR1to1s' for i in range(n_singlets)]
keys = np.concatenate((keys1to1d, keys1to1s))

# add keys to dictionary with concatenated data
concatenated_data['keys'] = keys

# create DataFrame
df = pd.DataFrame(concatenated_data)


# convert to more convenient units for plotting
df['Es_baseline'] *= 1e12  # convert to fJ
df['spreadingsize_baseline'] *= 1e12  # convert to µm²
df['sigma_x_baseline'] *= 1e3  # convert to mN/m
df['sigma_xx_baseline'] *= 1e3  # convert to mN/m
df['sigma_yy_baseline'] *= 1e3  # convert to mN/m
# %% load and prepare data for figure 2C, actin images, forces, ellipses and tangents

# prepare data first
pixelsize = 0.864  # in µm
initial_pixelsize = 0.108  # in µm
# concatenate TFM maps from different experiments and calculate average maps over first 20 frames and all cells to get average maps
doublet_example = 0
singlet_example = 1

# get data for one example
# forces
Tx_1to1d = AR1to1d_fullstim_long["TFM_data"]["Tx"][:, :, 0, doublet_example]
Ty_1to1d = AR1to1d_fullstim_long["TFM_data"]["Ty"][:, :, 0, doublet_example]

Tx_1to1s = AR1to1s_fullstim_long["TFM_data"]["Tx"][:, :, 0, singlet_example]
Ty_1to1s = AR1to1s_fullstim_long["TFM_data"]["Ty"][:, :, 0, singlet_example]

# actin images
actin_image_path = folder + "AR1to1_doublets_full_stim_long/actin_images/cell" + str(doublet_example) + "frame0.png"
actin_image_1to1d = rgb2gray(mpimg.imread(actin_image_path))
actin_image_path = folder + "AR1to1_singlets_full_stim_long/actin_images/cell" + str(
    singlet_example) + "frame0.png"  # cell 1 got filtered out so the data is not corresponding to the same index of the actin image
actin_image_1to1s = rgb2gray(mpimg.imread(actin_image_path))

# ellipse data
a_top_1to1d = AR1to1d_fullstim_long_CM["ellipse_data"]["a top [um]"][0, doublet_example]
b_top_1to1d = AR1to1d_fullstim_long_CM["ellipse_data"]["b top [um]"][0, doublet_example]
xc_top_1to1d = AR1to1d_fullstim_long_CM["ellipse_data"]["xc top [um]"][0, doublet_example]
yc_top_1to1d = AR1to1d_fullstim_long_CM["ellipse_data"]["yc top [um]"][0, doublet_example]
a_bottom_1to1d = AR1to1d_fullstim_long_CM["ellipse_data"]["a bottom [um]"][0, doublet_example]
b_bottom_1to1d = AR1to1d_fullstim_long_CM["ellipse_data"]["b bottom [um]"][0, doublet_example]
xc_bottom_1to1d = AR1to1d_fullstim_long_CM["ellipse_data"]["xc bottom [um]"][0, doublet_example]
yc_bottom_1to1d = AR1to1d_fullstim_long_CM["ellipse_data"]["yc bottom [um]"][0, doublet_example]

a_top_1to1s = AR1to1s_fullstim_long_CM["ellipse_data"]["a top [um]"][0, singlet_example]
b_top_1to1s = AR1to1s_fullstim_long_CM["ellipse_data"]["b top [um]"][0, singlet_example]
xc_top_1to1s = AR1to1s_fullstim_long_CM["ellipse_data"]["xc top [um]"][0, singlet_example]
yc_top_1to1s = AR1to1s_fullstim_long_CM["ellipse_data"]["yc top [um]"][0, singlet_example]
a_bottom_1to1s = AR1to1s_fullstim_long_CM["ellipse_data"]["a bottom [um]"][0, singlet_example]
b_bottom_1to1s = AR1to1s_fullstim_long_CM["ellipse_data"]["b bottom [um]"][0, singlet_example]
xc_bottom_1to1s = AR1to1s_fullstim_long_CM["ellipse_data"]["xc bottom [um]"][0, singlet_example]
yc_bottom_1to1s = AR1to1s_fullstim_long_CM["ellipse_data"]["yc bottom [um]"][0, singlet_example]

# tracking data
x_tracking_top_1to1d = AR1to1d_fullstim_long["shape_data"]["Xtop"][:, 0, doublet_example] * initial_pixelsize
y_tracking_top_1to1d = AR1to1d_fullstim_long["shape_data"]["Ytop"][:, 0, doublet_example] * initial_pixelsize
x_tracking_bottom_1to1d = AR1to1d_fullstim_long["shape_data"]["Xbottom"][:, 0, doublet_example] * initial_pixelsize
y_tracking_bottom_1to1d = AR1to1d_fullstim_long["shape_data"]["Ybottom"][:, 0, doublet_example] * initial_pixelsize

x_tracking_top_1to1s = AR1to1s_fullstim_long["shape_data"]["Xtop"][:, 0, singlet_example] * initial_pixelsize  # convert to µm
y_tracking_top_1to1s = AR1to1s_fullstim_long["shape_data"]["Ytop"][:, 0, singlet_example] * initial_pixelsize
x_tracking_bottom_1to1s = AR1to1s_fullstim_long["shape_data"]["Xbottom"][:, 0, singlet_example] * initial_pixelsize
y_tracking_bottom_1to1s = AR1to1s_fullstim_long["shape_data"]["Ybottom"][:, 0, singlet_example] * initial_pixelsize

# tangent data
tx_topleft_1to1d = AR1to1d_fullstim_long_CM["tangent_data"]["tx top left"][0, doublet_example] * initial_pixelsize  # convert to µm
ty_topleft_1to1d = AR1to1d_fullstim_long_CM["tangent_data"]["ty top left"][0, doublet_example] * initial_pixelsize
xc_topleft_1to1d = AR1to1d_fullstim_long_CM["tangent_data"]["xTouch top left"][0, doublet_example] * initial_pixelsize
yc_topleft_1to1d = AR1to1d_fullstim_long_CM["tangent_data"]["yTouch top left"][0, doublet_example] * initial_pixelsize

tx_topright_1to1d = AR1to1d_fullstim_long_CM["tangent_data"]["tx top right"][0, doublet_example] * initial_pixelsize
ty_topright_1to1d = AR1to1d_fullstim_long_CM["tangent_data"]["ty top right"][0, doublet_example] * initial_pixelsize
xc_topright_1to1d = AR1to1d_fullstim_long_CM["tangent_data"]["xTouch top right"][0, doublet_example] * initial_pixelsize
yc_topright_1to1d = AR1to1d_fullstim_long_CM["tangent_data"]["yTouch top right"][0, doublet_example] * initial_pixelsize

tx_bottomleft_1to1d = AR1to1d_fullstim_long_CM["tangent_data"]["tx bottom left"][0, doublet_example] * initial_pixelsize
ty_bottomleft_1to1d = AR1to1d_fullstim_long_CM["tangent_data"]["ty bottom left"][0, doublet_example] * initial_pixelsize
xc_bottomleft_1to1d = AR1to1d_fullstim_long_CM["tangent_data"]["xTouch bottom left"][0, doublet_example] * initial_pixelsize
yc_bottomleft_1to1d = AR1to1d_fullstim_long_CM["tangent_data"]["yTouch bottom left"][0, doublet_example] * initial_pixelsize

tx_bottomright_1to1d = AR1to1d_fullstim_long_CM["tangent_data"]["tx bottom right"][0, doublet_example] * initial_pixelsize
ty_bottomright_1to1d = AR1to1d_fullstim_long_CM["tangent_data"]["ty bottom right"][0, doublet_example] * initial_pixelsize
xc_bottomright_1to1d = AR1to1d_fullstim_long_CM["tangent_data"]["xTouch bottom right"][0, doublet_example] * initial_pixelsize
yc_bottomright_1to1d = AR1to1d_fullstim_long_CM["tangent_data"]["yTouch bottom right"][0, doublet_example] * initial_pixelsize

tx_topleft_1to1s = AR1to1s_fullstim_long_CM["tangent_data"]["tx top left"][0, singlet_example] * initial_pixelsize
ty_topleft_1to1s = AR1to1s_fullstim_long_CM["tangent_data"]["ty top left"][0, singlet_example] * initial_pixelsize
xc_topleft_1to1s = AR1to1s_fullstim_long_CM["tangent_data"]["xTouch top left"][0, singlet_example] * initial_pixelsize
yc_topleft_1to1s = AR1to1s_fullstim_long_CM["tangent_data"]["yTouch top left"][0, singlet_example] * initial_pixelsize

tx_topright_1to1s = AR1to1s_fullstim_long_CM["tangent_data"]["tx top right"][0, singlet_example] * initial_pixelsize
ty_topright_1to1s = AR1to1s_fullstim_long_CM["tangent_data"]["ty top right"][0, singlet_example] * initial_pixelsize
xc_topright_1to1s = AR1to1s_fullstim_long_CM["tangent_data"]["xTouch top right"][0, singlet_example] * initial_pixelsize
yc_topright_1to1s = AR1to1s_fullstim_long_CM["tangent_data"]["yTouch top right"][0, singlet_example] * initial_pixelsize

tx_bottomleft_1to1s = AR1to1s_fullstim_long_CM["tangent_data"]["tx bottom left"][0, singlet_example] * initial_pixelsize
ty_bottomleft_1to1s = AR1to1s_fullstim_long_CM["tangent_data"]["ty bottom left"][0, singlet_example] * initial_pixelsize
xc_bottomleft_1to1s = AR1to1s_fullstim_long_CM["tangent_data"]["xTouch bottom left"][0, singlet_example] * initial_pixelsize
yc_bottomleft_1to1s = AR1to1s_fullstim_long_CM["tangent_data"]["yTouch bottom left"][0, singlet_example] * initial_pixelsize

tx_bottomright_1to1s = AR1to1s_fullstim_long_CM["tangent_data"]["tx bottom right"][0, singlet_example] * initial_pixelsize
ty_bottomright_1to1s = AR1to1s_fullstim_long_CM["tangent_data"]["ty bottom right"][0, singlet_example] * initial_pixelsize
xc_bottomright_1to1s = AR1to1s_fullstim_long_CM["tangent_data"]["xTouch bottom right"][0, singlet_example] * initial_pixelsize
yc_bottomright_1to1s = AR1to1s_fullstim_long_CM["tangent_data"]["yTouch bottom right"][0, singlet_example] * initial_pixelsize

# calculate force amplitudes
T_1to1d = np.sqrt(Tx_1to1d ** 2 + Ty_1to1d ** 2)
T_1to1s = np.sqrt(Tx_1to1s ** 2 + Ty_1to1s ** 2)

# crop force maps and actin images
crop_start = 14
crop_end = 78

Tx_1to1d_crop = Tx_1to1d[crop_start:crop_end, crop_start:crop_end] * 1e-3
Ty_1to1d_crop = Ty_1to1d[crop_start:crop_end, crop_start:crop_end] * 1e-3
T_1to1d_crop = T_1to1d[crop_start:crop_end, crop_start:crop_end] * 1e-3

Tx_1to1s_crop = Tx_1to1s[crop_start:crop_end, crop_start:crop_end] * 1e-3
Ty_1to1s_crop = Ty_1to1s[crop_start:crop_end, crop_start:crop_end] * 1e-3
T_1to1s_crop = T_1to1s[crop_start:crop_end, crop_start:crop_end] * 1e-3

actin_image_1to1d_crop = actin_image_1to1d[crop_start * 8:crop_end * 8, crop_start * 8:crop_end * 8]

actin_image_1to1s_crop = actin_image_1to1s[crop_start * 8:crop_end * 8, crop_start * 8:crop_end * 8]

# remove 0 values from tracking data
x_tracking_top_1to1d = x_tracking_top_1to1d[x_tracking_top_1to1d != 0]
y_tracking_top_1to1d = y_tracking_top_1to1d[y_tracking_top_1to1d != 0]
x_tracking_bottom_1to1d = x_tracking_bottom_1to1d[x_tracking_bottom_1to1d != 0]
y_tracking_bottom_1to1d = y_tracking_bottom_1to1d[y_tracking_bottom_1to1d != 0]

x_tracking_top_1to1s = x_tracking_top_1to1s[x_tracking_top_1to1s != 0]
y_tracking_top_1to1s = y_tracking_top_1to1s[y_tracking_top_1to1s != 0]
x_tracking_bottom_1to1s = x_tracking_bottom_1to1s[x_tracking_bottom_1to1s != 0]
y_tracking_bottom_1to1s = y_tracking_bottom_1to1s[y_tracking_bottom_1to1s != 0]

# %% plot figure 2C, actin image with force map
# set up plot parameters
# ******************************************************************************************************************************************
n = 3  # every nth arrow will be plotted
pmax = 2  # in kPa
axtitle = 'kPa'  # unit of colorbar
suptitle = 'Traction forces'  # title of plot
x_end = np.shape(T_1to1d_crop)[1]  # create x- and y-axis for plotting maps
y_end = np.shape(T_1to1d_crop)[0]
extent = [0, x_end * pixelsize, 0, y_end * pixelsize]
colors = [colors_parent[2], colors_parent[0]]
xq, yq = np.meshgrid(np.linspace(0, extent[1], x_end), np.linspace(0, extent[3], y_end))  # create mesh for vectorplot
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(5, 2), gridspec_kw={'width_ratios': [5, 5, 1]})  # create figure and axes
plt.subplots_adjust(wspace=-0.5, hspace=0)  # adjust space in between plots
# ******************************************************************************************************************************************


# Set up plot parameters for first panel
#######################################################################################################
actin_image = actin_image_1to1d_crop
ax = axes[0]
x = xq[::n, ::n].flatten()
y = yq[::n, ::n].flatten()
Tx = Tx_1to1d_crop[::n, ::n].flatten()
Ty = Ty_1to1d_crop[::n, ::n].flatten()
T = T_1to1d_crop[::n, ::n].flatten()
a_top = a_top_1to1d
b_top = b_top_1to1d
a_bottom = a_bottom_1to1d
b_bottom = b_bottom_1to1d
tx_topleft = tx_topleft_1to1d
ty_topleft = -ty_topleft_1to1d
tx_topright = tx_topright_1to1d
ty_topright = -ty_topright_1to1d
tx_bottomleft = tx_bottomleft_1to1d
ty_bottomleft = -ty_bottomleft_1to1d
tx_bottomright = tx_bottomright_1to1d
ty_bottomright = -ty_bottomright_1to1d
# actin images, ellipse, tracking and tangent data don't live on the same coordinate system, so I have to transform
# the coordinates of the ellipse, tracking and tangent data
xc_top = xc_top_1to1d - (125 - x_end) / 2 * pixelsize  # 125 corresponds to the original width of the actin images before cropping
yc_top = -yc_top_1to1d + extent[1] + (125 - y_end) / 2 * pixelsize
xc_bottom = xc_bottom_1to1d - (125 - x_end) / 2 * pixelsize
yc_bottom = -yc_bottom_1to1d + extent[1] + (125 - y_end) / 2 * pixelsize
x_tracking_top = x_tracking_top_1to1d - (125 - x_end) / 2 * pixelsize
y_tracking_top = -y_tracking_top_1to1d + extent[1] + (125 - y_end) / 2 * pixelsize
x_tracking_bottom = x_tracking_bottom_1to1d - (125 - x_end) / 2 * pixelsize
y_tracking_bottom = -y_tracking_bottom_1to1d + extent[1] + (125 - y_end) / 2 * pixelsize
xc_topleft = xc_topleft_1to1d - (125 - x_end) / 2 * pixelsize
yc_topleft = -yc_topleft_1to1d + extent[1] + (125 - y_end) / 2 * pixelsize
xc_topright = xc_topright_1to1d - (125 - x_end) / 2 * pixelsize
yc_topright = -yc_topright_1to1d + extent[1] + (125 - y_end) / 2 * pixelsize
xc_bottomleft = xc_bottomleft_1to1d - (125 - x_end) / 2 * pixelsize
yc_bottomleft = -yc_bottomleft_1to1d + extent[1] + (125 - y_end) / 2 * pixelsize
xc_bottomright = xc_bottomright_1to1d - (125 - x_end) / 2 * pixelsize
yc_bottomright = -yc_bottomright_1to1d + extent[1] + (125 - y_end) / 2 * pixelsize

plot_actin_image_forces_ellipses_tracking_tangents(actin_image, x, y, Tx, Ty, T, a_top, b_top, a_bottom, b_bottom,
                                                   tx_topleft, ty_topleft, tx_topright, ty_topright, tx_bottomleft, ty_bottomleft,
                                                   tx_bottomright, ty_bottomright,
                                                   xc_top, yc_top, xc_bottom, yc_bottom, x_tracking_top, y_tracking_top, x_tracking_bottom,
                                                   y_tracking_bottom,
                                                   xc_topleft, yc_topleft, xc_topright, yc_topright, xc_bottomleft, yc_bottomleft,
                                                   xc_bottomright, yc_bottomright, ax, colors, extent)

# Set up plot parameters for second panel
#######################################################################################################
actin_image = actin_image_1to1s_crop
ax = axes[1]
x = xq[::n, ::n].flatten()
y = yq[::n, ::n].flatten()
Tx = Tx_1to1s_crop[::n, ::n].flatten()
Ty = Ty_1to1s_crop[::n, ::n].flatten()
T = T_1to1s_crop[::n, ::n].flatten()
a_top = a_top_1to1s
b_top = b_top_1to1s
a_bottom = a_bottom_1to1s
b_bottom = b_bottom_1to1s
tx_topleft = tx_topleft_1to1s
ty_topleft = -ty_topleft_1to1s
tx_topright = tx_topright_1to1s
ty_topright = -ty_topright_1to1s
tx_bottomleft = tx_bottomleft_1to1s
ty_bottomleft = -ty_bottomleft_1to1s
tx_bottomright = tx_bottomright_1to1s
ty_bottomright = -ty_bottomright_1to1s
# actin images, ellipse, tracking and tangent data don't live on the same coordinate system, so I have to transform
# the coordinates of the ellipse, tracking and tangent data
xc_top = xc_top_1to1s - (125 - x_end) / 2 * pixelsize  # 125 corresponds to the original width of the actin images before cropping
yc_top = -yc_top_1to1s + extent[1] + (125 - y_end) / 2 * pixelsize
xc_bottom = xc_bottom_1to1s - (125 - x_end) / 2 * pixelsize
yc_bottom = -yc_bottom_1to1s + extent[1] + (125 - y_end) / 2 * pixelsize
x_tracking_top = x_tracking_top_1to1s - (125 - x_end) / 2 * pixelsize
y_tracking_top = -y_tracking_top_1to1s + extent[1] + (125 - y_end) / 2 * pixelsize
x_tracking_bottom = x_tracking_bottom_1to1s - (125 - x_end) / 2 * pixelsize
y_tracking_bottom = -y_tracking_bottom_1to1s + extent[1] + (125 - y_end) / 2 * pixelsize
xc_topleft = xc_topleft_1to1s - (125 - x_end) / 2 * pixelsize
yc_topleft = -yc_topleft_1to1s + extent[1] + (125 - y_end) / 2 * pixelsize
xc_topright = xc_topright_1to1s - (125 - x_end) / 2 * pixelsize
yc_topright = -yc_topright_1to1s + extent[1] + (125 - y_end) / 2 * pixelsize
xc_bottomleft = xc_bottomleft_1to1s - (125 - x_end) / 2 * pixelsize
yc_bottomleft = -yc_bottomleft_1to1s + extent[1] + (125 - y_end) / 2 * pixelsize
xc_bottomright = xc_bottomright_1to1s - (125 - x_end) / 2 * pixelsize
yc_bottomright = -yc_bottomright_1to1s + extent[1] + (125 - y_end) / 2 * pixelsize

sm = plot_actin_image_forces_ellipses_tracking_tangents(actin_image, x, y, Tx, Ty, T, a_top, b_top, a_bottom, b_bottom,
                                                        tx_topleft, ty_topleft, tx_topright, ty_topright, tx_bottomleft, ty_bottomleft,
                                                        tx_bottomright, ty_bottomright,
                                                        xc_top, yc_top, xc_bottom, yc_bottom, x_tracking_top, y_tracking_top,
                                                        x_tracking_bottom,
                                                        y_tracking_bottom,
                                                        xc_topleft, yc_topleft, xc_topright, yc_topright, xc_bottomleft, yc_bottomleft,
                                                        xc_bottomright, yc_bottomright, ax, colors, extent)

# add colorbar
cbar = plt.colorbar(sm, ax=axes[2])
cbar.ax.set_title(axtitle)
axes[2].axis('off')

fig.savefig(figfolder + 'C.png', dpi=300, bbox_inches="tight")
fig.savefig(figfolder + 'C.svg', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 2D, Correlation plots of sigma_MSM and sigma_CM

# define plot parameters that are valid for the whole figure
# ******************************************************************************************************************************************
colors = [colors_parent[1], colors_parent[2]]  # defines colors for scatterplot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4.5, 2))  # create figure instance
plt.subplots_adjust(wspace=0.3, hspace=0.3)  # adjust space in between plots

# Set up plot parameters for first panel
#######################################################################################################
x = 'sigma_xx_baseline'
y = 'sigma_x_baseline'
hue = 'keys'
ax = axes[0]
xmin = 0
xmax = 20
ymin = 0
ymax = 20
xticks = np.arange(0, 20.1, 5)
yticks = np.arange(0, 20.1, 5)
xlabel = '$\mathrm{\sigma_{x, MSM}}$'
ylabel = '$\mathrm{\sigma_{x, CM}}$'

corr, p = make_correlationplotsplots(x, y, hue, df, ax, xmin, xmax, ymin, ymax, xticks, yticks, xlabel, ylabel, colors)


# annotate pearson R and p-value
plt.text(0.6 * xmax, 0.075 * ymax, 'R = ' + str(round(corr, 2)))

# Set up plot parameters for second panel
#######################################################################################################
x = 'sigma_yy_baseline'
y = 'sigma_y_baseline'
hue = 'keys'
ax = axes[1]
xmin = 0
xmax = 10
ymin = 0
ymax = 10
xticks = np.arange(0, 10.1, 5)
yticks = np.arange(0, 10.1, 5)
xlabel = '$\mathrm{\sigma_{y, MSM}}$'
ylabel = '$\mathrm{\sigma_{y, CM}}$'

corr, p = make_correlationplotsplots(x, y, hue, df, ax, xmin, xmax, ymin, ymax, xticks, yticks, xlabel, ylabel, colors)


# annotate pearson R and p-value
plt.text(0.6 * xmax, 0.075 * ymax, 'R = ' + str(round(corr, 2)))

plt.suptitle('MSM stresses vs. CM surface tensions')

plt.savefig(figfolder + 'D.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'D.svg', dpi=300, bbox_inches="tight")
plt.show()


# %% plot figure 2E, line tension and force of adherent fiber

# define plot parameters that are valid for the whole figure
# ******************************************************************************************************************************************
colors = [colors_parent[1], colors_parent[2]]  # defines colors
sns.set_palette(sns.color_palette(colors))  # sets colors
box_pairs = [('AR1to1d', 'AR1to1s')]  # which groups to perform statistical test on
xticklabels = ['Doublet', 'Singlet']  # which labels to put on x-axis
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4, 2))  # create figure instance
plt.subplots_adjust(wspace=0.4, hspace=0.25)  # adjust space in between plots
# ******************************************************************************************************************************************


# Set up plot parameters for first panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'line tension baseline [nN]'  # variable that goes on the y-axis
ax = axes[0]  # define on which axis the plot goes
ymin = -100  # minimum value on y-axis
ymax = 500  # maximum value on y-axis
yticks = np.arange(0, 501, 100)  # define where to put major ticks on y-axis
stat_annotation_offset = 0.02  # vertical offset of statistical annotation
ylabel = '$\mathrm{\lambda}$ [nN]'  # which label to put on y-axis
title = 'Line tension'  # title of plot

# make plots
make_box_and_swarmplots_with_test(x, y, df, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title, colors)

# Set up plot parameters for second panel
#######################################################################################################
x = 'keys'  # variable by which to group the data
y = 'f adherent baseline [nN]'  # variable that goes on the y-axis
ax = axes[1]  # define on which axis the plot goes
ymin = -100  # minimum value on y-axis
ymax = 500  # maximum value on y-axis
yticks = np.arange(0, 501, 100)  # define where to put major ticks on y-axis
stat_annotation_offset = 0.5  # vertical offset of statistical annotation
ylabel = '$\mathrm{f_a}$ [nN]'  # which label to put on y-axis
title = 'Force of adherent fiber'  # title of plot

# make plots
make_box_and_swarmplots_with_test(x, y, df, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title, colors)

# # save plot to file
plt.savefig(figfolder + 'E.png', dpi=300, bbox_inches="tight")
plt.savefig(figfolder + 'E.svg', dpi=300, bbox_inches="tight")
plt.show()

