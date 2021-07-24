import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statannot
from scipy.stats import pearsonr
import matplotlib.image as mpimg

# mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 8

# define some colors for the plots
colors_parent = ['#026473', '#E3CC69', '#77C8A6', '#D96248']

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def plot_actin_image_forces_ellipses_tracking_tangents(actin_image, x, y, Tx, Ty, T, a_top, b_top, a_bottom, b_bottom,
                                                       tx_topleft, ty_topleft, tx_topright, ty_topright, tx_bottomleft, ty_bottomleft,
                                                       tx_bottomright, ty_bottomright,
                                                       xc_top, yc_top, xc_bottom, yc_bottom, x_tracking_top, y_tracking_top,
                                                       x_tracking_bottom, y_tracking_bottom,
                                                       xc_topleft, yc_topleft, xc_topright, yc_topright, xc_bottomleft, yc_bottomleft,
                                                       xc_bottomright, yc_bottomright, ax, extent):
    norm = mpl.colors.Normalize()
    norm.autoscale([0, 2])
    colormap = mpl.cm.turbo
    sm = mpl.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])

    # plot actin image
    ax.imshow(actin_image, cmap=plt.get_cmap("Greys"), interpolation="bilinear", extent=extent, aspect=(1))

    # plot forces
    ax.quiver(x, y, Tx, Ty, angles='xy', scale_units='xy', scale=0.2, color=colormap(norm(T)))

    # plot ellipses
    t = np.linspace(0, 2 * np.pi, 100)
    ax.plot(xc_top + a_top * np.cos(t), yc_top + b_top * np.sin(t), color=colors_parent[2], linewidth=2)
    t = np.linspace(0, 2 * np.pi, 100)
    ax.plot(xc_bottom + a_bottom * np.cos(t), yc_bottom + b_bottom * np.sin(t), color=colors_parent[2], linewidth=2)

    # plot tracking data
    ax.plot(x_tracking_top[::2], y_tracking_top[::2],
             color=colors_parent[0], marker='o', markerfacecolor='none', markersize=4, markeredgewidth=1, linestyle='none')
    ax.plot(x_tracking_bottom[::2], y_tracking_bottom[::2],
             color=colors_parent[0], marker='o', markerfacecolor='none', markersize=4, markeredgewidth=1, linestyle='none')

    # plot tangents
    ax.plot([xc_topleft, xc_topleft + 150 * tx_topleft], [yc_topleft, yc_topleft + 150 * ty_topleft],
            color='white', linewidth=2, linestyle=':')
    ax.plot([xc_topright, xc_topright + 150 * tx_topright], [yc_topright, yc_topright + 150 * ty_topright],
            color='white', linewidth=2, linestyle=':')
    ax.plot([xc_bottomleft, xc_bottomleft + 150 * tx_bottomleft], [yc_bottomleft, yc_bottomleft + 150 * ty_bottomleft],
            color='white', linewidth=2, linestyle=':')
    ax.plot([xc_bottomright, xc_bottomright + 150 * tx_bottomright], [yc_bottomright, yc_bottomright + 150 * ty_bottomright],
            color='white', linewidth=2, linestyle=':')

    ax.set_xlim([-0.1 * extent[1], 1.1 * extent[1]])
    ax.set_ylim([-0.1 * extent[3], 1.1 * extent[3]])

    ax.axis('off')

    return sm
def make_actin_ellipse_plots(data_TFM, data_CM, cell, frame,  pixelsize, initial_pixelsize, title):
    Tx = data_TFM["TFM_data"]["Tx"][:, :, 0, cell]
    Ty = data_TFM["TFM_data"]["Ty"][:, :, 0, cell]

    # actin images
    actin_image_path = folder + title + "/actin_images/cell" + str(cell) + "frame0.png"
    actin_image = rgb2gray(mpimg.imread(actin_image_path))

    # ellipse data
    a_top = data_CM["ellipse_data"]["a top [um]"][0, cell]
    b_top = data_CM["ellipse_data"]["b top [um]"][0, cell]
    xc_top = data_CM["ellipse_data"]["xc top [um]"][0, cell]
    yc_top = data_CM["ellipse_data"]["yc top [um]"][0, cell]
    a_bottom = data_CM["ellipse_data"]["a bottom [um]"][0, cell]
    b_bottom = data_CM["ellipse_data"]["b bottom [um]"][0, cell]
    xc_bottom = data_CM["ellipse_data"]["xc bottom [um]"][0, cell]
    yc_bottom = data_CM["ellipse_data"]["yc bottom [um]"][0, cell]

    # tracking data
    x_tracking_top = data_TFM["shape_data"]["Xtop"][:, 0, cell] * initial_pixelsize
    y_tracking_top = data_TFM["shape_data"]["Ytop"][:, 0, cell] * initial_pixelsize
    x_tracking_bottom = data_TFM["shape_data"]["Xbottom"][:, 0, cell] * initial_pixelsize
    y_tracking_bottom = data_TFM["shape_data"]["Ybottom"][:, 0, cell] * initial_pixelsize

    # tangent data
    tx_topleft = data_CM["tangent_data"]["tx top left"][0, cell] * initial_pixelsize  # convert to µm
    ty_topleft = data_CM["tangent_data"]["ty top left"][0, cell] * initial_pixelsize
    xc_topleft = data_CM["tangent_data"]["xTouch top left"][0, cell] * initial_pixelsize
    yc_topleft = data_CM["tangent_data"]["yTouch top left"][0, cell] * initial_pixelsize

    tx_topright = data_CM["tangent_data"]["tx top right"][0, cell] * initial_pixelsize
    ty_topright = data_CM["tangent_data"]["ty top right"][0, cell] * initial_pixelsize
    xc_topright = data_CM["tangent_data"]["xTouch top right"][0, cell] * initial_pixelsize
    yc_topright = data_CM["tangent_data"]["yTouch top right"][0, cell] * initial_pixelsize

    tx_bottomleft = data_CM["tangent_data"]["tx bottom left"][0, cell] * initial_pixelsize
    ty_bottomleft = data_CM["tangent_data"]["ty bottom left"][0, cell] * initial_pixelsize
    xc_bottomleft = data_CM["tangent_data"]["xTouch bottom left"][0, cell] * initial_pixelsize
    yc_bottomleft = data_CM["tangent_data"]["yTouch bottom left"][0, cell] * initial_pixelsize

    tx_bottomright = data_CM["tangent_data"]["tx bottom right"][0, cell] * initial_pixelsize
    ty_bottomright = data_CM["tangent_data"]["ty bottom right"][0, cell] * initial_pixelsize
    xc_bottomright = data_CM["tangent_data"]["xTouch bottom right"][0, cell] * initial_pixelsize
    yc_bottomright = data_CM["tangent_data"]["yTouch bottom right"][0, cell] * initial_pixelsize


    # calculate force amplitudes
    T = np.sqrt(Tx ** 2 + Ty ** 2)

    # crop force maps and actin images
    crop_start = 14
    crop_end = 78

    Tx_crop = Tx[crop_start:crop_end, crop_start:crop_end] * 1e-3
    Ty_crop = Ty[crop_start:crop_end, crop_start:crop_end] * 1e-3
    T_crop = T[crop_start:crop_end, crop_start:crop_end] * 1e-3

    actin_image_crop = actin_image[crop_start * 8:crop_end * 8, crop_start * 8:crop_end * 8]

    # remove 0 values from tracking data
    x_tracking_top = x_tracking_top[x_tracking_top != 0]
    y_tracking_top = y_tracking_top[y_tracking_top != 0]
    x_tracking_bottom = x_tracking_bottom[x_tracking_bottom != 0]
    y_tracking_bottom = y_tracking_bottom[y_tracking_bottom != 0]

    x_end = np.shape(Tx_crop)[1]
    y_end = np.shape(Tx_crop)[0]

    # actin images, ellipse, tracking and tangent data don't live on the same coordinate system, so I have to transform
    # the coordinates of the ellipse, tracking and tangent data
    ty_topleft = -ty_topleft
    ty_topright = -ty_topright
    ty_bottomleft = -ty_bottomleft
    ty_bottomright = -ty_bottomright
    xc_top = xc_top - (125 - x_end) / 2 * pixelsize  # 125 corresponds to the original width of the actin images before cropping
    yc_top = -yc_top + x_end * pixelsize + (125 - y_end) / 2 * pixelsize
    xc_bottom = xc_bottom - (125 - x_end) / 2 * pixelsize
    yc_bottom = -yc_bottom + x_end * pixelsize + (125 - y_end) / 2 * pixelsize
    x_tracking_top = x_tracking_top - (125 - x_end) / 2 * pixelsize
    y_tracking_top = -y_tracking_top + x_end * pixelsize + (125 - y_end) / 2 * pixelsize
    x_tracking_bottom = x_tracking_bottom - (125 - x_end) / 2 * pixelsize
    y_tracking_bottom = -y_tracking_bottom + x_end * pixelsize + (125 - y_end) / 2 * pixelsize
    xc_topleft = xc_topleft - (125 - x_end) / 2 * pixelsize
    yc_topleft = -yc_topleft + x_end * pixelsize + (125 - y_end) / 2 * pixelsize
    xc_topright = xc_topright - (125 - x_end) / 2 * pixelsize
    yc_topright = -yc_topright + x_end * pixelsize + (125 - y_end) / 2 * pixelsize
    xc_bottomleft = xc_bottomleft - (125 - x_end) / 2 * pixelsize
    yc_bottomleft = -yc_bottomleft + x_end * pixelsize + (125 - y_end) / 2 * pixelsize
    xc_bottomright = xc_bottomright - (125 - x_end) / 2 * pixelsize
    yc_bottomright = -yc_bottomright + x_end * pixelsize + (125 - y_end) / 2 * pixelsize

    # set up plot parameters
    n = 3  # every nth arrow will be plotted
    pmax = 2  # in kPa
    axtitle = 'kPa'  # unit of colorbar
    extent = [0, x_end * pixelsize, 0, y_end * pixelsize]
    xq, yq = np.meshgrid(np.linspace(0, extent[1], x_end), np.linspace(0, extent[3], y_end))

    x = xq[::n, ::n].flatten()
    y = yq[::n, ::n].flatten()
    Tx = Tx_crop[::n, ::n].flatten()
    Ty = Ty_crop[::n, ::n].flatten()
    T = T_crop[::n, ::n].flatten()

    fig, ax = plt.subplots(figsize=(3, 3))

    sm=plot_actin_image_forces_ellipses_tracking_tangents(actin_image_crop, x, y, Tx, Ty, T, a_top, b_top, a_bottom, b_bottom,
                                                       tx_topleft, ty_topleft, tx_topright, ty_topright, tx_bottomleft, ty_bottomleft,
                                                       tx_bottomright, ty_bottomright,
                                                       xc_top, yc_top, xc_bottom, yc_bottom, x_tracking_top, y_tracking_top, x_tracking_bottom,
                                                       y_tracking_bottom,
                                                       xc_topleft, yc_topleft, xc_topright, yc_topright, xc_bottomleft, yc_bottomleft,
                                                       xc_bottomright, yc_bottomright, ax, extent)

    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.set_title(axtitle)
    ax.axis('off')

    plt.show()
    actin_image__result_path = folder + title + "/actin_images/_ellipses_cell" + str(cell) + "frame" + str(frame) + ".png"
    fig.savefig(actin_image__result_path, dpi=300, bbox_inches="tight")



def main(folder, title, datafile):
    data_TFM = pickle.load(open(folder + "analysed_data/" + datafile + ".dat", "rb"))
    data_CM = pickle.load(open(folder + "analysed_data/" + datafile + "_CM.dat", "rb"))

    # load and prepare data for plotting movies
    pixelsize = 0.864  # in µm
    initial_pixelsize = 0.108  # in µm
    frame_end = data_TFM["TFM_data"]["Dx"].shape[2]
    cell_end = data_TFM["TFM_data"]["Dx"].shape[3]

    for cell in range(cell_end):
        print("cell" + str(cell))
        for frame in range(frame_end):
            make_actin_ellipse_plots(data_TFM, data_CM, cell, frame, pixelsize, initial_pixelsize, title)

if __name__ == "__main__":
    folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"

    main(folder, "AR1to1 doublets full stim long", "AR1to1d_fullstim_long")
    main(folder, "AR1to1 singlets full stim long", "AR1to1s_fullstim_long")
    main(folder, "AR1to1 doublets full stim short", "AR1to1d_fullstim_short")
    main(folder, "AR1to1 singlets full stim short", "AR1to1s_fullstim_short")

    main(folder, "AR1to2 doublets half stim", "AR1to2d_halfstim")
    main(folder, "AR1to1 doublets half stim", "AR1to1d_halfstim")
    main(folder, "AR1to1 singlets half stim", "AR1to1s_halfstim")
    main(folder, "AR2to1 doublets half stim", "AR2to1d_halfstim")