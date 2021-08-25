# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 09:33:36 2021

@author: Artur Ruppel
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from lmfit import Model


def analyse_tfm_data(folder, stressmappixelsize):
    Dx = np.load(folder + "/Dx.npy")
    Dy = np.load(folder + "/Dy.npy")
    Tx = np.load(folder + "/Tx.npy")
    Ty = np.load(folder + "/Ty.npy")

    # calculate force amplitude
    T = np.sqrt(Tx ** 2 + Ty ** 2)

    x_end = np.shape(Dx)[1]
    y_end = np.shape(Dx)[0]
    t_end = np.shape(Dx)[2]
    cell_end = np.shape(Dx)[3]
    y_half = np.rint(y_end / 2).astype(int)
    x_half = np.rint(x_end / 2).astype(int)

    Es_density = 0.5 * (stressmappixelsize ** 2) * (Tx * Dx + Ty * Dy)

    # average over whole cell and then over left and right half
    Es = np.nansum(Es_density, axis=(0, 1))  # maps are coming from matlab calculations where x and y-axes are inverted
    Es_left = np.nansum(Es_density[:, 0:x_half, :, :], axis=(0, 1))
    Es_right = np.nansum(Es_density[:, x_half:x_end, :, :], axis=(0, 1))

    # average over first twenty frames before photoactivation
    Es_baseline = np.nanmean(Es[0:20, :], axis=0)

    # normalize strain energy by first substracting the baseline for each cell and then dividing by the average baseline
    relEs = (Es - np.nanmean(Es[0:20], axis=0)) / np.nanmean(Es[0:20], axis=(0, 1))

    # calculate total force in x- and y-direction
    Fx = np.nansum(abs(Tx), axis=(0, 1)) * (stressmappixelsize ** 2)
    Fy = np.nansum(abs(Ty), axis=(0, 1)) * (stressmappixelsize ** 2)

    # calculate force angle
    force_angle = np.arctan(np.divide(Fy, Fx)) * 360 / (2 * np.pi)
    force_angle_baseline = np.nanmean(force_angle[0:20, :], axis=0)

    # calculate cell-cell force
    Fx_left = np.nansum(Tx[:, 0:x_half, :, :], axis=(0, 1)) * (stressmappixelsize ** 2)
    Fx_right = np.nansum(Tx[:, x_half:x_end, :, :], axis=(0, 1)) * (stressmappixelsize ** 2)
    F_cellcell = Fx_left - Fx_right

    # calculate sigma_x for contour model by integrating the x-forces in the center of the TFM map and dividing by the height of the window
    sigmawindow = 24  # The x componenent of the force is calculated on a window of this width in pixel in the center of the TFM map
    # The x component of the force is summed up on a window from the edge minus sigmadistance_from_border to the center minus sigmadistance_from_center
    sigmadistance_from_center = 18
    sigmadistance_from_border = 18
    x_window = x_half - sigmadistance_from_center - sigmadistance_from_border

    # the other aspect ratios require different window parameters
    if "1to2" in folder:
        sigmawindow = 12  # The x componenent of the force is calculated on a window of this width in pixel in the center of the TFM map
        # The x component of the force is summed up on a window from the edge minus sigmadistance_from_border to the center minus sigmadistance_from_center
        sigmadistance_from_center = 20
        sigmadistance_from_border = 4
        x_window = x_half - sigmadistance_from_center - sigmadistance_from_border

    if "2to1" in folder:
        sigmawindow = 36  # The x componenent of the force is calculated on a window of this width in pixel in the center of the TFM map
        # The x component of the force is summed up on a window from the edge minus sigmadistance_from_border to the center minus sigmadistance_from_center
        sigmadistance_from_center = 8
        sigmadistance_from_border = 24
        x_window = x_half - sigmadistance_from_center - sigmadistance_from_border

    # plop = Tx[:,:,0,0]
    # plop[y_half - int(sigmawindow / 2):y_half + int(sigmawindow / 2), sigmadistance_from_border:x_half - sigmadistance_from_center] = 10
    # plt.imshow(plop)
    # plt.show()

    sigma_x_left = \
        np.nanmean(
            Tx[y_half - int(sigmawindow / 2):y_half + int(sigmawindow / 2), sigmadistance_from_border:x_half - sigmadistance_from_center, :,
            :], axis=(0, 1)) / x_window * stressmappixelsize
    sigma_x_right = \
        np.nansum(Tx[y_half - int(sigmawindow / 2):y_half + int(sigmawindow / 2),
                  x_half + sigmadistance_from_center:x_end - sigmadistance_from_border, :, :],
                  axis=(0, 1)) / x_window * stressmappixelsize

    sigma_x = (sigma_x_left - sigma_x_right) / 2

    # in some rare cases the surface tension value is negative due to high noise in the TFM map. This messes up further analysis so I set it to a small, positive value
    for c in range(sigma_x.shape[0]):
        for t in range(sigma_x.shape[1]):
            if sigma_x[c, t] < 0:
                sigma_x[c, t] = 0.1e-3

    # average over first twenty frames before photoactivation
    sigma_x_baseline = np.nanmean(sigma_x[0:20, :], axis=0)

    # calculate relative energy increase
    REI = relEs[32, :] - relEs[20, :]

    # calculate corner averages
    Fx_topleft = np.zeros((t_end, cell_end))
    Fx_topright = np.zeros((t_end, cell_end))
    Fx_bottomright = np.zeros((t_end, cell_end))
    Fx_bottomleft = np.zeros((t_end, cell_end))
    Fy_topleft = np.zeros((t_end, cell_end))
    Fy_topright = np.zeros((t_end, cell_end))
    Fy_bottomright = np.zeros((t_end, cell_end))
    Fy_bottomleft = np.zeros((t_end, cell_end))

    # find force peak and sum up all values within radius r
    r = 12  # ~10 µM
    x = np.arange(0, x_half)
    y = np.arange(0, y_half)
    for cell in np.arange(cell_end):
        # find peak in first frame
        T_topleft = T[0:y_half, 0:x_half, 0, cell]
        T_topleft_max = np.where(T_topleft == np.amax(T_topleft))

        T_topright = T[0:y_half, x_half:x_end, 0, cell]
        T_topright_max = np.where(T_topright == np.amax(T_topright))

        T_bottomleft = T[y_half:y_end, 0:x_half, 0, cell]
        T_bottomleft_max = np.where(T_bottomleft == np.amax(T_bottomleft))

        T_bottomright = T[y_half:y_end, x_half:x_end, 0, cell]
        T_bottomright_max = np.where(T_bottomright == np.amax(T_bottomright))
        for t in np.arange(t_end):
            mask_topleft = (x[np.newaxis, :] - T_topleft_max[1]) ** 2 + (
                    y[:, np.newaxis] - T_topleft_max[0]) ** 2 < r ** 2
            Fx_topleft[t, cell] = np.nansum(Tx[0:y_half, 0:x_half, t, cell] * mask_topleft) * stressmappixelsize ** 2\
                                  - sigma_x[t, cell] * r * stressmappixelsize   # correcting for the contribution of the surface tension
            Fy_topleft[t, cell] = np.nansum(Ty[0:y_half, 0:x_half, t, cell] * mask_topleft) * stressmappixelsize ** 2

            mask_topright = (x[np.newaxis, :] - T_topright_max[1]) ** 2 + (
                    y[:, np.newaxis] - T_topright_max[0]) ** 2 < r ** 2
            Fx_topright[t, cell] = np.nansum(
                Tx[0:y_half, x_half:x_end, t, cell] * mask_topright) * stressmappixelsize ** 2\
                                  - sigma_x[t, cell] * r * stressmappixelsize   # correcting for the contribution of the surface tension
            Fy_topright[t, cell] = np.nansum(
                Ty[0:y_half, x_half:x_end, t, cell] * mask_topright) * stressmappixelsize ** 2

            mask_bottomleft = (x[np.newaxis, :] - T_bottomleft_max[1]) ** 2 + (
                    y[:, np.newaxis] - T_bottomleft_max[0]) ** 2 < r ** 2
            Fx_bottomleft[t, cell] = np.nansum(
                Tx[y_half:y_end, 0:x_half, t, cell] * mask_bottomleft) * stressmappixelsize ** 2\
                                  - sigma_x[t, cell] * r * stressmappixelsize   # correcting for the contribution of the surface tension
            Fy_bottomleft[t, cell] = np.nansum(
                Ty[y_half:y_end, 0:x_half, t, cell] * mask_bottomleft) * stressmappixelsize ** 2

            mask_bottomright = (x[np.newaxis, :] - T_bottomright_max[1]) ** 2 + (
                    y[:, np.newaxis] - T_bottomright_max[0]) ** 2 < r ** 2
            Fx_bottomright[t, cell] = np.nansum(
                Tx[y_half:y_end, x_half:x_end, t, cell] * mask_bottomright) * stressmappixelsize ** 2\
                                  - sigma_x[t, cell] * r * stressmappixelsize   # correcting for the contribution of the surface tension
            Fy_bottomright[t, cell] = np.nansum(
                Ty[y_half:y_end, x_half:x_end, t, cell] * mask_bottomright) * stressmappixelsize ** 2

    data = {"Dx": Dx, "Dy": Dy, "Tx": Tx, "Ty": Ty, "sigma_x": sigma_x, "sigma_x_baseline": sigma_x_baseline,
            "Fx_topleft": Fx_topleft, "Fx_topright": Fx_topright, "Fx_bottomright": Fx_bottomright,
            "Fx_bottomleft": Fx_bottomleft,
            "Fy_topleft": Fy_topleft, "Fy_topright": Fy_topright, "Fy_bottomright": Fy_bottomright,
            "Fy_bottomleft": Fy_bottomleft,
            "Es": Es, "Es_left": Es_left, "Es_right": Es_right, "Es_baseline": Es_baseline,
            "relEs": relEs, "REI": REI,
            "Fx": Fx, "Fy": Fy, "force_angle": force_angle, "force_angle_baseline": force_angle_baseline,
            "F_cellcell": F_cellcell}

    return data


def analyse_msm_data(folder):
    sigma_xx = np.load(folder + "/sigma_xx.npy")
    sigma_yy = np.load(folder + "/sigma_yy.npy")
    masks = np.load(folder + "/mask.npy")

    # replace 0 with NaN to not mess up average calculations
    sigma_xx[sigma_xx == 0] = 'nan'
    sigma_yy[sigma_yy == 0] = 'nan'

    # calculate normal stress
    sigma_normal = (sigma_xx + sigma_yy) / 2

    # calculate stress profile along x-axis. 
    # I cut out the borders by multiplying with masks that describes the cell contour exactly to mitigate boundary effects
    sigma_normal_x_profile = np.nanmean(sigma_normal * masks, axis=0)

    x_end = np.shape(sigma_xx)[1]
    x_half = np.rint(x_end / 2).astype(int)

    # average over whole cell and then over left and right half
    sigma_xx_average = np.nanmean(sigma_xx, axis=(0, 1))  # maps are coming from matlab calculations where x and y-axes are inverted
    sigma_xx_left_average = np.nanmean(sigma_xx[:, 0:x_half, :, :], axis=(0, 1))
    sigma_xx_right_average = np.nanmean(sigma_xx[:, x_half:x_end, :, :], axis=(0, 1))
    sigma_yy_average = np.nanmean(sigma_yy, axis=(0, 1))
    sigma_yy_left_average = np.nanmean(sigma_yy[:, 0:x_half, :, :], axis=(0, 1))
    sigma_yy_right_average = np.nanmean(sigma_yy[:, x_half:x_end, :, :], axis=(0, 1))
    sigma_normal_average = np.nanmean(sigma_normal, axis=(0, 1))  # maps are coming from matlab calculations where x and y-axes are inverted
    sigma_normal_left_average = np.nanmean(sigma_normal[:, 0:x_half, :, :], axis=(0, 1))
    sigma_normal_right_average = np.nanmean(sigma_normal[:, x_half:x_end, :, :], axis=(0, 1))

    # average over first twenty frames before photoactivation
    sigma_xx_baseline = np.nanmean(sigma_xx_average[0:20, :], axis=0)
    sigma_yy_baseline = np.nanmean(sigma_yy_average[0:20, :], axis=0)
    sigma_normal_x_profile_baseline = np.nanmean(sigma_normal_x_profile[:, 0:20, :], axis=1)

    # normalize stesses by first substracting the baseline for each cell and then dividing by the average baseline
    relsigma_xx = (sigma_xx_average - np.nanmean(sigma_xx_average[0:20], axis=0)) / np.nanmean(sigma_xx_average[0:20], axis=(0, 1))
    relsigma_xx_left = \
        (sigma_xx_left_average - np.nanmean(sigma_xx_left_average[0:20], axis=0)) / np.nanmean(sigma_xx_left_average[0:20], axis=(0, 1))
    relsigma_xx_right = \
        (sigma_xx_right_average - np.nanmean(sigma_xx_right_average[0:20], axis=0)) / np.nanmean(sigma_xx_right_average[0:20], axis=(0, 1))

    relsigma_yy = (sigma_yy_average - np.nanmean(sigma_yy_average[0:20], axis=0)) / np.nanmean(sigma_yy_average[0:20], axis=(0, 1))
    relsigma_yy_left = \
        (sigma_yy_left_average - np.nanmean(sigma_yy_left_average[0:20], axis=0)) / np.nanmean(sigma_yy_left_average[0:20], axis=(0, 1))
    relsigma_yy_right = \
        (sigma_yy_right_average - np.nanmean(sigma_yy_right_average[0:20], axis=0)) / np.nanmean(sigma_yy_right_average[0:20], axis=(0, 1))

    relsigma_normal = (sigma_normal_average - np.nanmean(sigma_normal_average[0:20], axis=0)) / np.nanmean(sigma_normal_average[0:20],
                                                                                                           axis=(0, 1))
    relsigma_normal_left = \
        (sigma_normal_left_average - np.nanmean(sigma_normal_left_average[0:20], axis=0)) / np.nanmean(sigma_normal_left_average[0:20],
                                                                                                       axis=(0, 1))
    relsigma_normal_right = \
        (sigma_normal_right_average - np.nanmean(sigma_normal_right_average[0:20], axis=0)) / np.nanmean(sigma_normal_right_average[0:20],
                                                                                                         axis=(0, 1))

    # calculate anisotropy coefficient
    AIC = (sigma_xx_average - sigma_yy_average) / (sigma_xx_average + sigma_yy_average)
    AIC_baseline = np.nanmean(AIC[0:20, :], axis=0)

    # calculate relative stress increase
    RSI_xx = relsigma_xx[32, :] - relsigma_xx[20, :]
    RSI_xx_left = relsigma_xx_left[32, :] - relsigma_xx_left[20, :]
    RSI_xx_right = relsigma_xx_right[32, :] - relsigma_xx_right[20, :]

    RSI_yy = relsigma_yy[32, :] - relsigma_yy[20, :]
    RSI_yy_left = relsigma_yy_left[32, :] - relsigma_yy_left[20, :]
    RSI_yy_right = relsigma_yy_right[32, :] - relsigma_yy_right[20, :]

    RSI_normal = relsigma_normal[32, :] - relsigma_normal[20, :]
    RSI_normal_left = relsigma_normal_left[32, :] - relsigma_normal_left[20, :]
    RSI_normal_right = relsigma_normal_right[32, :] - relsigma_normal_right[20, :]

    # calculate relative stress profile along x-axis after photoactivation
    sigma_normal_x_profile_increase = (sigma_normal_x_profile[:, 32, :] - sigma_normal_x_profile[:, 20, :])

    # replace 0 with NaN to not mess up smoothing
    sigma_normal_x_profile_increase[sigma_normal_x_profile_increase == 0] = 'nan'

    # find position at which stress attenuates through sigmoid fit
    def find_stress_attenuation_position(stresscurve):
        def sigmoid(x, amp, x0, l0, offset):
            return amp / (1 + np.exp((x - x0) / l0)) + offset

        l0_all = np.zeros(stresscurve.shape[1])
        x0_all = np.zeros(stresscurve.shape[1])
        for c in range(stresscurve.shape[1]):
            x = np.linspace(-40, 40, stresscurve.shape[0])  # in µm
            y_current = stresscurve[:, c]
            x1 = np.delete(x, np.argwhere(np.isnan(y_current)))
            y1 = np.delete(y_current, np.argwhere(np.isnan(y_current)))

            gmodel = Model(sigmoid)
            result = gmodel.fit(y1, x=x1, amp=1e-3, x0=0, l0=1, offset=0)

            l0_all[c] = result.params.valuesdict()['l0']
            x0_all[c] = result.params.valuesdict()['x0']
        return l0_all, x0_all

    attenuation_length, attenuation_position = find_stress_attenuation_position(sigma_normal_x_profile_increase)

    data = {"sigma_xx": sigma_xx, "sigma_yy": sigma_yy, "sigma_normal": sigma_normal,
            "sigma_xx_average": sigma_xx_average, "sigma_yy_average": sigma_yy_average,
            "sigma_xx_left_average": sigma_xx_left_average,
            "sigma_yy_left_average": sigma_yy_left_average,
            "sigma_xx_right_average": sigma_xx_right_average,
            "sigma_yy_right_average": sigma_yy_right_average,
            "sigma_xx_baseline": sigma_xx_baseline, "sigma_yy_baseline": sigma_yy_baseline,
            "sigma_normal_x_profile_baseline": sigma_normal_x_profile_baseline,
            "relsigma_xx": relsigma_xx, "relsigma_yy": relsigma_yy,
            "relsigma_xx_left": relsigma_xx_left, "relsigma_yy_left": relsigma_yy_left, "relsigma_normal_left": relsigma_normal_left,
            "relsigma_xx_right": relsigma_xx_right, "relsigma_yy_right": relsigma_yy_right, "relsigma_normal_right": relsigma_normal_right,
            "AIC_baseline": AIC_baseline, "AIC": AIC,
            "RSI_xx": RSI_xx, "RSI_xx_left": RSI_xx_left, "RSI_xx_right": RSI_xx_right,
            "RSI_yy": RSI_yy, "RSI_yy_left": RSI_yy_left, "RSI_yy_right": RSI_yy_right,
            "RSI_normal": RSI_normal, "RSI_normal_left": RSI_normal_left, "RSI_normal_right": RSI_normal_right,
            "sigma_normal_x_profile_increase": sigma_normal_x_profile_increase,
            "attenuation_length": attenuation_length, "attenuation_position": attenuation_position}
    return data


def analyse_shape_data(folder, stressmappixelsize):
    Xtop = np.load(folder + "/Xtop.npy")
    Xbottom = np.load(folder + "/Xbottom.npy")

    Ytop = np.load(folder + "/Ytop.npy")
    Ybottom = np.load(folder + "/Ybottom.npy")

    noFrames = Xtop.shape[1]
    noCells = Xtop.shape[2]

    # I want to calculate the deformation of the contour, but the fibertracks have varying lengths, so to calculate the strain
    # I first need to interpolate them to a grid so that all tracks have the same shape
    nb_points_interpl = 50
    Xtop_interpl = np.zeros((nb_points_interpl, noFrames, noCells))
    Xbottom_interpl = np.zeros((nb_points_interpl, noFrames, noCells))

    Ytop_interpl = np.zeros((nb_points_interpl, noFrames, noCells))
    Ybottom_interpl = np.zeros((nb_points_interpl, noFrames, noCells))

    for cell in range(noCells):
        Xtop_current = Xtop[:, :, cell]
        Xbottom_current = Xbottom[:, :, cell]

        Ytop_current = Ytop[:, :, cell]
        Ybottom_current = Ybottom[:, :, cell]

        # remove zero lines
        Xtop_current = Xtop_current[~np.all(Xtop_current == 0, axis=1)]
        Xbottom_current = Xbottom_current[~np.all(Xbottom_current == 0, axis=1)]

        Ytop_current = Ytop_current[~np.all(Ytop_current == 0, axis=1)]
        Ybottom_current = Ybottom_current[~np.all(Ybottom_current == 0, axis=1)]

        # apply interpolation
        for frame in range(noFrames):
            Xtop_interpl[:, frame, cell] = np.linspace(Xtop_current[0, frame], Xtop_current[-1, frame], nb_points_interpl)
            Xbottom_interpl[:, frame, cell] = np.linspace(Xbottom_current[0, frame], Xbottom_current[-1, frame], nb_points_interpl)

            Ytop_interpl[:, frame, cell] = np.interp(Xtop_interpl[:, frame, cell], Xtop_current[:, frame], Ytop_current[:, frame])
            Ybottom_interpl[:, frame, cell] = np.interp(Xbottom_interpl[:, frame, cell], Xbottom_current[:, frame],
                                                        Ybottom_current[:, frame])

    # calculate width of the cell as function of x and t
    W = Ybottom_interpl - Ytop_interpl

    # calculate width of the cell center (where the junction is in the doublets) as function of t
    windowlength = 2
    center_left = int(nb_points_interpl / 2 - windowlength)
    center_right = int(nb_points_interpl / 2 + windowlength)
    W_center = np.nanmean(W[center_left:center_right, :, :], axis=0)
    relW_center = (W_center - np.nanmean(W_center[0:20], axis=0)) / np.nanmean(W_center[0:20])
    relW_center_end = relW_center[-1, :]

    # calculate contour strain at peak contraction
    epsilon = 1 - np.nanmean(W[:, 1:20, :], axis=1) / np.nanmean(W[:, 30:33, :], axis=1)

    # quantify degree of asymmetry of the strain
    epsilon_asymmetry_curve = epsilon - np.flipud(epsilon)
    epsilon_asymmetry_coefficient = np.nansum(epsilon_asymmetry_curve[0:int(epsilon_asymmetry_curve.shape[0] / 2)], axis=0)

    masks = np.load(folder + "/mask.npy")

    # calculate spreadingsizes with masks
    spreadingsize = (stressmappixelsize ** 2) * np.nansum(masks, axis=(0, 1))
    spreadingsize_baseline = np.nanmean(spreadingsize[0:20, :], axis=0)

    # load average actin angles
    actin_angles = np.load(folder + "/actin_angles.npy").squeeze(axis=0)

    actin_intensity_left = np.load(folder + "/actin_intensity_left.npy")
    actin_intensity_right = np.load(folder + "/actin_intensity_right.npy")

    # normalize actin intensities by first substracting the baseline for each cell and then dividing by the average baseline
    relactin_intensity_left = (actin_intensity_left - np.nanmean(actin_intensity_left[0:20, :], axis=0)) / np.nanmean(
        actin_intensity_left[0:20, :], axis=(0, 1))
    relactin_intensity_right = (actin_intensity_right - np.nanmean(actin_intensity_right[0:20, :], axis=0)) / np.nanmean(
        actin_intensity_right[0:20, :], axis=(0, 1))

    RAI_left = relactin_intensity_left[32, :] - relactin_intensity_left[20, :]
    RAI_right = relactin_intensity_right[32, :] - relactin_intensity_right[20, :]

    data = {"Xtop": Xtop, "Xbottom": Xbottom, "Ytop": Ytop, "Ybottom": Ybottom,
            "masks": masks, "cell_width_center": W_center, "relcell_width_center": relW_center, "relcell_width_center_end": relW_center_end,
            "contour_strain": epsilon, "ASC": epsilon_asymmetry_coefficient,
            "spreadingsize": spreadingsize, "spreadingsize_baseline": spreadingsize_baseline,
            "actin_angles": actin_angles,
            "actin_intensity_left": actin_intensity_left, "actin_intensity_right": actin_intensity_right,
            "relactin_intensity_left": relactin_intensity_left, "relactin_intensity_right": relactin_intensity_right,
            "RAI_left": RAI_left, "RAI_right": RAI_right}

    return data


def main_meta_analysis(folder, title, noFrames):
    stressmappixelsize = 0.864 * 1e-6  # in meter

    folder += title

    # calculate strain energies over all cells, normalize data to baseline values etc.
    TFM_data = analyse_tfm_data(folder, stressmappixelsize)

    # calculate averages over all cells, normalize data to baseline values etc.
    MSM_data = analyse_msm_data(folder)

    # calculate spreading area and such
    shape_data = analyse_shape_data(folder, stressmappixelsize)

    print(title + ": done!")
    alldata = {"TFM_data": TFM_data, "MSM_data": MSM_data, "shape_data": shape_data}

    return alldata


if __name__ == "__main__":
    # This is the folder where all the input data is stored
    folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"

    AR1to1dfsl = "AR1to1_doublets_full_stim_long"
    AR1to1sfsl = "AR1to1_singlets_full_stim_long"
    AR1to1dfss = "AR1to1_doublets_full_stim_short"
    AR1to1sfss = "AR1to1_singlets_full_stim_short"
    AR1to2dhs = "AR1to2_doublets_half_stim"
    AR1to1dhs = "AR1to1_doublets_half_stim"
    AR1to1shs = "AR1to1_singlets_half_stim"
    AR2to1dhs = "AR2to1_doublets_half_stim"

    # These functions perform a series of analyses and assemble a dictionary of dictionaries containing all the data that was used for plotting
    AR1to1d_fullstim_long = main_meta_analysis(folder, AR1to1dfsl, 60)
    AR1to1s_fullstim_long = main_meta_analysis(folder, AR1to1sfsl, 60)
    AR1to1d_fullstim_short = main_meta_analysis(folder, AR1to1dfss, 50)
    AR1to1s_fullstim_short = main_meta_analysis(folder, AR1to1sfss, 50)
    AR1to2d_halfstim = main_meta_analysis(folder, AR1to2dhs, 60)
    AR1to1d_halfstim = main_meta_analysis(folder, AR1to1dhs, 60)
    AR1to1s_halfstim = main_meta_analysis(folder, AR1to1shs, 60)
    AR2to1d_halfstim = main_meta_analysis(folder, AR2to1dhs, 60)

    # save dictionaries to a file using pickle
    if not os.path.exists(folder + "analysed_data"):
        os.mkdir(folder + "analysed_data")

    with open(folder + "analysed_data/AR1to1d_fullstim_long.dat", 'wb') as outfile:
        pickle.dump(AR1to1d_fullstim_long, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder + "analysed_data/AR1to1s_fullstim_long.dat", 'wb') as outfile:
        pickle.dump(AR1to1s_fullstim_long, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder + "analysed_data/AR1to1d_fullstim_short.dat", 'wb') as outfile:
        pickle.dump(AR1to1d_fullstim_short, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder + "analysed_data/AR1to1s_fullstim_short.dat", 'wb') as outfile:
        pickle.dump(AR1to1s_fullstim_short, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder + "analysed_data/AR1to2d_halfstim.dat", 'wb') as outfile:
        pickle.dump(AR1to2d_halfstim, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder + "analysed_data/AR1to1d_halfstim.dat", 'wb') as outfile:
        pickle.dump(AR1to1d_halfstim, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder + "analysed_data/AR1to1s_halfstim.dat", 'wb') as outfile:
        pickle.dump(AR1to1s_halfstim, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder + "analysed_data/AR2to1d_halfstim.dat", 'wb') as outfile:
        pickle.dump(AR2to1d_halfstim, outfile, protocol=pickle.HIGHEST_PROTOCOL)
