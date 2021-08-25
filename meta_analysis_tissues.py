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
            Fx_topleft[t, cell] = np.nansum(Tx[0:y_half, 0:x_half, t, cell] * mask_topleft) * stressmappixelsize ** 2
            Fy_topleft[t, cell] = np.nansum(Ty[0:y_half, 0:x_half, t, cell] * mask_topleft) * stressmappixelsize ** 2

            mask_topright = (x[np.newaxis, :] - T_topright_max[1]) ** 2 + (
                    y[:, np.newaxis] - T_topright_max[0]) ** 2 < r ** 2
            Fx_topright[t, cell] = np.nansum(
                Tx[0:y_half, x_half:x_end, t, cell] * mask_topright) * stressmappixelsize ** 2
            Fy_topright[t, cell] = np.nansum(
                Ty[0:y_half, x_half:x_end, t, cell] * mask_topright) * stressmappixelsize ** 2

            mask_bottomleft = (x[np.newaxis, :] - T_bottomleft_max[1]) ** 2 + (
                    y[:, np.newaxis] - T_bottomleft_max[0]) ** 2 < r ** 2
            Fx_bottomleft[t, cell] = np.nansum(
                Tx[y_half:y_end, 0:x_half, t, cell] * mask_bottomleft) * stressmappixelsize ** 2
            Fy_bottomleft[t, cell] = np.nansum(
                Ty[y_half:y_end, 0:x_half, t, cell] * mask_bottomleft) * stressmappixelsize ** 2

            mask_bottomright = (x[np.newaxis, :] - T_bottomright_max[1]) ** 2 + (
                    y[:, np.newaxis] - T_bottomright_max[0]) ** 2 < r ** 2
            Fx_bottomright[t, cell] = np.nansum(
                Tx[y_half:y_end, x_half:x_end, t, cell] * mask_bottomright) * stressmappixelsize ** 2
            Fy_bottomright[t, cell] = np.nansum(
                Ty[y_half:y_end, x_half:x_end, t, cell] * mask_bottomright) * stressmappixelsize ** 2

    data = {"Dx": Dx, "Dy": Dy, "Tx": Tx, "Ty": Ty,
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
    
    # recover masks from stress maps
    masks = sigma_xx > 0

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


def main_meta_analysis(folder, title, noFrames):
    stressmappixelsize = 1.296 * 1e-6  # in meter

    folder += title

    # calculate strain energies over all cells, normalize data to baseline values etc.
    TFM_data = analyse_tfm_data(folder, stressmappixelsize)

    # calculate averages over all cells, normalize data to baseline values etc.
    MSM_data = analyse_msm_data(folder)

    print(title + ": done!")
    alldata = {"TFM_data": TFM_data, "MSM_data": MSM_data}

    return alldata


if __name__ == "__main__":
    # This is the folder where all the input data is stored
    folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"

    # These functions perform a series of analyses and assemble a dictionary of dictionaries containing all the data that was used for plotting
    tissues_20micron_full_stim = main_meta_analysis(folder, "tissues_20micron_full_stim", 60)
    tissues_20micron_lefthalf_stim = main_meta_analysis(folder, "tissues_20micron_lefthalf_stim", 60)
    tissues_20micron_tophalf_stim = main_meta_analysis(folder, "tissues_20micron_tophalf_stim", 60)
    tissues_40micron_full_stim = main_meta_analysis(folder, "tissues_40micron_full_stim", 60)
    tissues_40micron_lefthalf_stim = main_meta_analysis(folder, "tissues_40micron_lefthalf_stim", 60)
    tissues_40micron_tophalf_stim = main_meta_analysis(folder, "tissues_40micron_tophalf_stim", 60)

    # save dictionaries to a file using pickle
    if not os.path.exists(folder + "analysed_data"):
        os.mkdir(folder + "analysed_data")

    with open(folder + "analysed_data/tissues_20micron_full_stim.dat", 'wb') as outfile:
        pickle.dump(tissues_20micron_full_stim, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder + "analysed_data/tissues_20micron_lefthalf_stim.dat", 'wb') as outfile:
        pickle.dump(tissues_20micron_lefthalf_stim, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder + "analysed_data/tissues_20micron_tophalf_stim.dat", 'wb') as outfile:
        pickle.dump(tissues_20micron_tophalf_stim, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder + "analysed_data/tissues_40micron_full_stim.dat", 'wb') as outfile:
        pickle.dump(tissues_40micron_full_stim, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder + "analysed_data/tissues_40micron_lefthalf_stim.dat", 'wb') as outfile:
        pickle.dump(tissues_40micron_lefthalf_stim, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder + "analysed_data/tissues_40micron_tophalf_stim.dat", 'wb') as outfile:
        pickle.dump(tissues_40micron_tophalf_stim, outfile, protocol=pickle.HIGHEST_PROTOCOL)

