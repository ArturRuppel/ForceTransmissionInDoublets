# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 09:33:36 2021

@author: Artur Ruppel
"""
import os
import pickle

import numpy as np
from sklearn.linear_model import LinearRegression


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
    Es = np.nansum(Es_density, axis=(0, 1))
    Es_left = np.nansum(Es_density[:, 0:x_half, :, :],
                            axis=(0, 1))  # maps are coming from matlab calculations where x and y-axes are inverted
    Es_right = np.nansum(Es_density[:, x_half:x_end, :, :], axis=(0, 1))

    # average over first twenty frames before photoactivation
    Es_baseline = np.nanmean(Es[0:20, :], axis=(0))

    # normalize stress data by their baseline
    relEs = np.divide(Es, Es_baseline)
    relEs_left = np.divide(Es_left, Es_baseline)
    relEs_right = np.divide(Es_right, Es_baseline)

    # calculate total force in x- and y-direction
    Fx = np.nansum(abs(Tx), axis=(0, 1)) * (stressmappixelsize ** 2)
    Fy = np.nansum(abs(Ty), axis=(0, 1)) * (stressmappixelsize ** 2)

    force_angle = np.arctan(np.divide(Fy, Fx)) * 360 / (2 * np.pi)
    force_angle_baseline = np.nanmean(force_angle[0:20, :], axis=0)

    # calculate cell-cell force
    Fx_left = np.nansum(Tx[:, 0:x_half, :, :], axis=(0, 1)) * (stressmappixelsize ** 2)
    Fx_right = np.nansum(Tx[:, x_half:x_end, :, :], axis=(0, 1)) * (stressmappixelsize ** 2)
    F_cellcell = Fx_left - Fx_right

    # calculate corneraverages
    Fx_topleft = np.zeros((t_end, cell_end))
    Fx_topright = np.zeros((t_end, cell_end))
    Fx_bottomright = np.zeros((t_end, cell_end))
    Fx_bottomleft = np.zeros((t_end, cell_end))
    Fy_topleft = np.zeros((t_end, cell_end))
    Fy_topright = np.zeros((t_end, cell_end))
    Fy_bottomright = np.zeros((t_end, cell_end))
    Fy_bottomleft = np.zeros((t_end, cell_end))

    # calculate relative energy increase
    REI = relEs[33, :] - relEs[20, :]

    # find peak and sum up all values within radius r
    r = 12  # ~10 uM
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

            # Fx_topleft[t,cell] = np.nansum(Tx[0:y_half,0:x_half,t,cell])*stressmappixelsize**2
            # Fy_topleft[t,cell] = np.nansum(Ty[0:y_half,0:x_half,t,cell])*stressmappixelsize**2            

            # Fx_topright[t,cell] = np.nansum(Tx[0:y_half,x_half:x_end,t,cell])*stressmappixelsize**2
            # Fy_topright[t,cell] = np.nansum(Ty[0:y_half,x_half:x_end,t,cell])*stressmappixelsize**2

            # Fx_bottomleft[t,cell] = np.nansum(Tx[y_half:y_end,0:x_half,t,cell])*stressmappixelsize**2
            # Fy_bottomleft[t,cell] = np.nansum(Ty[y_half:y_end,0:x_half,t,cell])*stressmappixelsize**2

            # Fx_bottomright[t,cell] = np.nansum(Tx[y_half:y_end,x_half:x_end,t,cell])*stressmappixelsize**2
            # Fy_bottomright[t,cell] = np.nansum(Ty[y_half:y_end,x_half:x_end,t,cell])*stressmappixelsize**2

            # mask_corneraverages = np.zeros((x_end,y_end))
            # mask_corneraverages[0:x_half,0:y_half] = mask_topleft
            # mask_corneraverages[x_half:x_end,0:y_half] = mask_topright
            # mask_corneraverages[0:x_half,y_half:y_end,] = mask_bottomleft
            # mask_corneraverages[x_half:x_end,y_half:y_end,] = mask_bottomright

            # if t == 0:
            #     plt.figure()
            #     plt.imshow(T[:,:,t,cell]*mask_corneraverages)
            #     plt.show()

    data = {"Dx": Dx, "Dy": Dy, "Tx": Tx, "Ty": Ty,
            "Fx_topleft": Fx_topleft, "Fx_topright": Fx_topright, "Fx_bottomright": Fx_bottomright,
            "Fx_bottomleft": Fx_bottomleft,
            "Fy_topleft": Fy_topleft, "Fy_topright": Fy_topright, "Fy_bottomright": Fy_bottomright,
            "Fy_bottomleft": Fy_bottomleft,
            "Es": Es, "Es_left": Es_left, "Es_right": Es_right, "Es_baseline": Es_baseline,
            "relEs": relEs, "relEs_left": relEs_left, "relEs_right": relEs_right, "REI": REI,
            "Fx": Fx, "Fy": Fy, "force_angle": force_angle, "force_angle_baseline": force_angle_baseline,
            "F_cellcell": F_cellcell}

    return data


def analyse_msm_data(folder):
    sigma_xx = np.load(folder + "/sigma_xx.npy")
    sigma_yy = np.load(folder + "/sigma_yy.npy")

    # replace 0 with NaN to not mess up average calculations
    sigma_xx[sigma_xx == 0] = 'nan'
    sigma_yy[sigma_yy == 0] = 'nan'

    x_end = np.shape(sigma_xx)[1]
    x_half = np.rint(x_end / 2).astype(int)

    # average over whole cell and then over left and right half
    sigma_xx_average = np.nanmean(sigma_xx, axis=(0, 1))
    sigma_xx_left_average = np.nanmean(sigma_xx[:, 0:x_half, :, :], axis=(0, 1))  # maps are coming from matlab calculations where x and y-axes are inverted
    sigma_xx_right_average = np.nanmean(sigma_xx[:, x_half:x_end, :, :], axis=(0, 1))
    sigma_yy_average = np.nanmean(sigma_yy, axis=(0, 1))
    sigma_yy_left_average = np.nanmean(sigma_yy[:, 0:x_half, :, :], axis=(0, 1))
    sigma_yy_right_average = np.nanmean(sigma_yy[:, x_half:x_end, :, :], axis=(0, 1))

    # average over first twenty frames before photoactivation
    sigma_xx_baseline = np.nanmean(sigma_xx_average[0:20, :], axis=0)
    sigma_xx_left_baseline = np.nanmean(sigma_xx_left_average[0:20, :], axis=0)
    sigma_xx_right_baseline = np.nanmean(sigma_xx_right_average[0:20, :], axis=0)

    sigma_yy_baseline = np.nanmean(sigma_yy_average[0:20, :], axis=0)
    sigma_yy_left_baseline = np.nanmean(sigma_yy_left_average[0:20, :], axis=0)
    sigma_yy_right_baseline = np.nanmean(sigma_yy_right_average[0:20, :], axis=0)

    # normalize stress data by their baseline
    relsigma_xx = sigma_xx_average / sigma_xx_baseline
    relsigma_xx_left = sigma_xx_left_average / sigma_xx_left_baseline
    relsigma_xx_right = sigma_xx_right_average / sigma_xx_right_baseline

    relsigma_yy = sigma_yy_average / sigma_yy_baseline
    relsigma_yy_left = sigma_yy_left_average / sigma_yy_left_baseline
    relsigma_yy_right = sigma_yy_right_average / sigma_yy_right_baseline

    # sigma_xx_left_noBL = relsigma_xx_left-np.nanmean(relsigma_xx_left[0:20, :], axis=0)
    # sigma_xx_right_noBL = relsigma_xx_right-np.nanmean(relsigma_xx_right[0:20, :], axis=0)
    # sigma_yy_left_noBL = relsigma_yy_left-np.nanmean(relsigma_yy_left[0:20, :], axis=0)
    # sigma_yy_right_noBL = relsigma_yy_right-np.nanmean(relsigma_yy_right[0:20, :], axis=0)

    # normsigma_xx_left = sigma_xx_left_noBL / max(np.nanmean(sigma_xx_left_noBL, axis=1))
    # normsigma_xx_right = sigma_xx_right_noBL / max(np.nanmean(sigma_xx_left_noBL, axis=1))
    # normsigma_yy_left = sigma_yy_left_noBL / max(np.nanmean(sigma_yy_left_noBL, axis=1))
    # normsigma_yy_right = sigma_yy_right_noBL / max(np.nanmean(sigma_yy_left_noBL, axis=1))

    # calculate anisotropy coefficient
    AIC = (sigma_xx_average - sigma_yy_average) / (sigma_xx_average + sigma_yy_average)
    AIC_left = (sigma_xx_left_average - sigma_yy_left_average) / (
                sigma_xx_left_average + sigma_yy_left_average)
    AIC_right = (sigma_xx_right_average - sigma_yy_right_average) / (
                sigma_xx_right_average + sigma_yy_right_average)

    AIC_baseline = np.nanmean(AIC[0:20, :], axis=0)
    relAIC = AIC - AIC_baseline
    relAIC_left = AIC_left - AIC_baseline
    relAIC_right = AIC_right - AIC_baseline

    # calculate relative stress increase
    RSI_xx = relsigma_xx[33, :] - relsigma_xx[20, :]
    RSI_xx_left = relsigma_xx_left[33, :] - relsigma_xx_left[20, :]
    RSI_xx_right = relsigma_xx_right[33, :] - relsigma_xx_right[20, :]

    RSI_yy = relsigma_yy[33, :] - relsigma_yy[20, :]
    RSI_yy_left = relsigma_yy_left[33, :] - relsigma_yy_left[20, :]
    RSI_yy_right = relsigma_yy_right[33, :] - relsigma_yy_right[20, :]

    data = {"sigma_xx": sigma_xx, "sigma_yy": sigma_yy,
            "sigma_xx_average": sigma_xx_average, "sigma_yy_average": sigma_yy_average,
            "sigma_xx_left_average": sigma_xx_left_average,
            "sigma_yy_left_average": sigma_yy_left_average,
            "sigma_xx_right_average": sigma_xx_right_average,
            "sigma_yy_right_average": sigma_yy_right_average,
            "sigma_xx_baseline": sigma_xx_baseline, "sigma_yy_baseline": sigma_yy_baseline,
            "relsigma_xx": relsigma_xx, "relsigma_yy": relsigma_yy,
            "relsigma_xx_left": relsigma_xx_left,
            "relsigma_yy_left": relsigma_yy_left,
            "relsigma_xx_right": relsigma_xx_right,
            "relsigma_yy_right": relsigma_yy_right,
            # "sigma_xx_left_noBL": sigma_xx_left_noBL,
            # "sigma_yy_left_noBL": sigma_yy_left_noBL,
            # "sigma_xx_right_noBL": sigma_xx_right_noBL,
            # "sigma_yy_right_noBL": sigma_yy_right_noBL,
            "AIC_baseline": AIC_baseline, "AIC": AIC, "AIC_left": AIC_left, "AIC_right": AIC_right,
            "relAIC": relAIC, "relAIC_left": relAIC_left, "relAIC_right": relAIC_right,
            "RSI_xx": RSI_xx, "RSI_xx_left": RSI_xx_left, "RSI_xx_right": RSI_xx_right,
            "RSI_yy": RSI_yy, "RSI_yy_left": RSI_yy_left, "RSI_yy_right": RSI_yy_right}

    return data


def analyse_shape_data(folder, pixelsize):
    Xtop = np.load(folder + "/Xtop.npy")
    Xright = np.load(folder + "/Xright.npy")
    Xbottom = np.load(folder + "/Xbottom.npy")
    Xleft = np.load(folder + "/Xleft.npy")

    Ytop = np.load(folder + "/Ytop.npy")
    Yright = np.load(folder + "/Yright.npy")
    Ybottom = np.load(folder + "/Ybottom.npy")
    Yleft = np.load(folder + "/Yleft.npy")

    masks = np.load(folder + "/mask.npy")

    spreadingsize = (pixelsize ** 2) * np.nansum(masks, axis=(0, 1))

    spreadingsize_baseline = np.nanmean(spreadingsize[0:20, :], axis=0)

    actin_angles = np.load(folder + "/actin_angles.npy").squeeze(axis=0)

    actin_intensity_left = np.load(folder + "/actin_intensity_left.npy")
    actin_intensity_right = np.load(folder + "/actin_intensity_right.npy")

    relactin_intensity_left = actin_intensity_left / np.nanmean(actin_intensity_left[0:20,:], axis=0)
    relactin_intensity_right = actin_intensity_right / np.nanmean(actin_intensity_right[0:20, :], axis=0)

    RAI_left = relactin_intensity_left[33, :] - relactin_intensity_left[20, :]
    RAI_right = relactin_intensity_right[33, :] - relactin_intensity_right[20, :]

    data = {"Xtop": Xtop, "Xright": Xright, "Xbottom": Xbottom, "Xleft": Xleft,
            "Ytop": Ytop, "Yright": Yright, "Ybottom": Ybottom, "Yleft": Yleft,
            "spreadingsize": spreadingsize, "spreadingsize_baseline": spreadingsize_baseline,
            "actin_angles": actin_angles,
            "actin_intensity_left": actin_intensity_left, "actin_intensity_right": actin_intensity_right,
            "relactin_intensity_left": relactin_intensity_left, "relactin_intensity_right": relactin_intensity_right,
            "RAI_left": RAI_left, "RAI_right": RAI_right}

    return data


def create_filter(data, threshold):
    # initialize variables
    if np.ndim(data) < 3:
        data = data[..., np.newaxis]
    noVariables = np.shape(data)[2]
    t_end = np.shape(data)[0]
    baselinefilter_all = []
    # for each vector in data, find a linear regression and compare the slope to the threshold value. store result of this comparison in an array
    for i in range(noVariables):
        t = np.arange(t_end)
        model = LinearRegression().fit(t.reshape((-1, 1)), data[:, :, i])
        baselinefilter = np.absolute(model.coef_) < threshold
        baselinefilter_all.append(baselinefilter)

    # all vectors are combined to one through elementwise logical AND operation
    return np.all(baselinefilter_all, axis=0).reshape(-1)


def apply_filter(data, baselinefilter):
    for key in data:
        shape = data[key].shape

        # find the new number of cells to find the new shape of data after filtering
        new_N = np.sum(baselinefilter)

        # to filter data of different dimensions, we first have to copy the filter vector into an array of the same shape as the data. We also create a variable with the new shape of the data
        if data[key].ndim == 1:
            baselinefilter_resized = baselinefilter
            newshape = [new_N]
            data[key] = data[key][baselinefilter_resized].reshape(newshape)
        elif data[key].ndim == 2:
            baselinefilter_resized = np.expand_dims(baselinefilter, axis=0).repeat(shape[0], 0)
            newshape = [shape[0], new_N]
            data[key] = data[key][baselinefilter_resized].reshape(newshape)
        elif data[key].ndim == 3:
            baselinefilter_resized = np.expand_dims(baselinefilter, axis=(0, 1)).repeat(shape[0], 0).repeat(shape[1], 1)
            newshape = [shape[0], shape[1], new_N]
            data[key] = data[key][baselinefilter_resized].reshape(newshape)
        elif data[key].ndim == 4:
            baselinefilter_resized = np.expand_dims(baselinefilter, axis=(0, 1, 2)).repeat(shape[0], 0).repeat(shape[1], 1).repeat(shape[2], 2)
            newshape = [shape[0], shape[1], shape[2], new_N]
            data[key] = data[key][baselinefilter_resized].reshape(newshape)
        else:
            print('Nothing filtered, shape of array not supported')


    return data


def analyse_MSM_data_after_filtering(data):
    sigma_xx_left_average = data["sigma_xx_left_average"]
    sigma_xx_right_average = data["sigma_xx_right_average"]
    sigma_yy_left_average = data["sigma_yy_left_average"]
    sigma_yy_right_average = data["sigma_yy_right_average"]

    sigma_xx_left_noBL = sigma_xx_left_average-np.nanmean(sigma_xx_left_average[0:20, :], axis=0)
    sigma_xx_right_noBL = sigma_xx_right_average - np.nanmean(sigma_xx_right_average[0:20, :], axis=0)
    sigma_yy_left_noBL = sigma_yy_left_average - np.nanmean(sigma_yy_left_average[0:20, :], axis=0)
    sigma_yy_right_noBL = sigma_yy_right_average - np.nanmean(sigma_yy_right_average[0:20, :], axis=0)

    normsigma_xx_left = sigma_xx_left_noBL / max(np.nanmean(sigma_xx_left_noBL, axis=1))
    normsigma_xx_right = sigma_xx_right_noBL / max(np.nanmean(sigma_xx_left_noBL, axis=1))
    normsigma_yy_left = sigma_yy_left_noBL / max(np.nanmean(sigma_yy_left_noBL, axis=1))
    normsigma_yy_right = sigma_yy_right_noBL / max(np.nanmean(sigma_yy_left_noBL, axis=1))

    # calculate normalized stress increase
    NSI_xx_left = normsigma_xx_left[33, :] - normsigma_xx_left[20, :]
    NSI_xx_right = normsigma_xx_right[33, :] - normsigma_xx_right[20, :]

    NSI_yy_left = normsigma_yy_left[33, :] - normsigma_yy_left[20, :]
    NSI_yy_right = normsigma_yy_right[33, :] - normsigma_yy_right[20, :]

    data["normsigma_xx_left"] = normsigma_xx_left
    data["normsigma_xx_right"] = normsigma_xx_right
    data["normsigma_yy_left"] = normsigma_yy_left
    data["normsigma_yy_right"] = normsigma_yy_right

    # data["NSI_xx_left"] = NSI_xx_left
    # data["NSI_xx_right"] = NSI_xx_right
    # data["NSI_yy_left"] = NSI_yy_left
    # data["NSI_yy_right"] = NSI_yy_right

    return data


def main_meta_analysis(folder, title, noCells, noFrames):
    stressmappixelsize = 0.864 * 1e-6  # in meter
    pixelsize = 0.108 * 1e-6

    folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"
    folder += title

    # # plots movies for displacement, traction and stress data for every cell. Takes about 4 hours
    # plot_TFM_and_MSM_individual_movies(folder, stressmappixelsize)

    # calculate strain energies over all cells, normalize data to baseline values etc.
    TFM_data = analyse_tfm_data(folder, stressmappixelsize)

    # calculate averages over all cells, normalize data to baseline values etc.
    MSM_data = analyse_msm_data(folder)

    # calculate spreading area and such
    shape_data = analyse_shape_data(folder, pixelsize)

    # filter data to make sure that the baselines are stable
    filterdata = TFM_data["relEs"][0:20, :]#np.dstack((, TFM_data["relEs"][0:20, :]))
    baselinefilter = create_filter(filterdata, 0.005)

    # filterdata = np.dstack((MSM_data["normsigma_xx_left"][0:20, :], MSM_data["normsigma_xx_right"][0:20, :]))
    # baselinefilter1 = create_filter(filterdata, 0.2)
    #
    # filterdata = np.dstack((MSM_data["normsigma_yy_left"][0:20, :], MSM_data["normsigma_yy_right"][0:20, :]))
    # baselinefilter2 = create_filter(filterdata, 0.2)

    # baselinefilter = np.logical_and(baselinefilter1, baselinefilter2)

    # remove cells with unstable baselines
    TFM_data = apply_filter(TFM_data, baselinefilter)
    MSM_data = apply_filter(MSM_data, baselinefilter)
    shape_data = apply_filter(shape_data, baselinefilter)

    # # after filtering, the average peaks change so we have to normalize data again
    # MSM_data = analyse_MSM_data_after_filtering(MSM_data)

    new_N = np.sum(baselinefilter)
    print(title + ": " + str(baselinefilter.shape[0] - new_N) + " cells were filtered out")

    alldata = {"TFM_data": TFM_data, "MSM_data": MSM_data, "shape_data": shape_data}

    return alldata


if __name__ == "__main__":
    # This is the folder where all the input data is stored
    folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"

    AR1to1dfsl = "AR1to1 doublets full stim long"
    AR1to1sfsl = "AR1to1 singlets full stim long"
    AR1to1dfss = "AR1to1 doublets full stim short"
    AR1to1sfss = "AR1to1 singlets full stim short"
    AR1to2dhs = "AR1to2 doublets half stim"
    AR1to1dhs = "AR1to1 doublets half stim"
    AR1to1shs = "AR1to1 singlets half stim"
    AR2to1dhs = "AR2to1 doublets half stim"

    # These functions perform a series of analyses and assemble a dictionary of dictionaries containing all the data that was used for plotting
    AR1to1d_fullstim_long = main_meta_analysis(folder, AR1to1dfsl, 42, 60)
    AR1to1s_fullstim_long = main_meta_analysis(folder, AR1to1sfsl, 17, 60)
    AR1to1d_fullstim_short = main_meta_analysis(folder, AR1to1dfss, 35, 50)
    AR1to1s_fullstim_short = main_meta_analysis(folder, AR1to1sfss, 14, 50)
    AR1to2d_halfstim = main_meta_analysis(folder, AR1to2dhs, 43, 60)
    AR1to1d_halfstim = main_meta_analysis(folder, AR1to1dhs, 29, 60)
    AR1to1s_halfstim = main_meta_analysis(folder, AR1to1shs, 41, 60)
    AR2to1d_halfstim = main_meta_analysis(folder, AR2to1dhs, 18, 60)

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
