# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 09:33:36 2021

@author: Artur Ruppel
"""
import numpy as np
import scipy.io
from scipy import ndimage
from skimage import transform, io
from skimage.draw import polygon
from skimage.morphology import closing, disk
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt


def load_fibertracking_data(folder, fibertrackingshape, stressmapshape, noCells):

    x_end = fibertrackingshape[0]
    t_end = fibertrackingshape[1]  # nomber of frames
    cell_end = noCells

    # initialize arrays to store stress maps
    Xtop_all = np.zeros([x_end, t_end, cell_end])
    Xright_all = np.zeros([x_end, t_end, cell_end])
    Xbottom_all = np.zeros([x_end, t_end, cell_end])
    Xleft_all = np.zeros([x_end, t_end, cell_end])

    Ytop_all = np.zeros([x_end, t_end, cell_end])
    Yright_all = np.zeros([x_end, t_end, cell_end])
    Ybottom_all = np.zeros([x_end, t_end, cell_end])
    Yleft_all = np.zeros([x_end, t_end, cell_end])

    mask_all = np.zeros((stressmapshape[0], stressmapshape[1], t_end, cell_end), dtype=bool)

    # loop over all folders (one folder per cell/tissue)
    for cell in range(cell_end):
        # assemble paths to load stres smaps
        if cell < 9:
            foldercellpath = folder + "/cell0" + str(cell + 1)
        else:
            foldercellpath = folder + "/cell" + str(cell + 1)

        # load fibertracks
        fibertracks_mat = scipy.io.loadmat(foldercellpath + "/fibertracking.mat")

        Xtop = fibertracks_mat['Xtop']
        Xright = fibertracks_mat['Xright']
        Xbottom = fibertracks_mat['Xbottom']
        Xleft = fibertracks_mat['Xleft']

        Ytop = fibertracks_mat['Ytop']
        Yright = fibertracks_mat['Yright']
        Ybottom = fibertracks_mat['Ybottom']
        Yleft = fibertracks_mat['Yleft']

        for t in range(t_end):
            # a little messy way of getting masks from fibertracks
            img1 = np.zeros((1000, 1000), dtype=bool)
            c = np.concatenate((Xtop[:, t], Xright[:, t], Xbottom[:, t], Xleft[:, t]))
            r = np.concatenate((Ytop[:, t], Yright[:, t], Ybottom[:, t], Yleft[:, t]))
            rr, cc = polygon(r, c)
            img1[rr, cc] = 1

            img2 = np.zeros((1000, 1000), dtype=bool)
            c = np.flip(np.concatenate((Xtop[:, t], Xright[:, t], Xbottom[:, t], Xleft[:, t])), axis=0)
            r = np.flip(np.concatenate((Ytop[:, t], Yright[:, t], Ybottom[:, t], Yleft[:, t])), axis=0)
            rr, cc = polygon(r, c)
            img2[rr, cc] = 1
            img2 = np.flip(img2, axis=0)

            mask = np.logical_or(img1, img2)
            footprint = disk(20)

            # crop a 92*8 by 92*8 window around center
            mask_cropped = closing(mask[132:868, 132:868], footprint)

            # resize to stressmapshape
            mask_resized = transform.resize(mask_cropped, (92, 92))
            mask_all[:, :, t, cell] = mask_resized
            print('Load fibertracking: cell' + str(cell) + ', frame' + str(t))

        # plt.figure()
        # plt.imshow(mask_all[:,:,0,cell])
        # plt.show()
        # account for variable array size
        Xtop_all[0:Xtop.shape[0], :, cell] = Xtop
        Xright_all[0:Xright.shape[0], :, cell] = Xright
        Xbottom_all[0:Xbottom.shape[0], :, cell] = Xbottom
        Xleft_all[0:Xleft.shape[0], :, cell] = Xleft

        Ytop_all[0:Ytop.shape[0], :, cell] = Ytop
        Yright_all[0:Yright.shape[0], :, cell] = Yright
        Ybottom_all[0:Ybottom.shape[0], :, cell] = Ybottom
        Yleft_all[0:Yleft.shape[0], :, cell] = Yleft

        print("Fibertracks from cell " + str(cell) + " loaded")

    return Xtop_all, Xright_all, Xbottom_all, Xleft_all, Ytop_all, Yright_all, Ybottom_all, Yleft_all, mask_all


def load_MSM_and_TFM_data_and_actin_images(folder, noCells, stressmapshape, stressmappixelsize):
    '''this function was used to load the TFM and stress maps that resulted from the TFM and MSM analysis. It takes those maps and a mask with the cell border,
    # centers the maps around the mask, sets all values outside to 0, crops all maps to a consistent size and saves them in a data structure. The result of this
    # process is going to be used as input for all subsequent analyses'''
    x_end = stressmapshape[0]
    y_end = stressmapshape[1]
    t_end = stressmapshape[2]  # nomber of frames
    cell_end = noCells

    # initialize arrays to store stress maps
    Tx_all = np.zeros([x_end, y_end, t_end, cell_end])
    Ty_all = np.zeros([x_end, y_end, t_end, cell_end])
    Dx_all = np.zeros([x_end, y_end, t_end, cell_end])
    Dy_all = np.zeros([x_end, y_end, t_end, cell_end])
    sigma_xx_all = np.zeros([x_end, y_end, t_end, cell_end])
    sigma_yy_all = np.zeros([x_end, y_end, t_end, cell_end])
    images_all = np.zeros([x_end, y_end, t_end, cell_end])


    # loop over all folders (one folder per cell/tissue)
    for cell in range(cell_end):
        print('Load TFM data, MSM data and actin images: cell' + str(cell))
        # assemble paths to load stres smaps
        if cell < 9:
            foldercellpath = folder + "/cell0" + str(cell + 1)
        else:
            foldercellpath = folder + "/cell" + str(cell + 1)
            
        # read stack of actin images
        image = io.imread(foldercellpath + '/actin_ec.tif')
        
        # move t-axis to the last index
        image = np.moveaxis(image, 0, -1)

        # crop out a 92*8 by 92*8 window around the center
        image = image[132:868, 132:868, :]

        # scale down to resolution of traction maps
        image = transform.resize(image, stressmapshape)

        # load masks, stress and displacement maps
        TFM_mat = scipy.io.loadmat(foldercellpath + "/Allresults2.mat")
        stresstensor = np.load(
            foldercellpath + "/stressmaps.npy") / stressmappixelsize  # stressmaps are stored in N/pixel and have to be converted to N/m

        # recover mask from stress maps
        mask = stresstensor[0, :, :, 0] > 0

        # set TFM values outside of mask to 0
        Tx_new = TFM_mat["Tx"]
        Ty_new = TFM_mat["Ty"]
        Dx_new = TFM_mat["Dx"]
        Dy_new = TFM_mat["Dy"]

        # find the center of the mask
        x_center, y_center = np.rint(ndimage.measurements.center_of_mass(mask))

        # find the cropboundaries around the center, round and convert to integer
        x_crop_start = np.rint(x_center - x_end / 2).astype(int)
        x_crop_end = np.rint(x_center + x_end / 2).astype(int)
        y_crop_start = np.rint(y_center - y_end / 2).astype(int)
        y_crop_end = np.rint(y_center + y_end / 2).astype(int)

        # crop and store in array
        Tx_all[:, :, :, cell] = Tx_new[x_crop_start:x_crop_end, y_crop_start:y_crop_end, :]
        Ty_all[:, :, :, cell] = Ty_new[x_crop_start:x_crop_end, y_crop_start:y_crop_end, :]
        Dx_all[:, :, :, cell] = Dx_new[x_crop_start:x_crop_end, y_crop_start:y_crop_end, :]
        Dy_all[:, :, :, cell] = Dy_new[x_crop_start:x_crop_end, y_crop_start:y_crop_end, :]
        sigma_xx_all[:, :, :, cell], sigma_yy_all[:, :, :, cell] = stresstensor[(0, 1), x_crop_start:x_crop_end, y_crop_start:y_crop_end, :]
        images_all[:, :, :, cell] = image

    return sigma_xx_all, sigma_yy_all, Tx_all, Ty_all, Dx_all, Dy_all, images_all



def load_actin_angle_data(folder):
    actin_angles = scipy.io.loadmat(folder + "/actin_angles.mat")
    actin_angles = actin_angles["angles"]

    return actin_angles


def load_actin_intensity_data(folder):
    actin_intensities = scipy.io.loadmat(folder + "/actin_intensities.mat")
    actin_intensity_left = actin_intensities["intensity_left"]
    actin_intensity_right = actin_intensities["intensity_right"]

    return actin_intensity_left, actin_intensity_right



def main(folder_old, folder_new, title, noCells, noFrames):
    stressmappixelsize = 0.864 * 10 ** -6  # in meter
    stressmapshape = [92, 92, noFrames]
    fibertrackingshape = [50, noFrames]

    print('Data loading of ' + title + ' started!')
    # Xtop, Xright, Xbottom, Xleft, Ytop, Yright, Ybottom, Yleft, mask = \
    #     load_fibertracking_data(folder_old, fibertrackingshape, stressmapshape, noCells)
    sigma_xx, sigma_yy, Tx, Ty, Dx, Dy, actin_images = load_MSM_and_TFM_data_and_actin_images(folder_old, noCells, stressmapshape, stressmappixelsize)
    actin_angles = load_actin_angle_data(folder_old)
    actin_intensity_left, actin_intensity_right = load_actin_intensity_data(folder_old)
    # actin_images = load_actin_images(folder_old, stressmapshape, noCells)

    # np.save(folder_new + title + "/Xtop.npy", Xtop)
    # np.save(folder_new + title + "/Xright.npy", Xright)
    # np.save(folder_new + title + "/Xbottom.npy", Xbottom)
    # np.save(folder_new + title + "/Xleft.npy", Xleft)

    # np.save(folder_new + title + "/Ytop.npy", Ytop)
    # np.save(folder_new + title + "/Yright.npy", Yright)
    # np.save(folder_new + title + "/Ybottom.npy", Ybottom)
    # np.save(folder_new + title + "/Yleft.npy", Yleft)

    # np.save(folder_new + title + "/mask.npy", mask)

    np.save(folder_new + title + "/Dx.npy", Dx)
    np.save(folder_new + title + "/Dy.npy", Dy)
    np.save(folder_new + title + "/Tx.npy", Tx)
    np.save(folder_new + title + "/Ty.npy", Ty)
    np.save(folder_new + title + "/sigma_xx.npy", sigma_xx)
    np.save(folder_new + title + "/sigma_yy.npy", sigma_yy)

    np.save(folder_new + title + "/actin_angles.npy", actin_angles)

    np.save(folder_new + title + "/actin_intensity_left.npy", actin_intensity_left)
    np.save(folder_new + title + "/actin_intensity_right.npy", actin_intensity_right)

    np.save(folder_new + title + "/actin_images.npy", actin_images)

    print('Data loading of ' + title + ' terminated!')


if __name__ == "__main__":
    folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"

    main("D:/2021_OPTO H2000 stimulate all for 10 minutes/doublets", folder, "AR1to1 doublets full stim long", 42, 60)
    main("D:/2021_OPTO H2000 stimulate all for 10 minutes/singlets", folder, "AR1to1 singlets full stim long", 17, 60)
    main("D:/2021_OPTO H2000 stimulate all for 3 minutes/doublets", folder, "AR1to1 doublets full stim short", 35, 50)
    main("D:/2021_OPTO H2000 stimulate all for 3 minutes/singlets", folder, "AR1to1 singlets full stim short", 14, 50)
    main("D:/2020_OPTO H2000 stimulate left half doublets and singlets/TFM_doublets/AR1to1", folder, "AR1to1 doublets half stim", 29, 60)
    main("D:/2020_OPTO H2000 stimulate left half doublets and singlets/TFM_doublets/AR1to2", folder, "AR1to2 doublets half stim", 43, 60)
    main("D:/2020_OPTO H2000 stimulate left half doublets and singlets/TFM_doublets/AR1to1", folder, "AR1to1 doublets half stim", 29, 60)
    main("D:/2020_OPTO H2000 stimulate left half doublets and singlets/TFM_singlets", folder, "AR1to1 singlets half stim", 41, 60)
    main("D:/2020_OPTO H2000 stimulate left half doublets and singlets/TFM_doublets/AR2to1", folder, "AR2to1 doublets half stim", 18, 60)
