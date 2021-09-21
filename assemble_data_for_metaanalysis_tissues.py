# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 09:33:36 2021

@author: Artur Ruppel
"""
import os

import numpy as np
import scipy.io
from scipy import ndimage
from skimage import transform, io
from skimage.draw import polygon
from skimage.morphology import closing, disk
import matplotlib


# def load_fibertracking_data(folder, fibertrackingshape, stressmapshape, noCells):
#     x_end = fibertrackingshape[0]
#     t_end = fibertrackingshape[1]  # nomber of frames
#     cell_end = noCells
#
#     # initialize arrays to store stress maps
#     Xtop_all = np.zeros([x_end, t_end, cell_end])
#     Xright_all = np.zeros([x_end, t_end, cell_end])
#     Xbottom_all = np.zeros([x_end, t_end, cell_end])
#     Xleft_all = np.zeros([x_end, t_end, cell_end])
#
#     Ytop_all = np.zeros([x_end, t_end, cell_end])
#     Yright_all = np.zeros([x_end, t_end, cell_end])
#     Ybottom_all = np.zeros([x_end, t_end, cell_end])
#     Yleft_all = np.zeros([x_end, t_end, cell_end])
#
#     mask_all = np.zeros((stressmapshape[0], stressmapshape[1], t_end, cell_end), dtype=bool)
#
#     # loop over all folders (one folder per cell/tissue)
#     for cell in range(cell_end):
#         # assemble paths to load stres smaps
#         if cell < 9:
#             foldercellpath = folder + "/cell0" + str(cell + 1)
#         else:
#             foldercellpath = folder + "/cell" + str(cell + 1)
#
#         # load fibertracks
#         fibertracks_mat = scipy.io.loadmat(foldercellpath + "/fibertracking.mat")
#
#         Xtop = fibertracks_mat['Xtop']
#         Xright = fibertracks_mat['Xright']
#         Xbottom = fibertracks_mat['Xbottom']
#         Xleft = fibertracks_mat['Xleft']
#
#         Ytop = fibertracks_mat['Ytop']
#         Yright = fibertracks_mat['Yright']
#         Ybottom = fibertracks_mat['Ybottom']
#         Yleft = fibertracks_mat['Yleft']
#
#         for t in range(t_end):
#             # a little messy way of getting masks from fibertracks
#             img1 = np.zeros((1000, 1000), dtype=bool)
#             c = np.concatenate((Xtop[:, t], Xright[:, t], Xbottom[:, t], Xleft[:, t]))
#             r = np.concatenate((Ytop[:, t], Yright[:, t], Ybottom[:, t], Yleft[:, t]))
#             rr, cc = polygon(r, c)
#             img1[rr, cc] = 1
#
#             img2 = np.zeros((1000, 1000), dtype=bool)
#             c = np.flip(np.concatenate((Xtop[:, t], Xright[:, t], Xbottom[:, t], Xleft[:, t])), axis=0)
#             r = np.flip(np.concatenate((Ytop[:, t], Yright[:, t], Ybottom[:, t], Yleft[:, t])), axis=0)
#             rr, cc = polygon(r, c)
#             img2[rr, cc] = 1
#             img2 = np.flip(img2, axis=0)
#
#             mask = np.logical_or(img1, img2)
#             footprint = disk(20)
#
#             # crop a 92*8 by 92*8 window around center
#             mask_cropped = closing(mask[132:868, 132:868], footprint)
#
#             # resize to stressmapshape
#             mask_resized = transform.resize(mask_cropped, (92, 92))
#             mask_all[:, :, t, cell] = mask_resized
#             print('Load fibertracking: cell' + str(cell) + ', frame' + str(t))
#
#         # plt.figure()
#         # plt.imshow(mask_all[:,:,0,cell])
#         # plt.show()
#         # account for variable array size
#         Xtop_all[0:Xtop.shape[0], :, cell] = Xtop
#         Xright_all[0:Xright.shape[0], :, cell] = Xright
#         Xbottom_all[0:Xbottom.shape[0], :, cell] = Xbottom
#         Xleft_all[0:Xleft.shape[0], :, cell] = Xleft
#
#         Ytop_all[0:Ytop.shape[0], :, cell] = Ytop
#         Yright_all[0:Yright.shape[0], :, cell] = Yright
#         Ybottom_all[0:Ybottom.shape[0], :, cell] = Ybottom
#         Yleft_all[0:Yleft.shape[0], :, cell] = Yleft
#
#         print("Fibertracks from cell " + str(cell) + " loaded")
#
#     return Xtop_all, Xright_all, Xbottom_all, Xleft_all, Ytop_all, Yright_all, Ybottom_all, Yleft_all, mask_all
#

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

    # loop over all folders (one folder per cell/tissue)
    for cell in range(cell_end):
        print('Load TFM data, MSM data and actin images: tissue' + str(cell))
        # assemble paths to load stres smaps
        if cell < 9:
            foldercellpath = folder + "/tissue0" + str(cell + 1)
        else:
            foldercellpath = folder + "/tissue" + str(cell + 1)

        # load masks, stress and displacement maps
        TFM_mat = scipy.io.loadmat(foldercellpath + "/Allresults2.mat")
        stresstensor = np.load(
            foldercellpath + "/stressmaps.npy") / stressmappixelsize  # stressmaps are stored in N/pixel and have to be converted to N/m

        x_half = int(x_end / 2)
        y_half = int(y_end / 2)
        x_center = int(TFM_mat["Tx"].shape[0] / 2)
        y_center = int(TFM_mat["Tx"].shape[1] / 2)

        # crop
        Tx_current = TFM_mat["Tx"][y_center - y_half:y_center + y_half, x_center - x_half:x_center + x_half, :]
        Ty_current = TFM_mat["Ty"][y_center - y_half:y_center + y_half, x_center - x_half:x_center + x_half, :]
        Dx_current = TFM_mat["Dx"][y_center - y_half:y_center + y_half, x_center - x_half:x_center + x_half, :]
        Dy_current = TFM_mat["Dy"][y_center - y_half:y_center + y_half, x_center - x_half:x_center + x_half, :]
        sigma_xx_current = stresstensor[0, y_center - y_half:y_center + y_half, x_center - x_half:x_center + x_half, :]
        sigma_yy_current = stresstensor[1, y_center - y_half:y_center + y_half, x_center - x_half:x_center + x_half, :]
        
        # tissues that were the top half was stimulated have to be rotated by 90 degrees counterclockwise so that the stimulation zone is equivalent to the lefthalf stimulation
        if "tophalf" in folder:
            Tx_current = np.rot90(Tx_current, k=1, axes=(0,1))
            Ty_current = np.rot90(Ty_current, k=1, axes=(0,1))
            Dx_current = np.rot90(Dx_current, k=1, axes=(0,1))
            Dy_current = np.rot90(Dy_current, k=1, axes=(0,1))
            sigma_xx_current_rot = np.rot90(sigma_yy_current, k=1, axes=(0,1))
            sigma_yy_current_rot = np.rot90(sigma_xx_current, k=1, axes=(0,1))
            sigma_xx_current = sigma_xx_current_rot
            sigma_yy_current = sigma_yy_current_rot
            
        # store in array
        Tx_all[:, :, :, cell] = Tx_current
        Ty_all[:, :, :, cell] = Ty_current
        Dx_all[:, :, :, cell] = Dx_current
        Dy_all[:, :, :, cell] = Dy_current
        sigma_xx_all[:, :, :, cell] = sigma_xx_current
        sigma_yy_all[:, :, :, cell] = sigma_yy_current

    return sigma_xx_all, sigma_yy_all, Tx_all, Ty_all, Dx_all, Dy_all


# def load_actin_angle_data(folder):
#     actin_angles = scipy.io.loadmat(folder + "/actin_angles.mat")
#     actin_angles = actin_angles["angles"]
#
#     return actin_angles
#
#
# def load_actin_intensity_data(folder):
#     actin_intensities = scipy.io.loadmat(folder + "/actin_intensities.mat")
#     actin_intensity_left = actin_intensities["intensity_left"]
#     actin_intensity_right = actin_intensities["intensity_right"]
#
#     return actin_intensity_left, actin_intensity_right
#
#
# def save_actin_images_as_png(folder_old, folder_new, title, noCells, stressmapshape, stressmappixelsize):
#     savepath = folder_new + title + "/actin_images"
#     if not os.path.exists(savepath):
#         os.mkdir(savepath)
#     # loop over all folders (one folder per cell/tissue)
#     for cell in range(noCells):
#         print('Load actin images: cell' + str(cell))
#         # assemble paths to load stres smaps
#         if cell < 9:
#             foldercellpath = folder_old + "/cell0" + str(cell + 1)
#         else:
#             foldercellpath = folder_old + "/cell" + str(cell + 1)
#
#         # read stack of actin images
#         image = io.imread(foldercellpath + '/actin_ec.tif')
#
#         # move t-axis to the last index
#         image = np.moveaxis(image, 0, -1)
#
#         # crop out a 92*8 by 92*8 window around the center
#         image = image[132:868, 132:868, :]
#
#         for frame in range(stressmapshape[2]):
#             imagepath = savepath + "/cell" + str(cell) + "frame" + str(frame) + ".png"
#             matplotlib.image.imsave(imagepath, image[:, :, frame], cmap="gray")


def main(folder_old, folder_new, title, noCells, noFrames):
    stressmappixelsize = 1.296 * 10 ** -6  # in meter
    stressmapshape = [120, 120, noFrames]

    print('Data loading of ' + title + ' started!')

    sigma_xx, sigma_yy, Tx, Ty, Dx, Dy = load_MSM_and_TFM_data_and_actin_images(folder_old, noCells, stressmapshape, stressmappixelsize)

    if not os.path.exists(folder_new + title):
        os.mkdir(folder_new + title)

    np.save(folder_new + title + "/Dx.npy", Dx)
    np.save(folder_new + title + "/Dy.npy", Dy)
    np.save(folder_new + title + "/Tx.npy", Tx)
    np.save(folder_new + title + "/Ty.npy", Ty)
    np.save(folder_new + title + "/sigma_xx.npy", sigma_xx)
    np.save(folder_new + title + "/sigma_yy.npy", sigma_yy)

    print('Data loading of ' + title + ' terminated!')


if __name__ == "__main__":
    folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"

    main("C:/Users/Balland\Desktop/_collaborations/Vladimir Misiak/20micron_full_stim", folder, "tissues_20micron_full_stim", 31, 60)
    main("C:/Users/Balland\Desktop/_collaborations/Vladimir Misiak/20micron_lefthalf_stim", folder, "tissues_20micron_lefthalf_stim", 25, 60)
    main("C:/Users/Balland\Desktop/_collaborations/Vladimir Misiak/20micron_tophalf_stim", folder, "tissues_20micron_tophalf_stim", 22, 60)
    main("C:/Users/Balland\Desktop/_collaborations/Vladimir Misiak/40micron_full_stim", folder, "tissues_40micron_full_stim", 15, 60)
    main("C:/Users/Balland\Desktop/_collaborations/Vladimir Misiak/40micron_lefthalf_stim", folder, "tissues_40micron_lefthalf_stim", 16, 60)
    main("C:/Users/Balland\Desktop/_collaborations/Vladimir Misiak/40micron_tophalf_stim", folder, "tissues_40micron_tophalf_stim", 13, 60)