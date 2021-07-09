# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 09:33:36 2021

@author: Artur Ruppel
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import scipy.io
import os
from sklearn.linear_model import LinearRegression
import moviepy.video.io.ImageSequenceClip
from skimage.morphology import closing, dilation, disk
# 
import pickle





   
def analyse_TFM_data(folder, stressmappixelsize):
    Dx = np.load(folder+"/Dx.npy") 
    Dy = np.load(folder+"/Dy.npy")
    Tx = np.load(folder+"/Tx.npy") 
    Ty = np.load(folder+"/Ty.npy")
    
    x_end = np.shape(Dx)[1]
    y_end = np.shape(Dx)[0]
    t_end = np.shape(Dx)[2]
    cell_end = np.shape(Dx)[3]
    y_half = np.rint(y_end/2).astype(int) 
    x_half = np.rint(x_end/2).astype(int)
    
    Es_density = 0.5*(stressmappixelsize**2)*(Tx*Dx+Ty*Dy)        
    
    
    # average over whole cell and then over left and right half
    Es = np.nansum(Es_density, axis=(0,1))
    Es_lefthalf = np.nansum(Es_density[:,0:x_half,:,:], axis=(0,1))     # maps are coming from matlab calculations where x and y-axes are inverted
    Es_righthalf = np.nansum(Es_density[:,x_half:x_end,:,:], axis=(0,1))    
   
    # average over first twenty frames before photoactivation
    Es_baseline = np.nanmean(Es[0:20,:], axis=(0))
    
    # normalize stress data by their baseline
    relEs = np.divide(Es,Es_baseline)
    relEs_lefthalf = np.divide(Es_lefthalf,Es_baseline)
    relEs_righthalf = np.divide(Es_righthalf,Es_baseline)
    
    # calculate total force in x- and y-direction
    Fx = np.nansum(abs(Tx), axis=(0,1))*(stressmappixelsize**2)
    Fy = np.nansum(abs(Ty), axis=(0,1))*(stressmappixelsize**2)  
    
    force_angle = np.arctan(np.divide(Fy,Fx))*360/(2*np.pi)    
    force_angle_baseline = np.nanmean(force_angle[0:20,:],axis=0)
    
    # calculate cell-cell force
    Fx_lefthalf = np.nansum(Tx[:,0:x_half,:,:], axis=(0,1))*(stressmappixelsize**2) 
    Fx_righthalf = np.nansum(Tx[:,x_half:x_end,:,:], axis=(0,1))*(stressmappixelsize**2)     
    F_cellcell = Fx_lefthalf-Fx_righthalf
    
    data={"Dx": Dx,"Dy": Dy, "Tx": Tx,"Ty": Ty,
          "Es": Es, "Es_lefthalf": Es_lefthalf, "Es_righthalf": Es_righthalf, "Es_baseline": Es_baseline,
          "relEs": relEs, "relEs_lefthalf": relEs_lefthalf, "relEs_righthalf": relEs_righthalf,
          "Fx": Fx, "Fy": Fy, "force_angle": force_angle, "force_angle_baseline": force_angle_baseline,"F_cellcell": F_cellcell}
    
    return data
     
def analyse_MSM_data(folder):
    sigma_xx = np.load(folder+"/sigma_xx.npy") 
    sigma_yy = np.load(folder+"/sigma_yy.npy")

    x_end = np.shape(sigma_xx)[1]    
    x_half = np.rint(x_end/2).astype(int)
    
    # average over whole cell and then over left and right half
    sigma_xx_average = np.nanmean(sigma_xx, axis=(0,1))
    sigma_xx_lefthalf_average = np.nanmean(sigma_xx[:,0:x_half,:,:], axis=(0,1)) # maps are coming from matlab calculations where x and y-axes are inverted
    sigma_xx_righthalf_average = np.nanmean(sigma_xx[:,x_half:x_end,:,:], axis=(0,1))    
    sigma_yy_average = np.nanmean(sigma_yy, axis=(0,1))
    sigma_yy_lefthalf_average = np.nanmean(sigma_yy[:,0:x_half,:,:], axis=(0,1))
    sigma_yy_righthalf_average = np.nanmean(sigma_yy[:,x_half:x_end,:,:], axis=(0,1))
    
    # average over first twenty frames before photoactivation
    sigma_xx_baseline = np.nanmean(sigma_xx_average[0:20,:], axis=(0))
    sigma_yy_baseline = np.nanmean(sigma_yy_average[0:20,:], axis=(0))
    
    # normalize stress data by their baseline
    relsigma_xx_average = np.divide(sigma_xx_average,sigma_xx_baseline)
    relsigma_xx_lefthalf_average = np.divide(sigma_xx_lefthalf_average,sigma_xx_baseline)
    relsigma_xx_righthalf_average = np.divide(sigma_xx_righthalf_average,sigma_xx_baseline)
    
    relsigma_yy_average = np.divide(sigma_yy_average,sigma_yy_baseline)
    relsigma_yy_lefthalf_average = np.divide(sigma_yy_lefthalf_average,sigma_yy_baseline)
    relsigma_yy_righthalf_average = np.divide(sigma_yy_righthalf_average,sigma_yy_baseline)
    
    AIC = (sigma_xx_average-sigma_yy_average)/(sigma_xx_average+sigma_yy_average)
    AIC_left = (sigma_xx_lefthalf_average-sigma_yy_lefthalf_average)/(sigma_xx_lefthalf_average+sigma_yy_lefthalf_average)
    AIC_right = (sigma_xx_righthalf_average-sigma_yy_righthalf_average)/(sigma_xx_righthalf_average+sigma_yy_righthalf_average)
    
    AIC_baseline = np.nanmean(AIC[0:20,:], axis=(0))
    relAIC = AIC-AIC_baseline
    relAIC_left = AIC_left-AIC_baseline
    relAIC_right = AIC_right-AIC_baseline
    
    data={"sigma_xx": sigma_xx,"sigma_yy": sigma_yy,
          "sigma_xx_average": sigma_xx_average,"sigma_yy_average": sigma_yy_average,
          "sigma_xx_lefthalf_average": sigma_xx_lefthalf_average,"sigma_yy_lefthalf_average": sigma_yy_lefthalf_average,
          "sigma_xx_righthalf_average": sigma_xx_righthalf_average,"sigma_yy_righthalf_average": sigma_yy_righthalf_average,
          "sigma_xx_baseline": sigma_xx_baseline,"sigma_yy_baseline": sigma_yy_baseline,
          "relsigma_xx_average": relsigma_xx_average,"relsigma_yy_average": relsigma_yy_average,
          "relsigma_xx_lefthalf_average": relsigma_xx_lefthalf_average,"relsigma_yy_lefthalf_average": relsigma_yy_lefthalf_average,
          "relsigma_xx_righthalf_average": relsigma_xx_righthalf_average,"relsigma_yy_righthalf_average": relsigma_yy_righthalf_average,
          "AIC": AIC, "AIC_left": AIC_left, "AIC_right": AIC_right,\
          "relAIC": relAIC, "relAIC_left": relAIC_left, "relAIC_right": relAIC_right}
    
    return data

def analyse_shape_data(folder, pixelsize):
    masks = np.load(folder+"/mask.npy") 
    
    x_end = np.shape(masks)[1]
    y_end = np.shape(masks)[0]
    t_end = np.shape(masks)[2]
    cell_end = np.shape(masks)[3]
  
    spreadingsize = (pixelsize**2)*np.nansum(masks,axis=(0,1))
    
    spreadingsize_baseline = np.nanmean(spreadingsize[0:20,:],axis=0)
    
    data={"spreadingsize": spreadingsize, "spreadingsize_baseline": spreadingsize_baseline}
    
    return data

def create_filter(data,threshold):
    # initialize variables
    noVariables = np.shape(data)[2]
    t_end = np.shape(data)[0]
    baselinefilter_all = []
    # for each vector in data, find a linear regression and compare the slope to the threshold value. store result of this comparison in an array
    for i in range(noVariables):
        t = np.arange(t_end)
        model = LinearRegression().fit(t.reshape((-1, 1)), data[:,:,i])
        baselinefilter = np.absolute(model.coef_)<threshold
        baselinefilter_all.append(baselinefilter)
        
    # all vectors are combined to one through elementwise logical AND operation
    return np.all(baselinefilter_all,axis=0).reshape(-1)
    
def apply_filter(data, baselinefilter):
    for key in data:
        shape = data[key].shape        
        
        # find the new number of cells to find the new shape of data after filtering
        new_N = np.sum(baselinefilter)
        
        # to filter data of different dimensions, we first have to copy the filter vector into an array of the same shape as the data. We also create a variable with the new shape of the data
        if data[key].ndim == 1:
            baselinefilter_resized = baselinefilter
            newshape = [new_N]
        elif data[key].ndim == 2:
            baselinefilter_resized = np.expand_dims(baselinefilter, axis=0).repeat(shape[0],0)
            newshape = [shape[0], new_N]
        elif data[key].ndim == 3:
            baselinefilter_resized = np.expand_dims(baselinefilter, axis=(0,1)).repeat(shape[0],0).repeat(shape[1],1)    
            newshape = [shape[0], shape[1], new_N]
        elif data[key].ndim == 4:
            baselinefilter_resized = np.expand_dims(baselinefilter, axis=(0,1,2)).repeat(shape[0],0).repeat(shape[1],1).repeat(shape[2],2)    
            newshape = [shape[0], shape[1], shape[2], new_N]
        
        # apply filter
        data[key] = data[key][baselinefilter_resized].reshape(newshape)
    
    return data   
       
   
def main_meta_analysis(folder, title, noCells, noFrames):
    stressmappixelsize = 0.864*1e-6 # in meter
    pixelsize = 0.108*1e-6
    
    folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"
    folder = folder+title
    
    # # plots movies for displacement, traction and stress data for every cell. Takes about 4 hours
    # plot_TFM_and_MSM_individual_movies(folder, stressmappixelsize)
    
    # calculate strain energies over all cells, normalize data to baseline values etc.
    TFM_data = analyse_TFM_data(folder, stressmappixelsize)
    
    # calculate averages over all cells, normalize data to baseline values etc.
    MSM_data = analyse_MSM_data(folder)
    
    # calculate spreading area and such
    shape_data = analyse_shape_data(folder, pixelsize)
    
    # stack arrays together that will be used to determine the cells' baseline stability
    filterdata = np.dstack((TFM_data["relEs_lefthalf"][0:20,:],TFM_data["relEs_righthalf"][0:20,:],\
                            MSM_data["relsigma_yy_lefthalf_average"][0:20,:],MSM_data["relsigma_yy_righthalf_average"][0:20,:],\
                            MSM_data["relsigma_xx_lefthalf_average"][0:20,:],MSM_data["relsigma_xx_righthalf_average"][0:20,:]))
    
    # identifiy cells with unstable baselines
    baselinefilter = create_filter(filterdata,0.01)
    
    # # plot average movies with filtered data
    # plot_TFM_and_MSM_average_movies(folder, stressmappixelsize, baselinefilter)
    
    # # remove cells with unstable baselines
    # TFM_data = apply_filter(TFM_data, baselinefilter)
    # MSM_data = apply_filter(MSM_data, baselinefilter)
    # shape_data = apply_filter(shape_data, baselinefilter)
    
    new_N = np.sum(baselinefilter)
    print(title +": "+str(baselinefilter.shape[0]-new_N) + " cells were filtered out")
    
    alldata = {"TFM_data": TFM_data, "MSM_data": MSM_data, "shape_data": shape_data}

    return alldata


    
if __name__ == "__main__":
    # This is the folder where all the data is stored
    folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"
    
    # # These functions perform a series of analyses and assemble a dictionary of dictionaries containing all the data that was used for plotting
    # AR1to1d_fullstim_long = main_meta_analysis(folder, "AR1to1 doublets full stim long",42,60)
    AR1to1s_fullstim_long = main_meta_analysis(folder, "AR1to1 singlets full stim long",17,60)
    # AR1to1d_fullstim_short = main_meta_analysis(folder, "AR1to1 doublets full stim short",35,50)
    # AR1to1s_fullstim_short = main_meta_analysis(folder, "AR1to1 singlets full stim short",14,50)
    # AR1to2d_halfstim = main_meta_analysis(folder, "AR1to2 doublets half stim",43,60)
    # AR1to1d_halfstim = main_meta_analysis(folder, "AR1to1 doublets half stim",29,60)
    # AR1to1s_halfstim = main_meta_analysis(folder, "AR1to1 singlets half stim",41,60)
    # AR2to1d_halfstim = main_meta_analysis(folder, "AR2to1 doublets half stim",18,60)
    
    savefolder = 'C:/Users\Balland/Desktop/sigma_xx_averages'
    
    if not os.path.exists(folder + "analysed_data"):
        os.mkdir(folder + "analysed_data")
    
    
    # np.save(savefolder+'/AR1to1 doublets full stim long/sigma_xx.npy', AR1to1d_fullstim_long['MSM_data']['sigma_xx_average'])
    np.save(savefolder+'/AR1to1 singlets full stim long/sigma_xx.npy', AR1to1s_fullstim_long['MSM_data']['sigma_xx_average'])
    # np.save(savefolder+'/AR1to1 doublets full stim short/sigma_xx.npy', AR1to1d_fullstim_short['MSM_data']['sigma_xx_average'])
    # np.save(savefolder+'/AR1to1 singlets full stim short/sigma_xx.npy', AR1to1s_fullstim_short['MSM_data']['sigma_xx_average'])
    # np.save(savefolder+'/AR1to2 doublets half stim/sigma_xx.npy', AR1to2d_halfstim['MSM_data']['sigma_xx_average'])
    # np.save(savefolder+'/AR1to1 doublets half stim/sigma_xx.npy', AR1to1d_halfstim['MSM_data']['sigma_xx_average'])
    # np.save(savefolder+'/AR1to1 singlets half stim/sigma_xx.npy', AR1to1s_halfstim['MSM_data']['sigma_xx_average'])
    # np.save(savefolder+'/AR2to1 doublets half stim/sigma_xx.npy', AR2to1d_halfstim['MSM_data']['sigma_xx_average'])
    # np.save()
    
    # save dictionaries to a file using pickle
    if not os.path.exists(folder + "analysed_data"):
        os.mkdir(folder + "analysed_data")
    
    # with open(folder + "analysed_data/AR1to1d_fullstim_long.dat", 'wb') as outfile: pickle.dump(AR1to1d_fullstim_long, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder + "analysed_data/AR1to1s_fullstim_long.dat", 'wb') as outfile: pickle.dump(AR1to1s_fullstim_long, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(folder + "analysed_data/AR1to1d_fullstim_short.dat", 'wb') as outfile:pickle.dump(AR1to1d_fullstim_short, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(folder + "analysed_data/AR1to1s_fullstim_short.dat", 'wb') as outfile:pickle.dump(AR1to1s_fullstim_short, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(folder + "analysed_data/AR1to2d_halfstim.dat", 'wb') as outfile:      pickle.dump(AR1to2d_halfstim, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(folder + "analysed_data/AR1to1d_halfstim.dat", 'wb') as outfile:      pickle.dump(AR1to1d_halfstim, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(folder + "analysed_data/AR1to1s_halfstim.dat", 'wb') as outfile:      pickle.dump(AR1to1s_halfstim, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(folder + "analysed_data/AR2to1d_halfstim.dat", 'wb') as outfile:      pickle.dump(AR2to1d_halfstim, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    
    
    
# old functions that are not needed anymore


# def load_MSM_and_TFM_data(folder, noCells, stressmapshape, stressmappixelsize):
# # this function was used to load the TFM and stress maps that resulted from the TFM and MSM analysis. It takes those maps and a mask with the cell border, 
# # centers the maps around the mask, sets all values outside to 0, crops all maps to a consistent size and saves them in a data structure. The result of this 
# # process is going to be used as input for all subse
#     x_end = stressmapshape[0]
#     y_end = stressmapshape[1]
#     t_end = stressmapshape[2] # nomber of frames
#     cell_end = noCells
    
#     # initialize arrays to store stress maps
#     Tx_all = np.zeros([x_end, y_end, t_end, cell_end])
#     Ty_all = np.zeros([x_end, y_end, t_end, cell_end])
#     Dx_all = np.zeros([x_end, y_end, t_end, cell_end])
#     Dy_all = np.zeros([x_end, y_end, t_end, cell_end])
#     sigma_xx_all = np.zeros([x_end, y_end, t_end, cell_end])
#     sigma_yy_all = np.zeros([x_end, y_end, t_end, cell_end])
#     sigma_xy_all = np.zeros([x_end, y_end, t_end, cell_end])
#     sigma_yx_all = np.zeros([x_end, y_end, t_end, cell_end])
    
    
#     # loop over all folders (one folder per cell/tissue)
#     for cell in range(cell_end):
#         # assemble paths to load stres smaps
#         if cell < 9:
#             foldercellpath = folder+"/cell0"+str(cell+1)
#         else:
#             foldercellpath = folder+"/cell"+str(cell+1)
            
#         # load masks, stress and displacement maps
#         TFM_mat = scipy.io.loadmat(foldercellpath+"/Allresults2.mat")
#         stresstensor = np.load(foldercellpath+"/stressmaps.npy")/stressmappixelsize # stressmaps are stored in N/pixel and have to be converted to N/m
        
#         # recover mask from stress maps
#         mask = stresstensor[0,:,:,0] > 0
#         mask_all = stresstensor[0,:,:,:] > 0
        
#         # mask has some holes that have to be closed, because MSM analysis gave NaN on some pixels.
#         footprint = disk(10)
#         for t in range(t_end):
#             mask_all[:,:,t] = closing(mask_all[:,:,t],footprint)
#         mask = mask_all[:,:,0]
        
#         # set TFM values outside of mask to 0
#         Tx_new = TFM_mat["Tx"]*mask_all
#         Ty_new = TFM_mat["Ty"]*mask_all
#         Dx_new = TFM_mat["Dx"]*mask_all
#         Dy_new = TFM_mat["Dy"]*mask_all
        
#         # find the center of the mask
#         x_center, y_center = np.rint(ndimage.measurements.center_of_mass(mask))
        
#         # find the cropboundaries around the center, round and convert to integer
#         x_crop_start = np.rint(x_center-x_end/2).astype(int)
#         x_crop_end = np.rint(x_center+x_end/2).astype(int)
#         y_crop_start = np.rint(y_center-y_end/2).astype(int)
#         y_crop_end = np.rint(y_center+y_end/2).astype(int)
        
#         # crop and store in array
#         Tx_all[:,:,:,cell] = Tx_new[x_crop_start:x_crop_end,y_crop_start:y_crop_end,:]
#         Ty_all[:,:,:,cell] = Ty_new[x_crop_start:x_crop_end,y_crop_start:y_crop_end,:]
#         Dx_all[:,:,:,cell] = Dx_new[x_crop_start:x_crop_end,y_crop_start:y_crop_end,:]
#         Dy_all[:,:,:,cell] = Dy_new[x_crop_start:x_crop_end,y_crop_start:y_crop_end,:]
#         sigma_xx_all[:,:,:,cell], sigma_yy_all[:,:,:,cell] = stresstensor[(0,1),x_crop_start:x_crop_end,y_crop_start:y_crop_end,:]
        
#         print("Data from cell "+str(cell)+" loaded")        
        
        
#     return sigma_xx_all, sigma_yy_all, Tx_all, Ty_all, Dx_all, Dy_all  


   