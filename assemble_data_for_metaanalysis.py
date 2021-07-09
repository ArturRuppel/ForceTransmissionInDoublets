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
from skimage import transform
from scipy import ndimage
import time

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)
    
TicToc2 = TicTocGenerator()
def toc2(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc2
    tempTimeInterval = next(TicToc2)
    if tempBool:
        print( "Elapsed time 2: %f seconds.\n" %tempTimeInterval )

def tic2():
    # Records a time in TicToc2, marks the beginning of a time interval
    toc2(False)


def plot_TFM_and_MSM_maps(folder, stressmappixelsize):
    # this function reads in all the displacement and traction data for one condition and makes movies out of them
    Dx = np.load(folder+"/Dx_all.npy") 
    Dy = np.load(folder+"/Dy_all.npy")
    Tx = np.load(folder+"/Tx_all.npy") 
    Ty = np.load(folder+"/Ty_all.npy")
    sigma_xx = np.load(folder+"/sigma_xx_all.npy") 
    sigma_yy = np.load(folder+"/sigma_yy_all.npy")
    
    x_end = np.shape(Dx)[0]
    y_end = np.shape(Dx)[1]
    t_end = np.shape(Dx)[2]
    cell_end = np.shape(Dx)[3]
    
    # define x and y axis extent for plotting
    extent = [0, x_end*stressmappixelsize*1e6, 0, y_end*stressmappixelsize*1e6] # convert to µm       
    xq, yq = np.meshgrid(np.linspace(0,extent[1],x_end), np.linspace(0,extent[3],y_end))
    
    
    # define paths to store the movies
    displacementpath = folder+"/displacement_maps/"
    tractionpath = folder+"/traction_maps/"    
    xxstresspath = folder+"/xxstress_maps/"
    yystresspath = folder+"/yystress_maps/"
    
    # create folders to save plots
    if not os.path.exists(displacementpath):
        os.mkdir(displacementpath)
    if not os.path.exists(tractionpath):
        os.mkdir(tractionpath)
    if not os.path.exists(xxstresspath):
        os.mkdir(xxstresspath)
    if not os.path.exists(yystresspath):
        os.mkdir(yystresspath)
        
    # calculate norm of vectorial quantities
    displacement    = np.sqrt(Dx[:,:,:,:]**2 + Dy[:,:,:,:]**2)
    traction        = np.sqrt(Tx[:,:,:,:]**2 + Ty[:,:,:,:]**2)
    # xxstress    = np.sqrt(sigma_xx[:,:,:,:]**2 + sigma_yy[:,:,:,:]**2)
    
    # make map plots and save in folder
    for cell in range(cell_end):
        for t in range(t_end):    
            if t < 10:
                displacementpath_frame = displacementpath + "frame0"+str(t)
                tractionpath_frame = tractionpath + "frame0"+str(t)
                xxstresspath_frame = xxstresspath +"frame0"+str(t)
                yystresspath_frame = yystresspath +"frame0"+str(t)
            else:
                displacementpath_frame = displacementpath + "frame"+str(t)
                tractionpath_frame = tractionpath + "frame"+str(t)
                xxstresspath_frame = xxstresspath +"frame"+str(t)
                yystresspath_frame = yystresspath +"frame"+str(t)
            
            n=5 # every 5th arrow will be plotted
            fig = plt.figure()
            im = plt.imshow(displacement[:,:,t,cell], cmap=plt.get_cmap("hot"), interpolation="bilinear", extent=extent, vmin=0, vmax=2e-6)
            plt.quiver(xq[::n,::n], yq[::n,::n], Dx[::n,::n,t,cell],Dy[::n,::n,t,cell],angles='xy',scale = 1e-5, units='width', color="r")
            plt.colorbar(im)
            fig.savefig(displacementpath_frame)
            plt.close(fig)
            
            fig = plt.figure()
            im = plt.imshow(traction[:,:,t,cell], cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0, vmax=2000)
            plt.quiver(xq[::n,::n], yq[::n,::n], Dx[::n,::n,t,cell],Dy[::n,::n,t,cell],angles='xy',scale = 1e-5, units='width', color="r")
            plt.colorbar(im)
            fig.savefig(tractionpath_frame)
            plt.close(fig)
            
            fig = plt.figure()
            im = plt.imshow(sigma_xx[:,:,t,cell], cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0, vmax=2e-2)
            plt.colorbar(im)
            fig.savefig(xxstresspath_frame)
            plt.close(fig) 
            
            fig = plt.figure()
            im = plt.imshow(sigma_yy[:,:,t,cell], cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0, vmax=2e-2)
            plt.colorbar(im)
            fig.savefig(yystresspath_frame)
            plt.close(fig) 
            
        # load images, make movies, then remove images
        fps=10    
        image_files = [displacementpath+img for img in os.listdir(displacementpath) if img.endswith(".png")]
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile(displacementpath + "cell"+str(cell)+"_displacement.mp4")
        for img in os.listdir(displacementpath):
            if img.endswith(".png"): 
                os.remove(displacementpath+img)
                
        image_files = [tractionpath+img for img in os.listdir(tractionpath) if img.endswith(".png")]
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile(tractionpath + "cell"+str(cell)+"_traction.mp4")
        for img in os.listdir(tractionpath):
            if img.endswith(".png"): 
                os.remove(tractionpath+img)
                
        image_files = [xxstresspath+img for img in os.listdir(xxstresspath) if img.endswith(".png")]
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile(xxstresspath + "cell"+str(cell)+"_sigma_xx.mp4")
        for img in os.listdir(xxstresspath):
            if img.endswith(".png"): 
                os.remove(xxstresspath+img)
                
        image_files = [yystresspath+img for img in os.listdir(yystresspath) if img.endswith(".png")]
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile(yystresspath + "cell"+str(cell)+"_sigma_yy.mp4")
        for img in os.listdir(yystresspath):
            if img.endswith(".png"): 
                os.remove(yystresspath+img)

def plot_TFM_and_MSM_average_movies(sigma_xx_all, sigma_yy_all, Tx_all, Ty_all, Dx_all, Dy_all, stressmappixelsize, folder):
    x_end = np.shape(sigma_xx_all)[0]
    y_end = np.shape(sigma_xx_all)[1]
    t_end = np.shape(sigma_xx_all)[2] # nomber of frames
    
    # calculate euklidian norm of displacement and traction
    displacement_all    = np.sqrt(Dx_all**2 + Dy_all**2)
    traction_all        = np.sqrt(Tx_all**2 + Ty_all**2)
    
    # average over cells
    displacement_average = np.nanmean(displacement_all, axis=3)
    traction_average = np.nanmean(traction_all, axis=3)
    sigma_xx_average = np.nanmean(sigma_xx_all, axis=3)
    sigma_yy_average = np.nanmean(sigma_yy_all, axis=3)
    
    
    # define x and y axis extent for plotting
    extent = [0, x_end*stressmappixelsize, 0, y_end*stressmappixelsize]
    
    # create folders to save plots
    if not os.path.exists(folder+"/_averagemovies"):
        os.mkdir(folder+"/_averagemovies")
    if not os.path.exists(folder+"/_averagemovies/displacement"):
        os.mkdir(folder+"/_averagemovies/displacement")
    if not os.path.exists(folder+"/_averagemovies/traction"):
        os.mkdir(folder+"/_averagemovies/traction")
    if not os.path.exists(folder+"/_averagemovies/sigmaxx"):
        os.mkdir(folder+"/_averagemovies/sigmaxx")
    if not os.path.exists(folder+"/_averagemovies/sigmayy"):
        os.mkdir(folder+"/_averagemovies/sigmayy")
    
    for t in range(t_end):
        if t < 10:
            displacementpath = folder+"/_averagemovies/displacement/displacement0" +str(t)
            tractionpath = folder+"/_averagemovies/traction/traction0" +str(t)
            sigmaxxpath = folder+"/_averagemovies/sigmaxx/sigmaxx0" +str(t)
            sigmayypath = folder+"/_averagemovies/sigmayy/sigmayy0" +str(t)
        else:
            displacementpath = folder+"/_averagemovies/displacement/displacement" +str(t)
            tractionpath = folder+"/_averagemovies/traction/traction" +str(t)
            sigmaxxpath = folder+"/_averagemovies/sigmaxx/sigmaxx" +str(t)
            sigmayypath = folder+"/_averagemovies/sigmayy/sigmayy" +str(t)

                
        fig = plt.figure()
        im = plt.imshow(displacement_average[:,:,t], cmap=plt.get_cmap("hot"), interpolation="bilinear", extent=extent, vmin=0, vmax=1e-6)
        plt.colorbar(im)
        fig.savefig(displacementpath)
        plt.close(fig)
        
        fig = plt.figure()
        im = plt.imshow(traction_average[:,:,t], cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0, vmax=2000)
        plt.colorbar(im)
        fig.savefig(tractionpath)
        plt.close(fig)
        
        fig = plt.figure()
        im = plt.imshow(sigma_xx_average[:,:,t], cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0, vmax=1.5e-2)
        plt.colorbar(im)
        fig.savefig(sigmaxxpath)
        plt.close(fig)
        
        fig = plt.figure()
        im = plt.imshow(sigma_yy_average[:,:,t], cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0, vmax=1.5e-2)
        plt.colorbar(im)
        fig.savefig(sigmayypath)
        plt.close(fig)
        
        print("Frame "+str(t)+" of average movies plotted")
    
    image_folder=folder+"/_averagemovies/displacement/"
    fps=10    
    image_files = [image_folder+img for img in os.listdir(image_folder) if img.endswith(".png")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(image_folder + 'average_displacement.mp4')
    
    image_folder=folder+"/_averagemovies/traction/"
    fps=10    
    image_files = [image_folder+img for img in os.listdir(image_folder) if img.endswith(".png")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(image_folder + 'average_traction.mp4')
    
    image_folder=folder+"/_averagemovies/sigmaxx/"
    fps=10    
    image_files = [image_folder+img for img in os.listdir(image_folder) if img.endswith(".png")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(image_folder + 'average_sigmaxx.mp4')
    
    image_folder=folder+"/_averagemovies/sigmayy/"
    fps=10    
    image_files = [image_folder+img for img in os.listdir(image_folder) if img.endswith(".png")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(image_folder + 'average_sigmayy.mp4')
    
    print("Average movies ready")
        
def analyse_MSM_data(sigma_xx_all, sigma_yy_all, folder):
    x_end = np.shape(sigma_xx_all)[0]    
    x_half = np.rint(x_end/2).astype(int)
    
    # average over whole cell and then over left and right half
    sigma_xx_average = np.nanmean(sigma_xx_all, axis=(0,1))
    sigma_xx_lefthalf_average = np.nanmean(sigma_xx_all[:,0:x_half,:,:], axis=(0,1))
    sigma_xx_righthalf_average = np.nanmean(sigma_xx_all[:,x_half:x_end,:,:], axis=(0,1))    
    sigma_yy_average = np.nanmean(sigma_yy_all, axis=(0,1))
    sigma_yy_lefthalf_average = np.nanmean(sigma_yy_all[:,0:x_half,:,:], axis=(0,1))
    sigma_yy_righthalf_average = np.nanmean(sigma_yy_all[:,x_half:x_end,:,:], axis=(0,1))
    
    # average over first twenty frames before photoactivation
    sigma_xx_baseline = np.nanmean(sigma_xx_average[1:20,:], axis=(0))
    sigma_yy_baseline = np.nanmean(sigma_yy_average[1:20,:], axis=(0))
    
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
    
    AIC_baseline = np.nanmean(AIC[1:20,:], axis=(0))
    relAIC = AIC-AIC_baseline
    relAIC_left = AIC_left-AIC_baseline
    relAIC_right = AIC_right-AIC_baseline
    
    data={"sigma_xx_average": sigma_xx_average,"sigma_yy_average": sigma_yy_average,\
          "sigma_xx_lefthalf_average": sigma_xx_lefthalf_average,"sigma_yy_lefthalf_average": sigma_yy_lefthalf_average,\
          "sigma_xx_righthalf_average": sigma_xx_righthalf_average,"sigma_yy_righthalf_average": sigma_yy_righthalf_average,\
          "sigma_xx_baseline": sigma_xx_baseline,"sigma_yy_baseline": sigma_yy_baseline,\
          "relsigma_xx_average": relsigma_xx_average,"relsigma_yy_average": relsigma_yy_average,\
          "relsigma_xx_lefthalf_average": relsigma_xx_lefthalf_average,"relsigma_yy_lefthalf_average": relsigma_yy_lefthalf_average,\
          "relsigma_xx_righthalf_average": relsigma_xx_righthalf_average,"relsigma_yy_righthalf_average": relsigma_yy_righthalf_average,\
          "AIC": AIC, "AIC_left": AIC_left, "AIC_right": AIC_right,\
          "relAIC": relAIC, "relAIC_left": relAIC_left, "relAIC_right": relAIC_right}
    
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
    print(str(baselinefilter.shape[0]-new_N) + " cells were filtered out")
    return data   
       
def plot_value_over_time(x, y, ylim, xlabel, ylabel, title):
    # calculate mean, standard deviation and standard error over time
    mean_y = np.nanmean(y,axis=1)
    std_y = np.nanstd(y,axis=1)
    se_y = std_y/np.shape(y)[1]
    
    # plot data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x,mean_y,linestyle='None',marker='o',mfc='C0',mec='None',markersize=2,label='')
    ax.fill_between(x, y1=(mean_y-se_y), y2=(mean_y+se_y), interpolate=False,alpha=0.3, color='C0')
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
def plot_twovalues_over_time(x, y1, y2, ylim, xlabel, ylabel,titleleft, titleright, titleboth):
    # calculate mean, standard deviation and standard error over time
    mean_y1 = np.nanmean(y1,axis=1)
    std_y1 = np.nanstd(y1,axis=1)
    se_y1 = std_y1/np.shape(y1)[1]
    mean_y2 = np.nanmean(y2,axis=1)
    std_y2 = np.nanstd(y2,axis=1)
    se_y2 = std_y2/np.shape(y2)[1]
    
    # plot data
    fig = plt.figure()
    
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x,mean_y1,linestyle='None',marker='o',mfc='C0',mec='None',markersize=2,label='')
    ax.fill_between(x, y1=(mean_y1-se_y1), y2=(mean_y1+se_y1), interpolate=False,alpha=0.3, color='C0')
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(titleleft)
    
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x,mean_y2,linestyle='None',marker='o',mfc='C0',mec='None',markersize=2,label='')
    ax.fill_between(x, y1=(mean_y2-se_y2), y2=(mean_y2+se_y2), interpolate=False,alpha=0.3, color='C0')
    plt.ylim(ylim)
    plt.xlabel(xlabel)    
    plt.title(titleright)
    plt.suptitle(titleboth)
   
def load_masks(folder, noCells, noFrames):
# this function was used to load the masks and stores them in a bigger structure    
    # initialize arrays to store stress masks
    masks = np.zeros([92, 92, noFrames, noCells])    
    masks = masks>0
    x_end = 92
    y_end = 92
    
    # loop over all folders (one folder per cell/tissue)
    for cell in range(noCells):
        # assemble paths to load stres smaps
        if cell < 9:
            foldercellpath = folder+"/cell0"+str(cell+1)
        else:
            foldercellpath = folder+"/cell"+str(cell+1)
            
        # load masks, stress and displacement maps
        mask_mat = scipy.io.loadmat(foldercellpath+"/mask.mat")            
        mask = mask_mat["mask"]>0
        
        mask_small = transform.resize(mask,(112,112,60))
        
        # find the center of the mask
        x_center, y_center = np.rint(ndimage.measurements.center_of_mass(mask_small[:,:,cell]))
        
        # find the cropboundaries around the center, round and convert to integer
        x_crop_start = np.rint(x_center-x_end/2).astype(int)
        x_crop_end = np.rint(x_center+x_end/2).astype(int)
        y_crop_start = np.rint(y_center-y_end/2).astype(int)
        y_crop_end = np.rint(y_center+y_end/2).astype(int)
        
        # mask has some holes that have to be closed, because MSM analysis gave NaN on some pixels.
        kernel = disk(5)
        for t in range(noFrames):
            mask_current = mask_small[x_crop_start:x_crop_end,y_crop_start:y_crop_end,t]
            mask_current = closing(mask_current,kernel)
            # plt.imshow(mask_current)
            # plt.show()
            masks[:,:,t,cell] = mask_current
            
        print("Cell "+str(cell)+" done")       
    
    return masks
        

        
       

def main_postMSM_analysis(folder_old, folder_new, title, noCells, noFrames):
    stressmappixelsize = 0.864 * 10**-6 # in meter
    
    masks = load_masks(folder_old, noCells, noFrames)
    
    np.save(folder_new+title+"/masks.npy",masks)
    



if __name__ == "__main__":
    folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"
    
    
    main_postMSM_analysis("D:/2021_OPTO H2000 stimulate all for 10 minutes/doublets", folder, "AR1to1 doublets full stim",42,60)  
    main_postMSM_analysis("D:/2021_OPTO H2000 stimulate all for 10 minutes/singlets", folder, "AR1to1 singlets full stim",17,60)
    main_postMSM_analysis("D:/2021_OPTO H2000 stimulate all for 3 minutes/doublets", folder, "AR1to1 doublets full stim short",35,60)  
    main_postMSM_analysis("D:/2021_OPTO H2000 stimulate all for 3 minutes/singlets", folder, "AR1to1 singlets full stim short",14,60)

    main_postMSM_analysis("D:/2020_OPTO H2000 stimulate left half doublets and singlets/TFM_doublets/AR1to2", folder, "AR1to2 doublets half stim",43,60)
    main_postMSM_analysis("D:/2020_OPTO H2000 stimulate left half doublets and singlets/TFM_doublets/AR1to1", folder, "AR1to1 doublets half stim",29,60)
    main_postMSM_analysis("D:/2020_OPTO H2000 stimulate left half doublets and singlets/TFM_singlets", folder, "AR1to1 singlets half stim",41,60)
    main_postMSM_analysis("D:/2020_OPTO H2000 stimulate left half doublets and singlets/TFM_doublets/AR2to1", folder, "AR2to1 doublets half stim",18,60)
    
    
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


   