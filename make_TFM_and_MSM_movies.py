# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 22:31:18 2021

@author: Artur Ruppel
"""
import numpy as np
from matplotlib import pyplot as plt
import os
import moviepy.video.io.ImageSequenceClip
# 

def plot_TFM_and_MSM_individual_movies(folder, stressmappixelsize):
    # this function reads in all the displacement and traction data for one condition and makes movies out of them
    Dx = np.load(folder+"/Dx.npy") 
    Dy = np.load(folder+"/Dy.npy")
    Tx = np.load(folder+"/Tx.npy") 
    Ty = np.load(folder+"/Ty.npy")
    sigma_xx = np.load(folder+"/sigma_xx.npy") 
    sigma_yy = np.load(folder+"/sigma_yy.npy")
    
    x_end = np.shape(Dx)[1]
    y_end = np.shape(Dx)[0]
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
        print('Movies for cell ' + str(cell)+' started')
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
            print('.',end='')
            
            n=4 # every 5th arrow will be plotted
            fig = plt.figure()
            im = plt.imshow(displacement[:,:,t,cell], cmap=plt.get_cmap("hot"), interpolation="bilinear", extent=extent, vmin=0, vmax=2e-6)
            plt.quiver(xq[::n,::n], yq[::n,::n], Dx[::n,::n,t,cell],Dy[::n,::n,t,cell],angles='xy',scale = 1e-5, units='width', color="r")
            plt.colorbar(im)
            fig.savefig(displacementpath_frame)
            plt.close(fig)
            
            fig = plt.figure()
            im = plt.imshow(traction[:,:,t,cell], cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0, vmax=2000)
            plt.quiver(xq[::n,::n], yq[::n,::n], Tx[::n,::n,t,cell],Ty[::n,::n,t,cell],angles='xy',scale = 1e4, units='width', color="r")
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
                
def plot_TFM_and_MSM_average_movies(folder, stressmappixelsize, baselinefilter):
    print('Average movie started')
    # this function reads in all the displacement and traction data for one condition, averages over all cells and makes movies out of them
    Dx = np.load(folder+"/Dx.npy") 
    Dy = np.load(folder+"/Dy.npy")
    Tx = np.load(folder+"/Tx.npy") 
    Ty = np.load(folder+"/Ty.npy")
    sigma_xx = np.load(folder+"/sigma_xx.npy") 
    sigma_yy = np.load(folder+"/sigma_yy.npy")
    
    data = {"Dx": Dx,"Dy": Dy, "Tx": Tx, "Ty": Ty, "sigma_xx": sigma_xx,"sigma_yy": sigma_yy}
    
    Dx = data["Dx"]
    Dy = data["Dy"]
    Tx = data["Tx"]
    Ty = data["Ty"]
    sigma_xx = data["sigma_xx"]
    sigma_yy = data["sigma_yy"]
    
    x_end = np.shape(sigma_xx)[1]
    y_end = np.shape(sigma_xx)[0]
    t_end = np.shape(sigma_xx)[2] # nomber of frames
    
    # calculate euklidian norm of displacement and traction
    displacement    = np.sqrt(Dx**2 + Dy**2)
    traction        = np.sqrt(Tx**2 + Ty**2)
    
    # average over cells
    Dx_average = np.nanmean(Dx, axis=3)
    Dy_average = np.nanmean(Dy, axis=3)
    Tx_average = np.nanmean(Tx, axis=3)
    Ty_average = np.nanmean(Ty, axis=3)
    displacement_average = np.nanmean(displacement, axis=3)
    traction_average = np.nanmean(traction, axis=3)
    sigma_xx_average = np.nanmean(sigma_xx, axis=3)
    sigma_yy_average = np.nanmean(sigma_yy, axis=3)
    
    
    # define x and y axis extent for plotting
    extent = [0, x_end*stressmappixelsize*1e6, 0, y_end*stressmappixelsize*1e6] # convert to µm       
    xq, yq = np.meshgrid(np.linspace(0,extent[1],x_end), np.linspace(0,extent[3],y_end))
    
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
            xxstresspath = folder+"/_averagemovies/sigmaxx/sigmaxx0" +str(t)
            yystresspath = folder+"/_averagemovies/sigmayy/sigmayy0" +str(t)
        else:
            displacementpath = folder+"/_averagemovies/displacement/displacement" +str(t)
            tractionpath = folder+"/_averagemovies/traction/traction" +str(t)
            xxstresspath = folder+"/_averagemovies/sigmaxx/sigmaxx" +str(t)
            yystresspath = folder+"/_averagemovies/sigmayy/sigmayy" +str(t)
                
            n=4 # every 5th arrow will be plotted
            fig = plt.figure()
            im = plt.imshow(displacement_average[:,:,t], cmap=plt.get_cmap("hot"), interpolation="bilinear", extent=extent, vmin=0, vmax=1e-6)
            plt.quiver(xq[::n,::n], yq[::n,::n], Dx_average[::n,::n,t],Dy_average[::n,::n,t],angles='xy',scale = 1e-5, units='width', color="r")
            plt.colorbar(im)
            fig.savefig(displacementpath)
            plt.close(fig)
            
            fig = plt.figure()
            im = plt.imshow(traction_average[:,:,t], cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0, vmax=1000)
            plt.quiver(xq[::n,::n], yq[::n,::n], Tx_average[::n,::n,t],Ty_average[::n,::n,t],angles='xy',scale = 1e4, units='width', color="r")
            plt.colorbar(im)
            fig.savefig(tractionpath)
            plt.close(fig)
            
            fig = plt.figure()
            im = plt.imshow(sigma_xx_average[:,:,t], cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0, vmax=1e-2)
            plt.colorbar(im)
            fig.savefig(xxstresspath)
            plt.close(fig) 
            
            fig = plt.figure()
            im = plt.imshow(sigma_yy_average[:,:,t], cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0, vmax=1e-2)
            plt.colorbar(im)
            fig.savefig(yystresspath)
            plt.close(fig) 

    
    image_folder=folder+"/_averagemovies/displacement/"
    fps=10    
    image_files = [image_folder+img for img in os.listdir(image_folder) if img.endswith(".png")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(image_folder + 'average_displacement.mp4')
    for img in os.listdir(image_folder):
            if img.endswith(".png"): 
                os.remove(image_folder+img)
                
    image_folder=folder+"/_averagemovies/traction/"
    fps=10    
    image_files = [image_folder+img for img in os.listdir(image_folder) if img.endswith(".png")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(image_folder + 'average_traction.mp4')
    for img in os.listdir(image_folder):
            if img.endswith(".png"): 
                os.remove(image_folder+img)
    
    image_folder=folder+"/_averagemovies/sigmaxx/"
    fps=10    
    image_files = [image_folder+img for img in os.listdir(image_folder) if img.endswith(".png")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(image_folder + 'average_sigmaxx.mp4')
    for img in os.listdir(image_folder):
            if img.endswith(".png"): 
                os.remove(image_folder+img)
    
    image_folder=folder+"/_averagemovies/sigmayy/"
    fps=10    
    image_files = [image_folder+img for img in os.listdir(image_folder) if img.endswith(".png")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(image_folder + 'average_sigmayy.mp4')
    for img in os.listdir(image_folder):
            if img.endswith(".png"): 
                os.remove(image_folder+img)
    
    print("Average movies ready")
    
def main(folder, title, noCells, noFrames):
    print('Movie generation for ' + title + ' started!')
    stressmappixelsize = 0.864*1e-6 # in meter
    pixelsize = 0.108*1e-6
    
    folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"
    folder = folder+title
    
    # plots movies for displacement, traction and stress data for every cell. Takes about 4 hours
    plot_TFM_and_MSM_individual_movies(folder, stressmappixelsize)
    

    # plot average movies with filtered data
    plot_TFM_and_MSM_average_movies(folder, stressmappixelsize)
        


    
if __name__ == "__main__":
    # This is the folder where all the data is stored
    folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"
    
    # # These functions perform a series of analyses and assemble a dictionary of dictionaries containing all the data that was used for plotting
    # main(folder, "AR1to1 doublets full stim long",42,60)
    main(folder, "AR1to1 singlets full stim long",17,60)
    # main(folder, "AR1to1 doublets full stim short",35,50)
    # main(folder, "AR1to1 singlets full stim short",14,50)
    # main(folder, "AR1to2 doublets half stim",43,60)
    # main(folder, "AR1to1 doublets half stim",29,60)
    # main(folder, "AR1to1 singlets half stim",41,60)
    # main(folder, "AR2to1 doublets half stim",18,60)
    
