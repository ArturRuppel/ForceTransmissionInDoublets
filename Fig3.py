# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 21:56:01 2021

@author: Artur Ruppel

"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib as mpl
import pickle
import seaborn as sns
import pandas as pd
from scipy.stats import normaltest, shapiro
from statannot import add_stat_annotation
import os
import pylustrator

# mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 8

# define some colors for the plots
colors_parent = ['#026473','#E3CC69','#77C8A6','#D96248'];
colors_parent_dark = ['#01353D','#564910','#235741','#A93B23'];

def test_if_gaussian(data1, data2, title):
    # test if data follows Gaussian distribution
    stat, p_n1 = normaltest(data1)
    stat, p_s1 = shapiro(data1)
    stat, p_n2 = normaltest(data2)
    stat, p_s2 = shapiro(data2)
    print('#############################################')
    # depending on the result of the Gaussian distribution test, perform either unpaired t-test or Mann-Whitney U test
    if (p_n1 > 0.05 and p_s1 > 0.05 and p_n2 > 0.05 and p_s2 > 0.05):    
        gaussian = True
        print(title + ': Probably Gaussian.')
    else:
        gaussian = False
        print(title + ': Probably not Gaussian.') 
    
    return gaussian


#%% load data for plotting
folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"
    
AR1to1d_fullstim_long =   pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_long.dat", "rb"))
AR1to1s_fullstim_long =   pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_long.dat", "rb"))
# AR1to1d_fullstim_short =  pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_short.dat", "rb"))
# AR1to1s_fullstim_short =  pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_short.dat", "rb"))
# AR1to2d_halfstim =        pickle.load(open(folder + "analysed_data/AR1to2d_halfstim.dat", "rb"))
AR1to1d_halfstim =        pickle.load(open(folder + "analysed_data/AR1to1d_halfstim.dat", "rb"))
AR1to1s_halfstim =        pickle.load(open(folder + "analysed_data/AR1to1s_halfstim.dat", "rb"))
# AR2to1d_halfstim =        pickle.load(open(folder + "analysed_data/AR2to1d_halfstim.dat", "rb"))



figfolder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/_Figure3/"
if not os.path.exists(figfolder):
        os.mkdir(figfolder)
#%% prepare dataframe for boxplots

# initialize empty dictionaries
concatenated_data_fs = {}
concatenated_data_hs = {}
concatenated_data_doublet = {}
concatenated_data_singlet = {}

# loop over all keys
for key1 in AR1to1d_fullstim_long: # keys are the same for all dictionaries so I'm just taking one example here
    for key2 in AR1to1d_fullstim_long[key1]:
        if AR1to1d_fullstim_long[key1][key2].ndim == 1: # only 1D data can be stored in the data frame
        
            # concatenate values from different experiments
            concatenated_data_fs[key2] = np.concatenate((AR1to1d_fullstim_long[key1][key2], AR1to1s_fullstim_long[key1][key2]))
            concatenated_data_hs[key2] = np.concatenate((AR1to1d_halfstim[key1][key2], AR1to1s_halfstim[key1][key2]))
            concatenated_data_doublet[key2] = np.concatenate((AR1to1d_fullstim_long[key1][key2], AR1to1d_halfstim[key1][key2]))
            concatenated_data_singlet[key2] = np.concatenate((AR1to1s_fullstim_long[key1][key2], AR1to1s_halfstim[key1][key2]))
            
            
# get number of elements for both condition
n_d_fullstim = AR1to1d_fullstim_long[key1][key2].shape[0]
n_d_halfstim = AR1to1d_halfstim[key1][key2].shape[0]
n_s_fullstim = AR1to1s_fullstim_long[key1][key2].shape[0]
n_s_halfstim = AR1to1s_halfstim[key1][key2].shape[0]

# create a list of keys with the same dimensions as the data
keys1to1d_fs = ['AR1to1d_fs' for i in range(n_d_fullstim)]
keys1to1s_fs = ['AR1to1s_fs' for i in range(n_s_fullstim)]
keys1to1d_hs = ['AR1to1d_hs' for i in range(n_d_halfstim)]
keys1to1s_hs = ['AR1to1s_hs' for i in range(n_s_halfstim)]

keys_fs = np.concatenate((keys1to1d_fs, keys1to1s_fs))
keys_hs = np.concatenate((keys1to1d_hs, keys1to1s_hs))
keys_doublet = np.concatenate((keys1to1d_fs, keys1to1d_hs))
keys_singlet = np.concatenate((keys1to1s_fs, keys1to1s_hs))

# add keys to dictionary with concatenated data
concatenated_data_fs['keys'] = keys_fs
concatenated_data_hs['keys'] = keys_hs
concatenated_data_doublet['keys'] = keys_doublet
concatenated_data_singlet['keys'] = keys_singlet

# Creates DataFrame
df_fs = pd.DataFrame(concatenated_data_fs)
df_hs = pd.DataFrame(concatenated_data_hs)
df_doublet = pd.DataFrame(concatenated_data_doublet)
df_singlet = pd.DataFrame(concatenated_data_singlet)

df_d_fs = df_fs[df_fs["keys"] == "AR1to1d_fs"]
df_d_hs = df_hs[df_hs["keys"] == "AR1to1d_hs"]

df_s_fs = df_fs[df_fs["keys"] == "AR1to1s_fs"]
df_s_hs = df_hs[df_hs["keys"] == "AR1to1s_hs"]
#%% plot figure 3B, TFM differences

# prepare data first
Tx_1to1d_fs = AR1to1d_fullstim_long["TFM_data"]["Tx"]
Ty_1to1d_fs = AR1to1d_fullstim_long["TFM_data"]["Ty"]
Tx_1to1s_fs = AR1to1s_fullstim_long["TFM_data"]["Tx"]
Ty_1to1s_fs = AR1to1s_fullstim_long["TFM_data"]["Ty"]

Tx_1to1d_hs = AR1to1d_halfstim["TFM_data"]["Tx"]
Ty_1to1d_hs = AR1to1d_halfstim["TFM_data"]["Ty"]
Tx_1to1s_hs = AR1to1s_halfstim["TFM_data"]["Tx"]
Ty_1to1s_hs = AR1to1s_halfstim["TFM_data"]["Ty"]

# calculate amplitudes
T_1to1d_fs = np.sqrt(Tx_1to1d_fs**2+Ty_1to1d_fs**2)
T_1to1s_fs = np.sqrt(Tx_1to1s_fs**2+Ty_1to1s_fs**2)
T_1to1d_hs = np.sqrt(Tx_1to1d_hs**2+Ty_1to1d_hs**2)
T_1to1s_hs = np.sqrt(Tx_1to1s_hs**2+Ty_1to1s_hs**2)

# calculate difference between after and before photoactivation
Tx_1to1d_fs_diff = np.nanmean(Tx_1to1d_fs[:,:,33,:]-Tx_1to1d_fs[:,:,20,:],axis=2)
Ty_1to1d_fs_diff = np.nanmean(Ty_1to1d_fs[:,:,33,:]-Ty_1to1d_fs[:,:,20,:],axis=2)
T_1to1d_fs_diff = np.nanmean(T_1to1d_fs[:,:,33,:]-T_1to1d_fs[:,:,20,:],axis=2)

Tx_1to1s_fs_diff = np.nanmean(Tx_1to1s_fs[:,:,33,:]-Tx_1to1s_fs[:,:,20,:],axis=2)
Ty_1to1s_fs_diff = np.nanmean(Ty_1to1s_fs[:,:,33,:]-Ty_1to1s_fs[:,:,20,:],axis=2)
T_1to1s_fs_diff = np.nanmean(T_1to1s_fs[:,:,33,:]-T_1to1s_fs[:,:,20,:],axis=2)

Tx_1to1d_hs_diff = np.nanmean(Tx_1to1d_hs[:,:,33,:]-Tx_1to1d_hs[:,:,20,:],axis=2)
Ty_1to1d_hs_diff = np.nanmean(Ty_1to1d_hs[:,:,33,:]-Ty_1to1d_hs[:,:,20,:],axis=2)
T_1to1d_hs_diff = np.nanmean(T_1to1d_hs[:,:,33,:]-T_1to1d_hs[:,:,20,:],axis=2)

Tx_1to1s_hs_diff = np.nanmean(Tx_1to1s_hs[:,:,33,:]-Tx_1to1s_hs[:,:,20,:],axis=2)
Ty_1to1s_hs_diff = np.nanmean(Ty_1to1s_hs[:,:,33,:]-Ty_1to1s_hs[:,:,20,:],axis=2)
T_1to1s_hs_diff = np.nanmean(T_1to1s_hs[:,:,33,:]-T_1to1s_hs[:,:,20,:],axis=2)

# crop maps 
crop_start = 8
crop_end = 84

Tx_1to1d_fs_diff_crop = Tx_1to1d_fs_diff[crop_start:crop_end,crop_start:crop_end]*1e-3 # convert to kPa
Ty_1to1d_fs_diff_crop = Ty_1to1d_fs_diff[crop_start:crop_end,crop_start:crop_end]*1e-3
T_1to1d_fs_diff_crop = T_1to1d_fs_diff[crop_start:crop_end,crop_start:crop_end]*1e-3

Tx_1to1s_fs_diff_crop = Tx_1to1s_fs_diff[crop_start:crop_end,crop_start:crop_end]*1e-3
Ty_1to1s_fs_diff_crop = Ty_1to1s_fs_diff[crop_start:crop_end,crop_start:crop_end]*1e-3
T_1to1s_fs_diff_crop = T_1to1s_fs_diff[crop_start:crop_end,crop_start:crop_end]*1e-3

Tx_1to1d_hs_diff_crop = Tx_1to1d_hs_diff[crop_start:crop_end,crop_start:crop_end]*1e-3
Ty_1to1d_hs_diff_crop = Ty_1to1d_hs_diff[crop_start:crop_end,crop_start:crop_end]*1e-3
T_1to1d_hs_diff_crop = T_1to1d_hs_diff[crop_start:crop_end,crop_start:crop_end]*1e-3


Tx_1to1s_hs_diff_crop = Tx_1to1s_hs_diff[crop_start:crop_end,crop_start:crop_end]*1e-3
Ty_1to1s_hs_diff_crop = Ty_1to1s_hs_diff[crop_start:crop_end,crop_start:crop_end]*1e-3
T_1to1s_hs_diff_crop = T_1to1s_hs_diff[crop_start:crop_end,crop_start:crop_end]*1e-3




# set up plot parameters
#*****************************************************************************
n=4 # every nth arrow will be plotted
pixelsize = 0.864 # in µm 
pmax = 0.2 # kPa
pmin = -0.2


# create x- and y-axis for plotting maps 
x_end = np.shape(T_1to1d_fs_diff_crop)[1]
y_end = np.shape(T_1to1d_fs_diff_crop)[0]
extent = [0, x_end*pixelsize, 0, y_end*pixelsize] 

# create mesh for vectorplot    
xq, yq = np.meshgrid(np.linspace(0,extent[1],x_end), np.linspace(0,extent[3],y_end)) 

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(3.5, 3))

im = axes[0,0].imshow(T_1to1d_fs_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent, vmin=pmin, vmax=pmax, aspect='auto')
axes[0,0].quiver(xq[::n,::n], yq[::n,::n], Tx_1to1d_fs_diff_crop[::n,::n],Ty_1to1d_fs_diff_crop[::n,::n],angles='xy',scale = 1, units='width', color="r")
# axes[0,0].set_title('n=1', pad=-400, color='r')

axes[0,1].imshow(T_1to1d_hs_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent, vmin=pmin, vmax=pmax, aspect='auto')
axes[0,1].quiver(xq[::n,::n], yq[::n,::n], Tx_1to1d_hs_diff_crop[::n,::n],Ty_1to1d_hs_diff_crop[::n,::n],angles='xy',scale = 1, units='width', color="r")

im = axes[1,0].imshow(T_1to1s_fs_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent, vmin=pmin, vmax=pmax, aspect='auto')
axes[1,0].quiver(xq[::n,::n], yq[::n,::n], Tx_1to1s_fs_diff_crop[::n,::n],Ty_1to1s_fs_diff_crop[::n,::n],angles='xy',scale = 1, units='width', color="r")
# axes[0,0].set_title('n=1', pad=-400, color='r')

axes[1,1].imshow(T_1to1s_hs_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent, vmin=pmin, vmax=pmax, aspect='auto')
axes[1,1].quiver(xq[::n,::n], yq[::n,::n], Tx_1to1s_hs_diff_crop[::n,::n],Ty_1to1s_hs_diff_crop[::n,::n],angles='xy',scale = 1, units='width', color="r")

# adjust space in between plots
plt.subplots_adjust(wspace=0.02, hspace=-0.02)


# remove axes
for ax in axes.flat:
    ax.axis('off')
    aspectratio=1.0
    ratio_default=(ax.get_xlim()[1]-ax.get_xlim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0])
    ax.set_aspect(ratio_default*aspectratio)


# add colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist())
cbar.ax.set_title('kPa')

plt.suptitle('$\mathrm{\Delta}$ Traction forces',y=0.94, x=0.44)


# make some annotations
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}

plt.figure(1).texts[0].set_position([0.440000, 0.98])
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[1].new
plt.figure(1).texts[1].set_position([0.239612, 0.840355])
plt.figure(1).texts[1].set_text("n=40")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[2].new
plt.figure(1).texts[2].set_position([0.13, 0.889135])
plt.figure(1).texts[2].set_text("global activation")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[3].new
plt.figure(1).texts[3].set_position([0.239612, 0.471175])
plt.figure(1).texts[3].set_text("n=16")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[4].new
plt.figure(1).texts[4].set_position([0.455, 0.889135])
plt.figure(1).texts[4].set_text("local activation")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[5].new
plt.figure(1).texts[5].set_position([0.553555, 0.840355])
plt.figure(1).texts[5].set_text("n=28")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[6].new
plt.figure(1).texts[6].set_position([0.555402, 0.471175])
plt.figure(1).texts[6].set_text("n=36")
#% end: automatic generated code from pylustrator


# save figure
fig.savefig(figfolder+'B.png', dpi=300, bbox_inches="tight")

plt.show()


#%% plot figure 3C, Relative strain energy over time

# define plot parameters
fig = plt.figure(2, figsize=(5, 3))             # figuresize in inches
gs = gridspec.GridSpec(2,3)                     # sets up subplotgrid rows by columns
gs.update(wspace=0.5, hspace=0.5)              # adjusts space in between the boxes in the grid
linewidth_bp = 0.7                              # linewidth of boxplot borders
width = 0.3                                     # width of boxplots
dotsize = 2                                     # size of datapoints in swarmplot
linewidth_sw = 0.3                              # linewidth of boxplot borders
alpha_sw = 1                                    # transparency of dots in swarmplot
alpha_bp = 0.8                                  # transparency of boxplots
ylabeloffset = 1                                # adjusts distance of ylabel to the plot
xlabeloffset = 1                                # adjusts distance of ylabel to the plot
titleoffset = 5                                 # adjusts distance of title to the plot
optolinewidth = 0.1                             # adjusts the linewidth of the annotations that represent the optogenetic activation
##############################################################################
#Generate first panel
##############################################################################

ymin = 0.9
ymax = 1.3

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[0,0])

# set plot variables
x = np.arange(60)
y = AR1to1d_fullstim_long["TFM_data"]["relEs"]
x = x[::2] # downsample data for nicer plotting
y = y[::2,:]
y_mean = np.nanmean(y,axis=1)
y_std = np.nanstd(y,axis=1)
y_sem = y_std/np.sqrt(np.shape(y)[1])

# create box- and swarmplots
fig_ax.errorbar(x,y_mean,yerr=y_sem, mfc='w', color=colors_parent[1], marker='o',ms=2, linewidth=0.5, ls='none',markeredgewidth=0.5)
     
# set labels
fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
fig_ax.set_ylabel(ylabel='doublet', labelpad=ylabeloffset)
# fig_ax.set_title(label='relative $\mathrm{E_s}$', pad=titleoffset)
fig_ax.set_title(label='global activation', pad=titleoffset)

# add anotations for opto pulses
for i in np.arange(10):
    plt.axline((20+i,ymin),(20+i,ymax),linewidth=optolinewidth, color="cyan")
    
# Define where you want ticks
xticks = np.arange(0,61,20)
yticks = np.arange(0.9,1.31,0.1)

plt.xticks(xticks)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=True, top=False, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=True, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
#Generate second panel
##############################################################################

ymin = 0.9
ymax = 1.3

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[0,1])

# set plot variables
x = np.arange(60)
y = AR1to1d_halfstim["TFM_data"]["relEs"]
x = x[::2] # downsample data for nicer plotting
y = y[::2,:]
y_mean = np.nanmean(y,axis=1)
y_std = np.nanstd(y,axis=1)
y_sem = y_std/np.sqrt(np.shape(y)[1])

# create box- and swarmplots
fig_ax.errorbar(x,y_mean,yerr=y_sem, mfc='w', color=colors_parent[1], marker='o',ms=2, linewidth=0.5, ls='none',markeredgewidth=0.5)
     
# set labels
fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
# fig_ax.set_ylabel(ylabel='doublet', labelpad=ylabeloffset)
# fig_ax.set_title(label='relative $\mathrm{E_s}$', pad=titleoffset)
fig_ax.set_title(label='local activation', pad=titleoffset)
fig_ax.set()

# add anotations for opto pulses
for i in np.arange(10):
    plt.axline((20+i,ymin),(20+i,ymax),linewidth=optolinewidth, color="cyan")

# Define where you want ticks
xticks = np.arange(0,61,20)
yticks = np.arange(0.9,1.31,0.1)

plt.xticks(xticks)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=True, top=False, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=True, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
#Generate third panel
##############################################################################

ymin = 0.9
ymax = 1.3

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[1,0])

# set plot variables
x = np.arange(60)
y = AR1to1s_fullstim_long["TFM_data"]["relEs"]
x = x[::2] # downsample data for nicer plotting
y = y[::2,:]
y_mean = np.nanmean(y,axis=1)
y_std = np.nanstd(y,axis=1)
y_sem = y_std/np.sqrt(np.shape(y)[1])

# create box- and swarmplots
fig_ax.errorbar(x,y_mean,yerr=y_sem, mfc='w', color=colors_parent[2], marker='o',ms=2, linewidth=0.5, ls='none',markeredgewidth=0.5)
     
# set labels
fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
fig_ax.set_ylabel(ylabel='singlet', labelpad=ylabeloffset)
# fig_ax.set_title(label='relative $\mathrm{E_s}$', pad=titleoffset)
fig_ax.set()

# add anotations for opto pulses
for i in np.arange(10):
    plt.axline((20+i,ymin),(20+i,ymax),linewidth=optolinewidth, color="cyan")

# Define where you want ticks
xticks = np.arange(0,61,20)
yticks = np.arange(0.9,1.31,0.1)

plt.xticks(xticks)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=True, top=False, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=True, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
#Generate fourth panel
##############################################################################

ymin = 0.9
ymax = 1.3

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[1,1])

# set plot variables
x = np.arange(60)
y = AR1to1s_halfstim["TFM_data"]["relEs"]
x = x[::2] # downsample data for nicer plotting
y = y[::2,:]
y_mean = np.nanmean(y,axis=1)
y_std = np.nanstd(y,axis=1)
y_sem = y_std/np.sqrt(np.shape(y)[1])

# create box- and swarmplots
fig_ax.errorbar(x,y_mean,yerr=y_sem, mfc='w', color=colors_parent[2], marker='o',ms=2, linewidth=0.5, ls='none',markeredgewidth=0.5)
     
# set labels
fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
# fig_ax.set_ylabel(ylabel='singlet', labelpad=ylabeloffset)
# fig_ax.set_title(label='relative $\mathrm{E_s}$', pad=titleoffset)
fig_ax.set()

# add anotations for opto pulses
for i in np.arange(10):
    plt.axline((20+i,ymin),(20+i,ymax),linewidth=optolinewidth, color="cyan")

# Define where you want ticks
xticks = np.arange(0,61,20)
yticks = np.arange(0.9,1.31,0.1)

plt.xticks(xticks)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=True, top=False, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=True, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
#Generate fifth panel
##############################################################################
colors = [colors_parent[1],colors_parent[1]];   # defines colors
sns.set_palette(sns.color_palette(colors))      # sets colors
ylabeloffset = -1

# extract data from dataframe to test if their distribution is gaussian
# data_1to1d = df_fs[df_fs["keys"]=="AR1to1d_fs"]["REI"].to_numpy()
# data_1to1s = df_fs[df_fs["keys"]=="AR1to1s_fs"]["REI"].to_numpy()
# if test_if_gaussian(data_1to1d,data_1to1s,'REI'):
#     test = 't-test_ind'
# else:
test = 'Mann-Whitney'

ymin = -0.2
ymax = 0.8
stat_annotation_offset = 0

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[0,2])

# set plot variables
x = 'keys'
y = 'REI'

# create box- and swarmplots
sns.swarmplot(x=x, y=y, data=df_doublet, ax=fig_ax,alpha=alpha_sw,linewidth=linewidth_sw, zorder=0, size=dotsize)
bp = sns.boxplot(x=x, y=y, data=df_doublet, ax=fig_ax,linewidth=linewidth_bp,notch=True, showfliers = False, width=width)

order = ['AR1to1d_fs', 'AR1to1s_fs']
add_stat_annotation(bp, data=df_fs, x=x, y=y, order=order, box_pairs=[('AR1to1d_fs', 'AR1to1s_fs')], 
                    line_offset_to_box=stat_annotation_offset, test=test, text_format='star', loc='inside', verbose=2)

# make boxplots transparent
for patch in bp.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, alpha_bp))

plt.setp(bp.artists, edgecolor = 'k')
plt.setp(bp.lines, color='k')
     
# set labels
fig_ax.set_xticklabels(['global \n act.', 'local \n act.'])
fig_ax.set_xlabel(xlabel=None)
fig_ax.set_ylabel(ylabel=None, labelpad=ylabeloffset)
# fig_ax.set_title(label='REI', pad=titleoffset)
fig_ax.set()

# Define where you want ticks
yticks = np.arange(-0.2,0.81,0.2)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=False, top=False, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=False, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
#Generate sixth panel
##############################################################################
colors = [colors_parent[2],colors_parent[2]];   # defines colors
sns.set_palette(sns.color_palette(colors))      # sets colors

# extract data from dataframe to test if their distribution is gaussian
data_1to1d = df_hs[df_hs["keys"]=="AR1to1d_hs"]["REI"].to_numpy()
data_1to1s = df_hs[df_hs["keys"]=="AR1to1s_hs"]["REI"].to_numpy()
# if test_if_gaussian(data_1to1d,data_1to1s,'REI'):
#     test = 't-test_ind'
# else:
test = 'Mann-Whitney'

ymin = -0.2
ymax = 0.8
stat_annotation_offset = 0.5

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[1,2])

# set plot variables
x = 'keys'
y = 'REI'

# create box- and swarmplots
sns.swarmplot(x=x, y=y, data=df_singlet, ax=fig_ax,alpha=alpha_sw,linewidth=linewidth_sw, zorder=0, size=dotsize)
bp = sns.boxplot(x=x, y=y, data=df_singlet, ax=fig_ax,linewidth=linewidth_bp,notch=True, showfliers = False, width=width)

order = ['AR1to1d_hs', 'AR1to1s_hs']
add_stat_annotation(bp, data=df_hs, x=x, y=y, order=order, box_pairs=[('AR1to1d_hs', 'AR1to1s_hs')], 
                    line_offset_to_box=stat_annotation_offset, test=test, text_format='star', loc='inside', verbose=2)

# make boxplots transparent
for patch in bp.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, alpha_bp))

plt.setp(bp.artists, edgecolor = 'k')
plt.setp(bp.lines, color='k')
     
# set labels
fig_ax.set_xticklabels(['global \n act.', 'local \n act.'])
fig_ax.set_xlabel(xlabel=None)
fig_ax.set_ylabel(ylabel=None, labelpad=ylabeloffset)
fig_ax.set_title(label=None, pad=titleoffset)
fig_ax.set()

# Define where you want ticks
yticks = np.arange(-0.2,0.81,0.2)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=False, top=False, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=False, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)


# write title for panels 1 to 4
plt.text(-5.7,2.575947,'Relative strain energy',fontsize=10)
# write title for panels 5 to 6
plt.text(-0.662416, 2.400220,'Relative energy \n     increase',fontsize=10)
# save plot to file
plt.savefig(figfolder+'C.png', dpi=300, bbox_inches="tight")

plt.show()


#%% plot figure 3D, stress map difference fullstim

# prepare data first

# concatenate TFM maps from different experiments and calculate average maps over first 20 frames and all cells to get average maps
sigmaxx_1to1d_diff = np.nanmean(AR1to1d_fullstim_long["MSM_data"]["sigma_xx"][:,:,33,:]-AR1to1d_fullstim_long["MSM_data"]["sigma_xx"][:,:,20,:],axis=2)
sigmayy_1to1d_diff = np.nanmean(AR1to1d_fullstim_long["MSM_data"]["sigma_yy"][:,:,33,:]-AR1to1d_fullstim_long["MSM_data"]["sigma_yy"][:,:,20,:],axis=2)
sigmaxx_1to1s_diff = np.nanmean(AR1to1s_fullstim_long["MSM_data"]["sigma_xx"][:,:,33,:]-AR1to1s_fullstim_long["MSM_data"]["sigma_xx"][:,:,20,:],axis=2)
sigmayy_1to1s_diff = np.nanmean(AR1to1s_fullstim_long["MSM_data"]["sigma_yy"][:,:,33,:]-AR1to1s_fullstim_long["MSM_data"]["sigma_yy"][:,:,20,:],axis=2)


# crop maps 
crop_start = 8
crop_end = 84

sigmaxx_1to1d_diff_crop = sigmaxx_1to1d_diff[crop_start:crop_end,crop_start:crop_end]*1e3 # convert to mN/m
sigmayy_1to1d_diff_crop = sigmayy_1to1d_diff[crop_start:crop_end,crop_start:crop_end]*1e3
sigmaxx_1to1s_diff_crop = sigmaxx_1to1s_diff[crop_start:crop_end,crop_start:crop_end]*1e3
sigmayy_1to1s_diff_crop = sigmayy_1to1s_diff[crop_start:crop_end,crop_start:crop_end]*1e3



# set up plot parameters
#*****************************************************************************

pixelsize = 0.864 # in µm 
sigma_max = 1 # kPa
sigma_min = -1 # kPa

# create x- and y-axis for plotting maps 
x_end = np.shape(sigmaxx_1to1d_diff_crop)[1]
y_end = np.shape(sigmaxx_1to1d_diff_crop)[0]
extent = [0, x_end*pixelsize, 0, y_end*pixelsize] 

# create mesh for vectorplot    
xq, yq = np.meshgrid(np.linspace(0,extent[1],x_end), np.linspace(0,extent[3],y_end)) 

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(3.5, 3))

im = axes[0,0].imshow(sigmaxx_1to1d_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent, vmin=sigma_min, vmax=sigma_max, aspect='auto')

axes[0,1].imshow(sigmayy_1to1d_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent, vmin=sigma_min, vmax=sigma_max, aspect='auto')

axes[1,0].imshow(sigmaxx_1to1s_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent, vmin=sigma_min, vmax=sigma_max, aspect='auto')

axes[1,1].imshow(sigmayy_1to1s_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent, vmin=sigma_min, vmax=sigma_max, aspect='auto')

# adjust space in between plots
plt.subplots_adjust(wspace=0.02, hspace=-0.02)


# remove axes
for ax in axes.flat:
    ax.axis('off')
    aspectratio=1.0
    ratio_default=(ax.get_xlim()[1]-ax.get_xlim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0])
    ax.set_aspect(ratio_default*aspectratio)


# add colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist())
cbar.ax.set_title('mN/m')

# add title
plt.suptitle('$\mathrm{\Delta}$ Stresses, global activation',y=0.94, x=0.44)

# make some annotations
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}

plt.figure(1).texts[0].set_position([0.440000, 0.98])
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[1].new
plt.figure(1).texts[1].set_position([0.239612, 0.840355])
plt.figure(1).texts[1].set_text("n=40")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[2].new
plt.figure(1).texts[2].set_position([0.2, 0.889135])
plt.figure(1).texts[2].set_text("xx-Stress")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[3].new
plt.figure(1).texts[3].set_position([0.239612, 0.471175])
plt.figure(1).texts[3].set_text("n=16")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[4].new
plt.figure(1).texts[4].set_position([0.5, 0.889135])
plt.figure(1).texts[4].set_text("yy-Stress")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[5].new
plt.figure(1).texts[5].set_position([0.553555, 0.840355])
plt.figure(1).texts[5].set_text("n=28")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[6].new
plt.figure(1).texts[6].set_position([0.555402, 0.471175])
plt.figure(1).texts[6].set_text("n=36")
#% end: automatic generated code from pylustrator


# save figure
fig.savefig(figfolder+'D.png', dpi=300, bbox_inches="tight")
plt.show()

#%% plot figure 3E, Relative stress over time fullstim

# define plot parameters
fig = plt.figure(2, figsize=(5, 3))             # figuresize in inches
gs = gridspec.GridSpec(2,3)                     # sets up subplotgrid rows by columns
gs.update(wspace=0.5, hspace=0.5)              # adjusts space in between the boxes in the grid
linewidth_bp = 0.7                              # linewidth of boxplot borders
width = 0.3                                     # width of boxplots
dotsize = 2                                     # size of datapoints in swarmplot
linewidth_sw = 0.3                              # linewidth of boxplot borders
alpha_sw = 1                                    # transparency of dots in swarmplot
alpha_bp = 0.8                                  # transparency of boxplots
ylabeloffset = 1                                # adjusts distance of ylabel to the plot
xlabeloffset = 1                                # adjusts distance of ylabel to the plot
titleoffset = 5                                 # adjusts distance of title to the plot
optolinewidth = 0.1                             # adjusts the linewidth of the annotations that represent the optogenetic activation
##############################################################################
#Generate first panel
##############################################################################

ymin = 0.9
ymax = 1.3

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[0,0])

# set plot variables
x = np.arange(60)
y = AR1to1d_fullstim_long["MSM_data"]["relsigma_xx_average"]
x = x[::2] # downsample data for nicer plotting
y = y[::2,:]
y_mean = np.nanmean(y,axis=1)
y_std = np.nanstd(y,axis=1)
y_sem = y_std/np.sqrt(np.shape(y)[1])

# create box- and swarmplots
fig_ax.errorbar(x,y_mean,yerr=y_sem, mfc='w', color=colors_parent[1], marker='o',ms=2, linewidth=0.5, ls='none',markeredgewidth=0.5)
     
# set labels
fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
fig_ax.set_ylabel(ylabel='doublet', labelpad=ylabeloffset)
# fig_ax.set_title(label='relative $\mathrm{E_s}$', pad=titleoffset)
fig_ax.set_title(label='xx-Stress', pad=titleoffset)

# add anotations for opto pulses
for i in np.arange(10):
    plt.axline((20+i,ymin),(20+i,ymax),linewidth=optolinewidth, color="cyan")
    
# Define where you want ticks
xticks = np.arange(0,61,20)
yticks = np.arange(0.9,1.31,0.1)

plt.xticks(xticks)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=True, top=False, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=True, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
#Generate second panel
##############################################################################

ymin = 0.9
ymax = 1.3

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[0,1])

# set plot variables
x = np.arange(60)
y = AR1to1d_fullstim_long["MSM_data"]["relsigma_yy_average"]
x = x[::2] # downsample data for nicer plotting
y = y[::2,:]
y_mean = np.nanmean(y,axis=1)
y_std = np.nanstd(y,axis=1)
y_sem = y_std/np.sqrt(np.shape(y)[1])

# create box- and swarmplots
fig_ax.errorbar(x,y_mean,yerr=y_sem, mfc='w', color=colors_parent[1], marker='o',ms=2, linewidth=0.5, ls='none',markeredgewidth=0.5)
     
# set labels
fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
# fig_ax.set_ylabel(ylabel='doublet', labelpad=ylabeloffset)
# fig_ax.set_title(label='relative $\mathrm{E_s}$', pad=titleoffset)
fig_ax.set_title(label='yy-Stress', pad=titleoffset)
fig_ax.set()

# add anotations for opto pulses
for i in np.arange(10):
    plt.axline((20+i,ymin),(20+i,ymax),linewidth=optolinewidth, color="cyan")

# Define where you want ticks
xticks = np.arange(0,61,20)
yticks = np.arange(0.9,1.31,0.1)

plt.xticks(xticks)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=True, top=False, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=True, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
#Generate third panel
##############################################################################

ymin = 0.9
ymax = 1.3

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[1,0])

# set plot variables
x = np.arange(60)
y = AR1to1s_fullstim_long["MSM_data"]["relsigma_xx_average"]
x = x[::2] # downsample data for nicer plotting
y = y[::2,:]
y_mean = np.nanmean(y,axis=1)
y_std = np.nanstd(y,axis=1)
y_sem = y_std/np.sqrt(np.shape(y)[1])

# create box- and swarmplots
fig_ax.errorbar(x,y_mean,yerr=y_sem, mfc='w', color=colors_parent[2], marker='o',ms=2, linewidth=0.5, ls='none',markeredgewidth=0.5)
     
# set labels
fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
fig_ax.set_ylabel(ylabel='singlet', labelpad=ylabeloffset)
# fig_ax.set_title(label='relative $\mathrm{E_s}$', pad=titleoffset)
fig_ax.set()

# add anotations for opto pulses
for i in np.arange(10):
    plt.axline((20+i,ymin),(20+i,ymax),linewidth=optolinewidth, color="cyan")

# Define where you want ticks
xticks = np.arange(0,61,20)
yticks = np.arange(0.9,1.31,0.1)

plt.xticks(xticks)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=True, top=False, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=True, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
#Generate fourth panel
##############################################################################

ymin = 0.9
ymax = 1.3

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[1,1])

# set plot variables
x = np.arange(60)
y = AR1to1s_fullstim_long["MSM_data"]["relsigma_yy_average"]
x = x[::2] # downsample data for nicer plotting
y = y[::2,:]
y_mean = np.nanmean(y,axis=1)
y_std = np.nanstd(y,axis=1)
y_sem = y_std/np.sqrt(np.shape(y)[1])

# create box- and swarmplots
fig_ax.errorbar(x,y_mean,yerr=y_sem, mfc='w', color=colors_parent[2], marker='o',ms=2, linewidth=0.5, ls='none',markeredgewidth=0.5)
     
# set labels
fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
# fig_ax.set_ylabel(ylabel='singlet', labelpad=ylabeloffset)
# fig_ax.set_title(label='relative $\mathrm{E_s}$', pad=titleoffset)
fig_ax.set()

# add anotations for opto pulses
for i in np.arange(10):
    plt.axline((20+i,ymin),(20+i,ymax),linewidth=optolinewidth, color="cyan")

# Define where you want ticks
xticks = np.arange(0,61,20)
yticks = np.arange(0.9,1.31,0.1)

plt.xticks(xticks)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=True, top=False, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=True, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
#Generate fifth panel
##############################################################################
colors = [colors_parent[1],colors_parent[1]];   # defines colors
sns.set_palette(sns.color_palette(colors))      # sets colors
ylabeloffset = -1

# extract data from dataframe to test if their distribution is gaussian
# data_1to1d = df_fs[df_fs["keys"]=="AR1to1d_fs"]["RSI_xx"].to_numpy()
# data_1to1s = df_fs[df_fs["keys"]=="AR1to1s_fs"]["RSI_xx"].to_numpy()
# if test_if_gaussian(data_1to1d,data_1to1s,'REI'):
#     test = 't-test_ind'
# else:
test = 'Mann-Whitney'

ymin = -0.2
ymax = 0.6
stat_annotation_offset = 0

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[0,2])

# a little weird way to get the dataframe format I need for seaborn boxplots...
df1 = pd.DataFrame(df_d_fs['RSI_xx'])
df2 = pd.DataFrame(df_d_fs['RSI_yy'])
df1 = df1.transpose().reset_index(drop=True).transpose()
df2 = df2.transpose().reset_index(drop=True).transpose()

df_plot = pd.concat([df1,df2],axis=0)

df_plot.rename( columns={0:'sigma'}, inplace=True )

keys_sx = ['sigma_xx' for i in range(n_d_fullstim)]
keys_sy = ['sigma_yy' for i in range(n_d_fullstim)]
keys = np.concatenate((keys_sx, keys_sy))
df_plot['keys'] = keys


# create box- and swarmplots
sns.swarmplot(x='keys', y='sigma', data=df_plot, ax=fig_ax,alpha=alpha_sw,linewidth=linewidth_sw, zorder=0, size=dotsize)
bp = sns.boxplot(x='keys', y='sigma', data=df_plot, ax=fig_ax,linewidth=linewidth_bp,notch=True, showfliers = False, width=width)

# order = ['AR1to1d_fs', 'AR1to1s_fs']
# add_stat_annotation(bp, data=df_fs, x=x, y=y, order=order, box_pairs=[('AR1to1d_fs', 'AR1to1s_fs')], 
#                     line_offset_to_box=stat_annotation_offset, test=test, text_format='star', loc='inside', verbose=2)

# make boxplots transparent
for patch in bp.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, alpha_bp))

plt.setp(bp.artists, edgecolor = 'k')
plt.setp(bp.lines, color='k')
     
# set labels
fig_ax.set_xticklabels(['global \n act.', 'local \n act.'])
fig_ax.set_xlabel(xlabel=None)
fig_ax.set_ylabel(ylabel=None, labelpad=ylabeloffset)
# fig_ax.set_title(label='REI', pad=titleoffset)
fig_ax.set()

# Define where you want ticks
yticks = np.arange(-0.2,0.61,0.2)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=False, top=False, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=False, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
#Generate sixth panel
##############################################################################
colors = [colors_parent[2],colors_parent[2]];   # defines colors
sns.set_palette(sns.color_palette(colors))      # sets colors

# extract data from dataframe to test if their distribution is gaussian
# data_1to1d = df_hs[df_hs["keys"]=="AR1to1d_hs"]["REI"].to_numpy()
# data_1to1s = df_hs[df_hs["keys"]=="AR1to1s_hs"]["REI"].to_numpy()
# if test_if_gaussian(data_1to1d,data_1to1s,'REI'):
#     test = 't-test_ind'
# else:
test = 'Mann-Whitney'

ymin = -0.2
ymax = 0.6
stat_annotation_offset = 0.5

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[1,2])

# a little weird way to get the dataframe format I need for seaborn boxplots...
df1 = pd.DataFrame(df_s_fs['RSI_xx'])
df2 = pd.DataFrame(df_s_fs['RSI_yy'])
df1 = df1.transpose().reset_index(drop=True).transpose()
df2 = df2.transpose().reset_index(drop=True).transpose()

df_plot = pd.concat([df1,df2],axis=0)

df_plot.rename( columns={0:'sigma'}, inplace=True )

keys_sx = ['sigma_xx' for i in range(n_s_fullstim)]
keys_sy = ['sigma_yy' for i in range(n_s_fullstim)]
keys = np.concatenate((keys_sx, keys_sy))
df_plot['keys'] = keys


# create box- and swarmplots
sns.swarmplot(x='keys', y='sigma', data=df_plot, ax=fig_ax,alpha=alpha_sw,linewidth=linewidth_sw, zorder=0, size=dotsize)
bp = sns.boxplot(x='keys', y='sigma', data=df_plot, ax=fig_ax,linewidth=linewidth_bp,notch=True, showfliers = False, width=width)

# order = ['AR1to1d_hs', 'AR1to1s_hs']
# add_stat_annotation(bp, data=df_hs, x=x, y=y, order=order, box_pairs=[('AR1to1d_hs', 'AR1to1s_hs')], 
#                     line_offset_to_box=stat_annotation_offset, test=test, text_format='star', loc='inside', verbose=2)

# make boxplots transparent
for patch in bp.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, alpha_bp))

plt.setp(bp.artists, edgecolor = 'k')
plt.setp(bp.lines, color='k')
     
# set labels
fig_ax.set_xticklabels(['global \n act.', 'local \n act.'])
fig_ax.set_xlabel(xlabel=None)
fig_ax.set_ylabel(ylabel=None, labelpad=ylabeloffset)
fig_ax.set_title(label=None, pad=titleoffset)
fig_ax.set()

# Define where you want ticks
yticks = np.arange(-0.2,0.61,0.2)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=False, top=False, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=False, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)


# write title for panels 1 to 4
plt.text(-5.7,2.575947,'Relative strain energy',fontsize=10)
# write title for panels 5 to 6
plt.text(-0.662416, 2.400220,'Relative energy \n     increase',fontsize=10)
# save plot to file
plt.savefig(figfolder+'E.png', dpi=300, bbox_inches="tight")

plt.show()
#%% plot figure 3F, stress map difference halfstim

# prepare data first

# concatenate TFM maps from different experiments and calculate average maps over first 20 frames and all cells to get average maps
sigmaxx_1to1d_diff = np.nanmean(AR1to1d_halfstim["MSM_data"]["sigma_xx"][:,:,33,:]-AR1to1d_halfstim["MSM_data"]["sigma_xx"][:,:,20,:],axis=2)
sigmayy_1to1d_diff = np.nanmean(AR1to1d_halfstim["MSM_data"]["sigma_yy"][:,:,33,:]-AR1to1d_halfstim["MSM_data"]["sigma_yy"][:,:,20,:],axis=2)
sigmaxx_1to1s_diff = np.nanmean(AR1to1s_halfstim["MSM_data"]["sigma_xx"][:,:,33,:]-AR1to1s_halfstim["MSM_data"]["sigma_xx"][:,:,20,:],axis=2)
sigmayy_1to1s_diff = np.nanmean(AR1to1s_halfstim["MSM_data"]["sigma_yy"][:,:,33,:]-AR1to1s_halfstim["MSM_data"]["sigma_yy"][:,:,20,:],axis=2)


# crop maps 
crop_start = 8
crop_end = 84

sigmaxx_1to1d_diff_crop = sigmaxx_1to1d_diff[crop_start:crop_end,crop_start:crop_end]*1e3 # convert to mN/m
sigmayy_1to1d_diff_crop = sigmayy_1to1d_diff[crop_start:crop_end,crop_start:crop_end]*1e3
sigmaxx_1to1s_diff_crop = sigmaxx_1to1s_diff[crop_start:crop_end,crop_start:crop_end]*1e3
sigmayy_1to1s_diff_crop = sigmayy_1to1s_diff[crop_start:crop_end,crop_start:crop_end]*1e3



# set up plot parameters
#*****************************************************************************

pixelsize = 0.864 # in µm 
sigma_max = 1 # kPa
sigma_min = -1 # kPa

# create x- and y-axis for plotting maps 
x_end = np.shape(sigmaxx_1to1d_diff_crop)[1]
y_end = np.shape(sigmaxx_1to1d_diff_crop)[0]
extent = [0, x_end*pixelsize, 0, y_end*pixelsize] 

# create mesh for vectorplot    
xq, yq = np.meshgrid(np.linspace(0,extent[1],x_end), np.linspace(0,extent[3],y_end)) 

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(3.5, 3))

im = axes[0,0].imshow(sigmaxx_1to1d_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent, vmin=sigma_min, vmax=sigma_max, aspect='auto')

axes[0,1].imshow(sigmayy_1to1d_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent, vmin=sigma_min, vmax=sigma_max, aspect='auto')

axes[1,0].imshow(sigmaxx_1to1s_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent, vmin=sigma_min, vmax=sigma_max, aspect='auto')

axes[1,1].imshow(sigmayy_1to1s_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent, vmin=sigma_min, vmax=sigma_max, aspect='auto')

# adjust space in between plots
plt.subplots_adjust(wspace=0.02, hspace=-0.02)

# axes[0,0].set_xlabel("lol")
# # add annotations
plt.text(-50,120,'sigmaxx',color = 'k')
plt.text(20,120,'sigmayy',color = 'k')
# plt.text(-40.5,119,'n=1',color = 'white')
# plt.text(23,55.5,'n=101',color = 'white')
# plt.text(23.5,119,'n=66',color = 'white')

# remove axes
for ax in axes.flat:
    ax.axis('off')
    aspectratio=1.0
    ratio_default=(ax.get_xlim()[1]-ax.get_xlim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0])
    ax.set_aspect(ratio_default*aspectratio)


# add colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist())
cbar.ax.set_title('mN/m')

# add title
plt.suptitle('$\mathrm{\Delta}$ Stresses, local activation',y=0.94, x=0.44)

# make some annotations
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}

plt.figure(1).texts[0].set_position([0.440000, 0.98])
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[1].new
plt.figure(1).texts[1].set_position([0.239612, 0.840355])
plt.figure(1).texts[1].set_text("n=40")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[2].new
plt.figure(1).texts[2].set_position([0.2, 0.889135])
plt.figure(1).texts[2].set_text("xx-Stress")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[3].new
plt.figure(1).texts[3].set_position([0.239612, 0.471175])
plt.figure(1).texts[3].set_text("n=16")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[4].new
plt.figure(1).texts[4].set_position([0.5, 0.889135])
plt.figure(1).texts[4].set_text("yy-Stress")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[5].new
plt.figure(1).texts[5].set_position([0.553555, 0.840355])
plt.figure(1).texts[5].set_text("n=28")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[6].new
plt.figure(1).texts[6].set_position([0.555402, 0.471175])
plt.figure(1).texts[6].set_text("n=36")
#% end: automatic generated code from pylustrator

# save figure
fig.savefig(figfolder+'F.png', dpi=300, bbox_inches="tight")
plt.close()

#%% plot figure 3G, Relative stress over time halfstim

# define plot parameters
fig = plt.figure(2, figsize=(5, 3))             # figuresize in inches
gs = gridspec.GridSpec(2,3)                     # sets up subplotgrid rows by columns
gs.update(wspace=0.5, hspace=0.5)              # adjusts space in between the boxes in the grid
linewidth_bp = 0.7                              # linewidth of boxplot borders
width = 0.3                                     # width of boxplots
dotsize = 2                                     # size of datapoints in swarmplot
linewidth_sw = 0.3                              # linewidth of boxplot borders
alpha_sw = 1                                    # transparency of dots in swarmplot
alpha_bp = 0.8                                  # transparency of boxplots
ylabeloffset = 1                                # adjusts distance of ylabel to the plot
xlabeloffset = 1                                # adjusts distance of ylabel to the plot
titleoffset = 5                                 # adjusts distance of title to the plot
optolinewidth = 0.1                             # adjusts the linewidth of the annotations that represent the optogenetic activation
##############################################################################
#Generate first panel
##############################################################################

ymin = 0.9
ymax = 1.3

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[0,0])

# set plot variables
x = np.arange(60)
y = AR1to1d_halfstim["MSM_data"]["relsigma_xx_lefthalf_average"]
x = x[::2] # downsample data for nicer plotting
y = y[::2,:]
y_mean = np.nanmean(y,axis=1)
y_std = np.nanstd(y,axis=1)
y_sem = y_std/np.sqrt(np.shape(y)[1])

# create first plot
fig_ax.errorbar(x,y_mean,yerr=y_sem, mfc='w', color=colors_parent[1], marker='o',ms=2, linewidth=0.5, ls='none',markeredgewidth=0.5)
 # set plot variables
x = np.arange(60)
y = AR1to1d_halfstim["MSM_data"]["relsigma_xx_righthalf_average"]
x = x[::2] # downsample data for nicer plotting
y = y[::2,:]
y_mean = np.nanmean(y,axis=1)
y_std = np.nanstd(y,axis=1)
y_sem = y_std/np.sqrt(np.shape(y)[1])

# create second plot
fig_ax.errorbar(x,y_mean,yerr=y_sem, mfc='w', color=colors_parent_dark[1], marker='o',ms=2, linewidth=0.5, ls='none',markeredgewidth=0.5)
    
# set labels
fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
fig_ax.set_ylabel(ylabel='doublet', labelpad=ylabeloffset)
fig_ax.set_title(label='relative $\mathrm{E_s}$', pad=titleoffset)
fig_ax.set_title(label='xx-Stress', pad=titleoffset)

# add anotations for opto pulses
for i in np.arange(10):
    plt.axline((20+i,ymin),(20+i,ymax),linewidth=optolinewidth, color="cyan")
    
# Define where you want ticks
xticks = np.arange(0,61,20)
yticks = np.arange(0.9,1.31,0.1)

plt.xticks(xticks)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=True, top=False, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=True, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
#Generate second panel
##############################################################################

ymin = 0.9
ymax = 1.3

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[0,1])

# set plot variables
x = np.arange(60)
y = AR1to1d_halfstim["MSM_data"]["relsigma_yy_lefthalf_average"]
x = x[::2] # downsample data for nicer plotting
y = y[::2,:]
y_mean = np.nanmean(y,axis=1)
y_std = np.nanstd(y,axis=1)
y_sem = y_std/np.sqrt(np.shape(y)[1])

# create first plot
fig_ax.errorbar(x,y_mean,yerr=y_sem, mfc='w', color=colors_parent[1], marker='o',ms=2, linewidth=0.5, ls='none',markeredgewidth=0.5)
 # set plot variables
x = np.arange(60)
y = AR1to1d_halfstim["MSM_data"]["relsigma_yy_righthalf_average"]
x = x[::2] # downsample data for nicer plotting
y = y[::2,:]
y_mean = np.nanmean(y,axis=1)
y_std = np.nanstd(y,axis=1)
y_sem = y_std/np.sqrt(np.shape(y)[1])

# create second plot
fig_ax.errorbar(x,y_mean,yerr=y_sem, mfc='w', color=colors_parent_dark[1], marker='o',ms=2, linewidth=0.5, ls='none',markeredgewidth=0.5)
     
# set labels
fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
# fig_ax.set_ylabel(ylabel='doublet', labelpad=ylabeloffset)
# fig_ax.set_title(label='relative $\mathrm{E_s}$', pad=titleoffset)
fig_ax.set_title(label='yy-Stress', pad=titleoffset)
fig_ax.set()

# add anotations for opto pulses
for i in np.arange(10):
    plt.axline((20+i,ymin),(20+i,ymax),linewidth=optolinewidth, color="cyan")

# Define where you want ticks
xticks = np.arange(0,61,20)
yticks = np.arange(0.9,1.31,0.1)

plt.xticks(xticks)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=True, top=False, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=True, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
#Generate third panel
##############################################################################

ymin = 0.9
ymax = 1.3

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[1,0])

# set plot variables
x = np.arange(60)
y = AR1to1s_halfstim["MSM_data"]["relsigma_xx_lefthalf_average"]
x = x[::2] # downsample data for nicer plotting
y = y[::2,:]
y_mean = np.nanmean(y,axis=1)
y_std = np.nanstd(y,axis=1)
y_sem = y_std/np.sqrt(np.shape(y)[1])

# create first plot
fig_ax.errorbar(x,y_mean,yerr=y_sem, mfc='w', color=colors_parent[2], marker='o',ms=2, linewidth=0.5, ls='none',markeredgewidth=0.5)
 # set plot variables
x = np.arange(60)
y = AR1to1s_halfstim["MSM_data"]["relsigma_xx_righthalf_average"]
x = x[::2] # downsample data for nicer plotting
y = y[::2,:]
y_mean = np.nanmean(y,axis=1)
y_std = np.nanstd(y,axis=1)
y_sem = y_std/np.sqrt(np.shape(y)[1])

# create second plot
fig_ax.errorbar(x,y_mean,yerr=y_sem, mfc='w', color=colors_parent_dark[2], marker='o',ms=2, linewidth=0.5, ls='none',markeredgewidth=0.5)
     
# set labels
fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
fig_ax.set_ylabel(ylabel='singlet', labelpad=ylabeloffset)
# fig_ax.set_title(label='relative $\mathrm{E_s}$', pad=titleoffset)
fig_ax.set()

# add anotations for opto pulses
for i in np.arange(10):
    plt.axline((20+i,ymin),(20+i,ymax),linewidth=optolinewidth, color="cyan")

# Define where you want ticks
xticks = np.arange(0,61,20)
yticks = np.arange(0.9,1.31,0.1)

plt.xticks(xticks)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=True, top=False, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=True, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
#Generate fourth panel
##############################################################################

ymin = 0.9
ymax = 1.3

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[1,1])

# set plot variables
x = np.arange(60)
y = AR1to1s_halfstim["MSM_data"]["relsigma_yy_lefthalf_average"]
x = x[::2] # downsample data for nicer plotting
y = y[::2,:]
y_mean = np.nanmean(y,axis=1)
y_std = np.nanstd(y,axis=1)
y_sem = y_std/np.sqrt(np.shape(y)[1])

# create first plot
fig_ax.errorbar(x,y_mean,yerr=y_sem, mfc='w', color=colors_parent[2], marker='o',ms=2, linewidth=0.5, ls='none',markeredgewidth=0.5)
 # set plot variables
x = np.arange(60)
y = AR1to1s_halfstim["MSM_data"]["relsigma_yy_righthalf_average"]
x = x[::2] # downsample data for nicer plotting
y = y[::2,:]
y_mean = np.nanmean(y,axis=1)
y_std = np.nanstd(y,axis=1)
y_sem = y_std/np.sqrt(np.shape(y)[1])

# create second plot
fig_ax.errorbar(x,y_mean,yerr=y_sem, mfc='w', color=colors_parent_dark[2], marker='o',ms=2, linewidth=0.5, ls='none',markeredgewidth=0.5)

# set labels
fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
# fig_ax.set_ylabel(ylabel='singlet', labelpad=ylabeloffset)
# fig_ax.set_title(label='relative $\mathrm{E_s}$', pad=titleoffset)
fig_ax.set()

# add anotations for opto pulses
for i in np.arange(10):
    plt.axline((20+i,ymin),(20+i,ymax),linewidth=optolinewidth, color="cyan")

# Define where you want ticks
xticks = np.arange(0,61,20)
yticks = np.arange(0.9,1.31,0.1)

plt.xticks(xticks)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=True, top=False, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=True, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
#Generate fifth panel
##############################################################################
colors = [colors_parent[1],colors_parent_dark[1],colors_parent[1],colors_parent_dark[1]];   # defines colors
sns.set_palette(sns.color_palette(colors))      # sets colors
ylabeloffset = -1

# extract data from dataframe to test if their distribution is gaussian
# data_1to1d = df_fs[df_fs["keys"]=="AR1to1d_fs"]["RSI_xx"].to_numpy()
# data_1to1s = df_fs[df_fs["keys"]=="AR1to1s_fs"]["RSI_xx"].to_numpy()
# if test_if_gaussian(data_1to1d,data_1to1s,'REI'):
#     test = 't-test_ind'
# else:
test = 'Mann-Whitney'

ymin = -0.2
ymax = 0.6
stat_annotation_offset = 0

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[0,2])

# a little weird way to get the dataframe format I need for seaborn boxplots...
df1 = pd.DataFrame(df_d_hs['RSI_xx_left'])
df2 = pd.DataFrame(df_d_hs['RSI_xx_right'])
df3 = pd.DataFrame(df_d_hs['RSI_yy_left'])
df4 = pd.DataFrame(df_d_hs['RSI_yy_right'])

df1 = df1.transpose().reset_index(drop=True).transpose()
df2 = df2.transpose().reset_index(drop=True).transpose()
df3 = df3.transpose().reset_index(drop=True).transpose()
df4 = df4.transpose().reset_index(drop=True).transpose()

df_plot = pd.concat([df1,df2,df3,df4],axis=0)

df_plot.rename( columns={0:'sigma'}, inplace=True )

keys_sx_l = ['sigma_xx_left' for i in range(n_d_halfstim)]
keys_sx_r = ['sigma_xx_right' for i in range(n_d_halfstim)]
keys_sy_l = ['sigma_yy_left' for i in range(n_d_halfstim)]
keys_sy_r = ['sigma_yy_right' for i in range(n_d_halfstim)]
keys = np.concatenate((keys_sx_l, keys_sx_r, keys_sy_l, keys_sy_r))
df_plot['keys'] = keys


# create box- and swarmplots
sns.swarmplot(x='keys', y='sigma', data=df_plot, ax=fig_ax,alpha=alpha_sw,linewidth=linewidth_sw, zorder=0, size=dotsize)
bp = sns.boxplot(x='keys', y='sigma', data=df_plot, ax=fig_ax,linewidth=linewidth_bp,notch=True, showfliers = False, width=width)

# order = ['AR1to1d_fs', 'AR1to1s_fs']
# add_stat_annotation(bp, data=df_fs, x=x, y=y, order=order, box_pairs=[('AR1to1d_fs', 'AR1to1s_fs')], 
#                     line_offset_to_box=stat_annotation_offset, test=test, text_format='star', loc='inside', verbose=2)

# make boxplots transparent
for patch in bp.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, alpha_bp))

plt.setp(bp.artists, edgecolor = 'k')
plt.setp(bp.lines, color='k')
     
# set labels
fig_ax.set_xticklabels(['1', '2', '3', '4'])
fig_ax.set_xlabel(xlabel=None)
fig_ax.set_ylabel(ylabel=None, labelpad=ylabeloffset)
# fig_ax.set_title(label='REI', pad=titleoffset)
fig_ax.set()

# Define where you want ticks
yticks = np.arange(-0.2,0.61,0.2)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=False, top=False, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=False, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
#Generate sixth panel
##############################################################################
colors = [colors_parent[2],colors_parent_dark[2],colors_parent[2],colors_parent_dark[2]];   # defines colors
sns.set_palette(sns.color_palette(colors))      # sets colors

# extract data from dataframe to test if their distribution is gaussian
# data_1to1d = df_hs[df_hs["keys"]=="AR1to1d_hs"]["REI"].to_numpy()
# data_1to1s = df_hs[df_hs["keys"]=="AR1to1s_hs"]["REI"].to_numpy()
# if test_if_gaussian(data_1to1d,data_1to1s,'REI'):
#     test = 't-test_ind'
# else:
test = 'Mann-Whitney'

ymin = -0.2
ymax = 0.6
stat_annotation_offset = 0.5

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[1,2])

# a little weird way to get the dataframe format I need for seaborn boxplots...
df1 = pd.DataFrame(df_s_hs['RSI_xx_left'])
df2 = pd.DataFrame(df_s_hs['RSI_xx_right'])
df3 = pd.DataFrame(df_s_hs['RSI_yy_left'])
df4 = pd.DataFrame(df_s_hs['RSI_yy_right'])

df1 = df1.transpose().reset_index(drop=True).transpose()
df2 = df2.transpose().reset_index(drop=True).transpose()
df3 = df3.transpose().reset_index(drop=True).transpose()
df4 = df4.transpose().reset_index(drop=True).transpose()

df_plot = pd.concat([df1,df2,df3,df4],axis=0)

df_plot.rename( columns={0:'sigma'}, inplace=True )

keys_sx_l = ['sigma_xx_left' for i in range(n_s_halfstim)]
keys_sx_r = ['sigma_xx_right' for i in range(n_s_halfstim)]
keys_sy_l = ['sigma_yy_left' for i in range(n_s_halfstim)]
keys_sy_r = ['sigma_yy_right' for i in range(n_s_halfstim)]
keys = np.concatenate((keys_sx_l, keys_sx_r, keys_sy_l, keys_sy_r))
df_plot['keys'] = keys

# create box- and swarmplots
sns.swarmplot(x='keys', y='sigma', data=df_plot, ax=fig_ax,alpha=alpha_sw,linewidth=linewidth_sw, zorder=0, size=dotsize)
bp = sns.boxplot(x='keys', y='sigma', data=df_plot, ax=fig_ax,linewidth=linewidth_bp,notch=True, showfliers = False, width=width)

# order = ['AR1to1d_hs', 'AR1to1s_hs']
# add_stat_annotation(bp, data=df_hs, x=x, y=y, order=order, box_pairs=[('AR1to1d_hs', 'AR1to1s_hs')], 
#                     line_offset_to_box=stat_annotation_offset, test=test, text_format='star', loc='inside', verbose=2)

# make boxplots transparent
for patch in bp.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, alpha_bp))

plt.setp(bp.artists, edgecolor = 'k')
plt.setp(bp.lines, color='k')
     
# set labels
fig_ax.set_xticklabels(['1', '2', '3', '4'])
fig_ax.set_xlabel(xlabel=None)
fig_ax.set_ylabel(ylabel=None, labelpad=ylabeloffset)
fig_ax.set_title(label=None, pad=titleoffset)
fig_ax.set()

# Define where you want ticks
yticks = np.arange(-0.2,0.61,0.2)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=False, top=False, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=False, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)


# write title for panels 1 to 4
plt.text(-5.7,2.575947,'Relative strain energy',fontsize=10)
# write title for panels 5 to 6
plt.text(-0.662416, 2.400220,'Relative energy \n     increase',fontsize=10)
# save plot to file
plt.savefig(figfolder+'G.png', dpi=300, bbox_inches="tight")

plt.show()
#%% plot figure 3E

# # define plot parameters
# fig = plt.figure(2, figsize=(5, 3))             # figuresize in inches
# gs = gridspec.GridSpec(2,3)                     # sets up subplotgrid rows by columns
# gs.update(wspace=0.6, hspace=0.7)              # adjusts space in between the boxes in the grid
# colors = [colors_parent[1],colors_parent[2]];   # defines colors
# sns.set_palette(sns.color_palette(colors))      # sets colors
# linewidth_bp = 0.7                              # linewidth of boxplot borders
# width = 0.3                                     # width of boxplots
# dotsize = 2                                     # size of datapoints in swarmplot
# linewidth_sw = 0.3                              # linewidth of boxplot borders
# alpha_sw = 1                                    # transparency of dots in swarmplot
# alpha_bp = 0.8                                  # transparency of boxplots
# ylabeloffset = 1                                # adjusts distance of ylabel to the plot
# xlabeloffset = 1                                # adjusts distance of ylabel to the plot
# titleoffset = 5                                 # adjusts distance of title to the plot

# ##############################################################################
# #Generate first panel
# ##############################################################################

# ymin = 0.9
# ymax = 1.3

# # the grid spec is rows, then columns
# fig_ax = fig.add_subplot(gs[0,0])

# # set plot variables
# x = np.arange(60)
# y = AR1to1d_fullstim_long["MSM_data"]["relsigma_yy_average"]
# x = x[::2] # downsample data for nicer plotting
# y = y[::2,:]
# y_mean = np.nanmean(y,axis=1)
# y_std = np.nanstd(y,axis=1)
# y_sem = y_std/np.sqrt(np.shape(y)[1])

# # create box- and swarmplots
# fig_ax.errorbar(x,y_mean,yerr=y_sem, mfc='w', color=colors_parent[1], marker='o',ms=2, linewidth=0.5, ls='none',markeredgewidth=0.5)
     
# # set labels
# fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
# fig_ax.set_ylabel(ylabel='doublet', labelpad=ylabeloffset)
# fig_ax.set_title(label='relative $\mathrm{\sigma_{yy}}$', pad=titleoffset)
# fig_ax.set()

# # Define where you want ticks
# xticks = np.arange(0,61,20)
# yticks = np.arange(0.9,1.31,0.1)

# plt.xticks(xticks)
# plt.yticks(yticks)

# # provide info on tick parameters
# plt.minorticks_on()
# plt.tick_params(direction='in',which='minor', length=3, bottom=True, top=False, left=True, right=True)
# plt.tick_params(direction='in',which='major', length=6, bottom=True, top=False, left=True, right=True)

# # set limits
# fig_ax.set_ylim(ymin=ymin)
# fig_ax.set_ylim(ymax=ymax)

# ##############################################################################
# #Generate second panel
# ##############################################################################

# ymin = 0.9
# ymax = 1.3

# # the grid spec is rows, then columns
# fig_ax = fig.add_subplot(gs[0,1])

# # set plot variables
# x = np.arange(60)
# y = AR1to1d_halfstim["MSM_data"]["relsigma_yy_average"]
# x = x[::2] # downsample data for nicer plotting
# y = y[::2,:]
# y_mean = np.nanmean(y,axis=1)
# y_std = np.nanstd(y,axis=1)
# y_sem = y_std/np.sqrt(np.shape(y)[1])

# # create box- and swarmplots
# fig_ax.errorbar(x,y_mean,yerr=y_sem, mfc='w', color=colors_parent[1], marker='o',ms=2, linewidth=0.5, ls='none',markeredgewidth=0.5)
     
# # set labels
# fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
# fig_ax.set_ylabel(ylabel='doublet', labelpad=ylabeloffset)
# fig_ax.set_title(label='relative $\mathrm{\sigma_{yy}}$', pad=titleoffset)
# fig_ax.set()

# # Define where you want ticks
# xticks = np.arange(0,61,20)
# yticks = np.arange(0.9,1.31,0.1)

# plt.xticks(xticks)
# plt.yticks(yticks)

# # provide info on tick parameters
# plt.minorticks_on()
# plt.tick_params(direction='in',which='minor', length=3, bottom=True, top=False, left=True, right=True)
# plt.tick_params(direction='in',which='major', length=6, bottom=True, top=False, left=True, right=True)

# # set limits
# fig_ax.set_ylim(ymin=ymin)
# fig_ax.set_ylim(ymax=ymax)

# ##############################################################################
# #Generate third panel
# ##############################################################################

# ymin = 0.9
# ymax = 1.3

# # the grid spec is rows, then columns
# fig_ax = fig.add_subplot(gs[1,0])

# # set plot variables
# x = np.arange(60)
# y = AR1to1s_fullstim_long["MSM_data"]["relsigma_yy_average"]
# x = x[::2] # downsample data for nicer plotting
# y = y[::2,:]
# y_mean = np.nanmean(y,axis=1)
# y_std = np.nanstd(y,axis=1)
# y_sem = y_std/np.sqrt(np.shape(y)[1])

# # create box- and swarmplots
# fig_ax.errorbar(x,y_mean,yerr=y_sem, mfc='w', color=colors_parent[2], marker='o',ms=2, linewidth=0.5, ls='none',markeredgewidth=0.5)
     
# # set labels
# fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
# fig_ax.set_ylabel(ylabel='singlet', labelpad=ylabeloffset)
# fig_ax.set_title(label='relative $\mathrm{\sigma_{yy}}$', pad=titleoffset)
# fig_ax.set()

# # Define where you want ticks
# xticks = np.arange(0,61,20)
# yticks = np.arange(0.9,1.31,0.1)

# plt.xticks(xticks)
# plt.yticks(yticks)

# # provide info on tick parameters
# plt.minorticks_on()
# plt.tick_params(direction='in',which='minor', length=3, bottom=True, top=False, left=True, right=True)
# plt.tick_params(direction='in',which='major', length=6, bottom=True, top=False, left=True, right=True)

# # set limits
# fig_ax.set_ylim(ymin=ymin)
# fig_ax.set_ylim(ymax=ymax)

# ##############################################################################
# #Generate fourth panel
# ##############################################################################

# ymin = 0.9
# ymax = 1.3

# # the grid spec is rows, then columns
# fig_ax = fig.add_subplot(gs[1,1])

# # set plot variables
# x = np.arange(60)
# y = AR1to1s_halfstim["MSM_data"]["relsigma_yy_average"]
# x = x[::2] # downsample data for nicer plotting
# y = y[::2,:]
# y_mean = np.nanmean(y,axis=1)
# y_std = np.nanstd(y,axis=1)
# y_sem = y_std/np.sqrt(np.shape(y)[1])

# # create box- and swarmplots
# fig_ax.errorbar(x,y_mean,yerr=y_sem, mfc='w', color=colors_parent[2], marker='o',ms=2, linewidth=0.5, ls='none',markeredgewidth=0.5)
     
# # set labels
# fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
# fig_ax.set_ylabel(ylabel='singlet', labelpad=ylabeloffset)
# fig_ax.set_title(label='relative $\mathrm{\sigma_{yy}}$', pad=titleoffset)
# fig_ax.set()

# # Define where you want ticks
# xticks = np.arange(0,61,20)
# yticks = np.arange(0.9,1.31,0.1)

# plt.xticks(xticks)
# plt.yticks(yticks)

# # provide info on tick parameters
# plt.minorticks_on()
# plt.tick_params(direction='in',which='minor', length=3, bottom=True, top=False, left=True, right=True)
# plt.tick_params(direction='in',which='major', length=6, bottom=True, top=False, left=True, right=True)

# # set limits
# fig_ax.set_ylim(ymin=ymin)
# fig_ax.set_ylim(ymax=ymax)




# #%% plot figure 3D

# # define plot parameters
# fig = plt.figure(2, figsize=(5, 3))             # figuresize in inches
# gs = gridspec.GridSpec(2,3)                     # sets up subplotgrid rows by columns
# gs.update(wspace=0.6, hspace=0.7)              # adjusts space in between the boxes in the grid
# colors = [colors_parent[1],colors_parent[2]];   # defines colors
# sns.set_palette(sns.color_palette(colors))      # sets colors
# linewidth_bp = 0.7                              # linewidth of boxplot borders
# width = 0.3                                     # width of boxplots
# dotsize = 2                                     # size of datapoints in swarmplot
# linewidth_sw = 0.3                              # linewidth of boxplot borders
# alpha_sw = 1                                    # transparency of dots in swarmplot
# alpha_bp = 0.8                                  # transparency of boxplots
# ylabeloffset = 1                                # adjusts distance of ylabel to the plot
# xlabeloffset = 1                                # adjusts distance of ylabel to the plot
# titleoffset = 5                                 # adjusts distance of title to the plot

# ##############################################################################
# #Generate first panel
# ##############################################################################

# ymin = 0.9
# ymax = 1.3

# # the grid spec is rows, then columns
# fig_ax = fig.add_subplot(gs[0,0])

# # set plot variables
# x = np.arange(60)
# y = AR1to1d_fullstim_long["MSM_data"]["relsigma_xx_average"]
# x = x[::2] # downsample data for nicer plotting
# y = y[::2,:]
# y_mean = np.nanmean(y,axis=1)
# y_std = np.nanstd(y,axis=1)
# y_sem = y_std/np.sqrt(np.shape(y)[1])

# # create box- and swarmplots
# fig_ax.errorbar(x,y_mean,yerr=y_sem, mfc='w', color=colors_parent[1], marker='o',ms=2, linewidth=0.5, ls='none',markeredgewidth=0.5)
     
# # set labels
# fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
# fig_ax.set_ylabel(ylabel='doublet', labelpad=ylabeloffset)
# fig_ax.set_title(label='relative $\mathrm{\sigma_{xx}}$', pad=titleoffset)
# fig_ax.set()

# # Define where you want ticks
# xticks = np.arange(0,61,20)
# yticks = np.arange(0.9,1.31,0.1)

# plt.xticks(xticks)
# plt.yticks(yticks)

# # provide info on tick parameters
# plt.minorticks_on()
# plt.tick_params(direction='in',which='minor', length=3, bottom=True, top=False, left=True, right=True)
# plt.tick_params(direction='in',which='major', length=6, bottom=True, top=False, left=True, right=True)

# # set limits
# fig_ax.set_ylim(ymin=ymin)
# fig_ax.set_ylim(ymax=ymax)

# ##############################################################################
# #Generate second panel
# ##############################################################################

# ymin = 0.9
# ymax = 1.3

# # the grid spec is rows, then columns
# fig_ax = fig.add_subplot(gs[0,1])

# # set plot variables
# x = np.arange(60)
# y = AR1to1d_halfstim["MSM_data"]["relsigma_xx_average"]
# x = x[::2] # downsample data for nicer plotting
# y = y[::2,:]
# y_mean = np.nanmean(y,axis=1)
# y_std = np.nanstd(y,axis=1)
# y_sem = y_std/np.sqrt(np.shape(y)[1])

# # create box- and swarmplots
# fig_ax.errorbar(x,y_mean,yerr=y_sem, mfc='w', color=colors_parent[1], marker='o',ms=2, linewidth=0.5, ls='none',markeredgewidth=0.5)
     
# # set labels
# fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
# fig_ax.set_ylabel(ylabel='doublet', labelpad=ylabeloffset)
# fig_ax.set_title(label='relative $\mathrm{\sigma_{xx}}$', pad=titleoffset)
# fig_ax.set()

# # Define where you want ticks
# xticks = np.arange(0,61,20)
# yticks = np.arange(0.9,1.31,0.1)

# plt.xticks(xticks)
# plt.yticks(yticks)

# # provide info on tick parameters
# plt.minorticks_on()
# plt.tick_params(direction='in',which='minor', length=3, bottom=True, top=False, left=True, right=True)
# plt.tick_params(direction='in',which='major', length=6, bottom=True, top=False, left=True, right=True)

# # set limits
# fig_ax.set_ylim(ymin=ymin)
# fig_ax.set_ylim(ymax=ymax)

# ##############################################################################
# #Generate third panel
# ##############################################################################

# ymin = 0.9
# ymax = 1.3

# # the grid spec is rows, then columns
# fig_ax = fig.add_subplot(gs[1,0])

# # set plot variables
# x = np.arange(60)
# y = AR1to1s_fullstim_long["MSM_data"]["relsigma_xx_average"]
# x = x[::2] # downsample data for nicer plotting
# y = y[::2,:]
# y_mean = np.nanmean(y,axis=1)
# y_std = np.nanstd(y,axis=1)
# y_sem = y_std/np.sqrt(np.shape(y)[1])

# # create box- and swarmplots
# fig_ax.errorbar(x,y_mean,yerr=y_sem, mfc='w', color=colors_parent[2], marker='o',ms=2, linewidth=0.5, ls='none',markeredgewidth=0.5)
     
# # set labels
# fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
# fig_ax.set_ylabel(ylabel='singlet', labelpad=ylabeloffset)
# fig_ax.set_title(label='relative $\mathrm{\sigma_{xx}}$', pad=titleoffset)
# fig_ax.set()

# # Define where you want ticks
# xticks = np.arange(0,61,20)
# yticks = np.arange(0.9,1.31,0.1)

# plt.xticks(xticks)
# plt.yticks(yticks)

# # provide info on tick parameters
# plt.minorticks_on()
# plt.tick_params(direction='in',which='minor', length=3, bottom=True, top=False, left=True, right=True)
# plt.tick_params(direction='in',which='major', length=6, bottom=True, top=False, left=True, right=True)

# # set limits
# fig_ax.set_ylim(ymin=ymin)
# fig_ax.set_ylim(ymax=ymax)

# ##############################################################################
# #Generate fourth panel
# ##############################################################################

# ymin = 0.9
# ymax = 1.3

# # the grid spec is rows, then columns
# fig_ax = fig.add_subplot(gs[1,1])

# # set plot variables
# x = np.arange(60)
# y = AR1to1s_halfstim["MSM_data"]["relsigma_xx_average"]
# x = x[::2] # downsample data for nicer plotting
# y = y[::2,:]
# y_mean = np.nanmean(y,axis=1)
# y_std = np.nanstd(y,axis=1)
# y_sem = y_std/np.sqrt(np.shape(y)[1])

# # create box- and swarmplots
# fig_ax.errorbar(x,y_mean,yerr=y_sem, mfc='w', color=colors_parent[2], marker='o',ms=2, linewidth=0.5, ls='none',markeredgewidth=0.5)
     
# # set labels
# fig_ax.set_xlabel(xlabel='time [min]', labelpad=xlabeloffset)
# fig_ax.set_ylabel(ylabel='singlet', labelpad=ylabeloffset)
# fig_ax.set_title(label='relative $\mathrm{\sigma_{xx}}$', pad=titleoffset)
# fig_ax.set()

# # Define where you want ticks
# xticks = np.arange(0,61,20)
# yticks = np.arange(0.9,1.31,0.1)

# plt.xticks(xticks)
# plt.yticks(yticks)

# # provide info on tick parameters
# plt.minorticks_on()
# plt.tick_params(direction='in',which='minor', length=3, bottom=True, top=False, left=True, right=True)
# plt.tick_params(direction='in',which='major', length=6, bottom=True, top=False, left=True, right=True)

# # set limits
# fig_ax.set_ylim(ymin=ymin)
# fig_ax.set_ylim(ymax=ymax)

# # save plot to file
# plt.savefig(figfolder+'D.png', dpi=300, bbox_inches="tight")
# plt.close()

# # ##############################################################################
# # #Generate fifth panel
# # ##############################################################################
# # ylabeloffset = -1

# # # extract data from dataframe to test if their distribution is gaussian
# data_1to1d = df_fs[df_fs["keys"]=="AR1to1d_fs"]["REI"].to_numpy()
# data_1to1s = df_fs[df_fs["keys"]=="AR1to1s_fs"]["REI"].to_numpy()
# # if test_if_gaussian(data_1to1d,data_1to1s,'REI'):
# #     test = 't-test_ind'
# # else:
# test = 'Mann-Whitney'

# ymin = -0.2
# ymax = 0.8
# stat_annotation_offset = 0

# # the grid spec is rows, then columns
# fig_ax = fig.add_subplot(gs[0,2])

# # set plot variables
# x = 'keys'
# y = 'REI'

# # create box- and swarmplots
# sns.swarmplot(x=x, y=y, data=df_fs, ax=fig_ax,alpha=alpha_sw,linewidth=linewidth_sw, zorder=0, size=dotsize)
# bp = sns.boxplot(x=x, y=y, data=df_fs, ax=fig_ax,linewidth=linewidth_bp,notch=True, showfliers = False, width=width)

# order = ['AR1to1d_fs', 'AR1to1s_fs']
# add_stat_annotation(bp, data=df_fs, x=x, y=y, order=order, box_pairs=[('AR1to1d_fs', 'AR1to1s_fs')], 
#                     line_offset_to_box=stat_annotation_offset, test=test, text_format='star', loc='inside', verbose=2)

# # make boxplots transparent
# for patch in bp.artists:
#     r, g, b, a = patch.get_facecolor()
#     patch.set_facecolor((r, g, b, alpha_bp))

# plt.setp(bp.artists, edgecolor = 'k')
# plt.setp(bp.lines, color='k')
     
# # set labels
# fig_ax.set_xticklabels(['doublet', 'singlet'])
# fig_ax.set_xlabel(xlabel=None)
# fig_ax.set_ylabel(ylabel='REI', labelpad=ylabeloffset)
# fig_ax.set_title(label='Relative energy increase', pad=titleoffset)
# fig_ax.set()

# # Define where you want ticks
# yticks = np.arange(-0.2,0.81,0.2)
# plt.yticks(yticks)

# # provide info on tick parameters
# plt.minorticks_on()
# plt.tick_params(direction='in',which='minor', length=3, bottom=False, top=False, left=True, right=True)
# plt.tick_params(direction='in',which='major', length=6, bottom=False, top=False, left=True, right=True)

# # set limits
# fig_ax.set_ylim(ymin=ymin)
# fig_ax.set_ylim(ymax=ymax)

# ##############################################################################
# #Generate sixth panel
# ##############################################################################
# # extract data from dataframe to test if their distribution is gaussian
# data_1to1d = df_hs[df_hs["keys"]=="AR1to1d_hs"]["REI"].to_numpy()
# data_1to1s = df_hs[df_hs["keys"]=="AR1to1s_hs"]["REI"].to_numpy()
# # if test_if_gaussian(data_1to1d,data_1to1s,'REI'):
# #     test = 't-test_ind'
# # else:
# test = 'Mann-Whitney'

# ymin = -0.2
# ymax = 0.8
# stat_annotation_offset = 0.7

# # the grid spec is rows, then columns
# fig_ax = fig.add_subplot(gs[1,2])

# # set plot variables
# x = 'keys'
# y = 'REI'

# # create box- and swarmplots
# sns.swarmplot(x=x, y=y, data=df_hs, ax=fig_ax,alpha=alpha_sw,linewidth=linewidth_sw, zorder=0, size=dotsize)
# bp = sns.boxplot(x=x, y=y, data=df_hs, ax=fig_ax,linewidth=linewidth_bp,notch=True, showfliers = False, width=width)

# order = ['AR1to1d_hs', 'AR1to1s_hs']
# add_stat_annotation(bp, data=df_hs, x=x, y=y, order=order, box_pairs=[('AR1to1d_hs', 'AR1to1s_hs')], 
#                     line_offset_to_box=stat_annotation_offset, test=test, text_format='star', loc='inside', verbose=2)

# # make boxplots transparent
# for patch in bp.artists:
#     r, g, b, a = patch.get_facecolor()
#     patch.set_facecolor((r, g, b, alpha_bp))

# plt.setp(bp.artists, edgecolor = 'k')
# plt.setp(bp.lines, color='k')
     
# # set labels
# fig_ax.set_xticklabels(['doublet', 'singlet'])
# fig_ax.set_xlabel(xlabel=None)
# fig_ax.set_ylabel(ylabel='REI', labelpad=ylabeloffset)
# fig_ax.set_title(label='Relative energy increase', pad=titleoffset)
# fig_ax.set()

# # Define where you want ticks
# yticks = np.arange(-0.2,0.81,0.2)
# plt.yticks(yticks)

# # provide info on tick parameters
# plt.minorticks_on()
# plt.tick_params(direction='in',which='minor', length=3, bottom=False, top=False, left=True, right=True)
# plt.tick_params(direction='in',which='major', length=6, bottom=False, top=False, left=True, right=True)

# # set limits
# fig_ax.set_ylim(ymin=ymin)
# fig_ax.set_ylim(ymax=ymax)

# save plot to file
# plt.savefig(figfolder+'E.png', dpi=300, dpi=300, bbox_inches="tight")
# plt.close()

# #%% Plot figure 3A
# x = np.arange(0,60)
# y1 = AR1to1d_fullstim_long["TFM_data"]["relEs"]
# y2 = AR1to1s_fullstim_long["TFM_data"]["relEs"]
# ylim = (0.9,1.2)
# xlabel = "time [min]"
# ylabel = "Es_{rel}"
# titleleft = "doublet"
# titleright = "singlet"
# titleboth = "Relative strain energy full stim"
# plot_twovalues_over_time(x, y1, y2, ylim, xlabel, ylabel,titleleft, titleright, titleboth)

# #%% Plot figure 3B
# x = np.arange(0,60)
# y1 = AR1to1d_halfstim["TFM_data"]["relEs"]
# y2 = AR1to1s_halfstim["TFM_data"]["relEs"]
# ylim = (0.9,1.2)
# xlabel = "time [min]"
# ylabel = "Es_{rel}"
# titleleft = "doublet"
# titleright = "singlet"
# titleboth = "Relative strain energy half stim"
# plot_twovalues_over_time(x, y1, y2, ylim, xlabel, ylabel,titleleft, titleright, titleboth)

# #%% Plot figure 3C
# x = np.arange(0,60)
# y1 = AR1to1d_fullstim_long["MSM_data"]["relsigma_xx_average"]
# y2 = AR1to1s_fullstim_long["MSM_data"]["relsigma_xx_average"]
# ylim = (0.9,1.2)
# xlabel = "time [min]"
# ylabel = "sigma_{xx, rel}"
# titleleft = "doublet"
# titleright = "singlet"
# titleboth = "Relative xx-stress full stim"
# plot_twovalues_over_time(x, y1, y2, ylim, xlabel, ylabel,titleleft, titleright, titleboth)

# #%% Plot figure 3D
# x = np.arange(0,60)
# y1 = AR1to1d_halfstim["MSM_data"]["relsigma_xx_average"]
# y2 = AR1to1s_halfstim["MSM_data"]["relsigma_xx_average"]
# ylim = (0.9,1.2)
# xlabel = "time [min]"
# ylabel = "sigma_{xx, rel}"
# titleleft = "doublet"
# titleright = "singlet"
# titleboth = "Relative xx-stress half stim"
# fig=plot_twovalues_over_time(x, y1, y2, ylim, xlabel, ylabel,titleleft, titleright, titleboth)
# fig.savefig(folder+'a.png', dpi=300, dpi=300, bbox_inches="tight")
# plt.close()

# #%% Plot figure 3F
# x = np.arange(0,60)
# y1 = AR1to1d_halfstim["MSM_data"]["relsigma_yy_average"]
# y2 = AR1to1s_halfstim["MSM_data"]["relsigma_yy_average"]
# ylim = (0.9,1.2)
# xlabel = "time [min]"
# ylabel = "sigma_{yy, rel}"
# titleleft = "doublet"
# titleright = "singlet"
# titleboth = "Relative yy-stress half stim"
# fig=plot_twovalues_over_time(x, y1, y2, ylim, xlabel, ylabel,titleleft, titleright, titleboth)
# fig.savefig(folder+'b.png', dpi=300, dpi=300, bbox_inches="tight")
# plt.close()
# #%% Plot figure 3G
# x = np.arange(0,60)
# y1 = AR1to1d_halfstim["MSM_data"]["relsigma_yy_lefthalf_average"]
# y2 = AR1to1d_halfstim["MSM_data"]["relsigma_yy_righthalf_average"]
# ylim = (0.9,1.2)
# xlabel = "time [min]"
# ylabel = "sigma_{yy, rel}"
# titleleft = "left"
# titleright = "right"
# titleboth = "Relative yy-stress half stim doublet"
# fig=plot_twovalues_over_time(x, y1, y2, ylim, xlabel, ylabel,titleleft, titleright, titleboth)
# fig.savefig(folder+'c.png', dpi=300, dpi=300, bbox_inches="tight")
# plt.close()
# #%% Plot figure 3H
# x = np.arange(0,60)
# y1 = AR1to1d_halfstim["MSM_data"]["relsigma_xx_lefthalf_average"]
# y2 = AR1to1d_halfstim["MSM_data"]["relsigma_xx_righthalf_average"]
# ylim = (0.9,1.2)
# xlabel = "time [min]"
# ylabel = "sigma_{xx, rel}"
# titleleft = "left"
# titleright = "right"
# titleboth = "Relative xx-stress half stim doublet"
# fig=plot_twovalues_over_time(x, y1, y2, ylim, xlabel, ylabel,titleleft, titleright, titleboth)
# fig.savefig(folder+'d.png', dpi=300, dpi=300, bbox_inches="tight")
# plt.close()
# #%% Plot figure 3I
# x = np.arange(0,60)
# y1 = AR1to1s_halfstim["MSM_data"]["relsigma_yy_lefthalf_average"]
# y2 = AR1to1s_halfstim["MSM_data"]["relsigma_yy_righthalf_average"]
# ylim = (0.9,1.2)
# xlabel = "time [min]"
# ylabel = "sigma_{yy, rel}"
# titleleft = "left"
# titleright = "right"
# titleboth = "Relative yy-stress half stim doublet"
# fig=plot_twovalues_over_time(x, y1, y2, ylim, xlabel, ylabel,titleleft, titleright, titleboth)
# fig.savefig(folder+'e.png', dpi=300, dpi=300, bbox_inches="tight")
# plt.close()
# #%% Plot figure 3J
# x = np.arange(0,60)
# y1 = AR1to1s_halfstim["MSM_data"]["relsigma_xx_lefthalf_average"]
# y2 = AR1to1s_halfstim["MSM_data"]["relsigma_xx_righthalf_average"]
# ylim = (0.9,1.2)
# xlabel = "time [min]"
# ylabel = "sigma_{xx, rel}"
# titleleft = "left"
# titleright = "right"
# titleboth = "Relative xx-stress half stim doublet"
# fig=plot_twovalues_over_time(x, y1, y2, ylim, xlabel, ylabel,titleleft, titleright, titleboth)
# fig.savefig(folder+'f.png', dpi=300, dpi=300, bbox_inches="tight")
# plt.close()
# #%% Plot figure 3L
# x = np.arange(0,60)
# y1 = AR1to1d_fullstim_long["MSM_data"]["relAIC"]
# y2 = AR1to1s_fullstim_long["MSM_data"]["relAIC"]
# ylim = (-0.1,0.1)
# xlabel = "time [min]"
# ylabel = "relative AIC"
# titleleft = "doublet"
# titleright = "singlet"
# titleboth = "Anisotropy Coefficient fullstim"
# plot_twovalues_over_time(x, y1, y2, ylim, xlabel, ylabel,titleleft, titleright, titleboth)
# #%% Plot figure 3L
# x = np.arange(0,60)
# y1 = AR1to1d_halfstim["MSM_data"]["relAIC"]
# y2 = AR1to1s_halfstim["MSM_data"]["relAIC"]
# ylim = (-0.1,0.1)
# xlabel = "time [min]"
# ylabel = "relative AIC"
# titleleft = "doublet"
# titleright = "singlet"
# titleboth = "Anisotropy Coefficient halfstim"
# plot_twovalues_over_time(x, y1, y2, ylim, xlabel, ylabel,titleleft, titleright, titleboth)
# #%% Plot figure 3L
# x = np.arange(0,60)
# y1 = AR1to1d_halfstim["MSM_data"]["AIC_left"]
# y2 = AR1to1d_halfstim["MSM_data"]["AIC_right"]
# ylim = (-0.2,0.2)
# xlabel = "time [min]"
# ylabel = "relative AIC"
# titleleft = "left"
# titleright = "right"
# titleboth = "Anisotropy Coefficient doublet"
# plot_twovalues_over_time(x, y1, y2, ylim, xlabel, ylabel,titleleft, titleright, titleboth)
# #%% Plot figure 3M
# x = np.arange(0,60)
# y1 = AR1to1s_halfstim["MSM_data"]["AIC_left"]
# y2 = AR1to1s_halfstim["MSM_data"]["AIC_right"]
# ylim = (0.3,0.5)
# xlabel = "time [min]"
# ylabel = "relative AIC"
# titleleft = "left"
# titleright = "right"
# titleboth = "Anisotropy Coefficient singlet"
# plot_twovalues_over_time(x, y1, y2, ylim, xlabel, ylabel,titleleft, titleright, titleboth)