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

# mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 8


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

# define some colors for the plots
colors_parent = ['#026473','#E3CC69','#77C8A6','#D96248'];

figfolder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/_Figure3/"
if not os.path.exists(figfolder):
        os.mkdir(figfolder)
#%% prepare dataframe for boxplots

# initialize empty dictionaries
concatenated_data_fs = {}
concatenated_data_hs = {}

# loop over all keys
for key1 in AR1to1d_fullstim_long: # keys are the same for all dictionaries so I'm just taking one example here
    for key2 in AR1to1d_fullstim_long[key1]:
        if AR1to1d_fullstim_long[key1][key2].ndim == 1: # only 1D data can be stored in the data frame
        
            # concatenate values from different experiments
            concatenated_data_fs[key2] = np.concatenate((AR1to1d_fullstim_long[key1][key2], AR1to1s_fullstim_long[key1][key2]))
            concatenated_data_hs[key2] = np.concatenate((AR1to1d_halfstim[key1][key2], AR1to1s_halfstim[key1][key2]))

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

# add keys to dictionary with concatenated data
concatenated_data_fs['keys'] = keys_fs
concatenated_data_hs['keys'] = keys_hs

# Creates DataFrame
df_fs = pd.DataFrame(concatenated_data_fs)
df_hs = pd.DataFrame(concatenated_data_hs)

#%% plot figure 3B, stress map difference fullstim

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

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(3.25, 2.5))

im = axes[0,0].imshow(sigmaxx_1to1d_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent, vmin=sigma_min, vmax=sigma_max, aspect='auto')

axes[0,1].imshow(sigmayy_1to1d_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent, vmin=sigma_min, vmax=sigma_max, aspect='auto')

axes[1,0].imshow(sigmaxx_1to1s_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent, vmin=sigma_min, vmax=sigma_max, aspect='auto')

axes[1,1].imshow(sigmayy_1to1s_diff_crop, cmap=plt.get_cmap("seismic"), interpolation="bilinear", extent=extent, vmin=sigma_min, vmax=sigma_max, aspect='auto')

# adjust space in between plots
plt.subplots_adjust(wspace=0.02, hspace=-0.02)

# # add annotations
plt.text(-50,120,'sigmaxx',color = 'k')
plt.text(20,120,'sigmayy',color = 'k')
# plt.text(-40.5,55.5,'n=1',color = 'white')
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
plt.suptitle('MSM fullstim',y=0.94, x=0.44)

# save figure
fig.savefig(figfolder+'B1.png', dpi=300, bbox_inches="tight")
plt.close()

#%% plot figure 3B, stress map difference halfstim

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

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(3, 2.5))

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
plt.suptitle('MSM halfstim',y=0.94, x=0.44)

# save figure
fig.savefig(figfolder+'B2.png', dpi=300, bbox_inches="tight")
plt.close()

#%% plot figure 3C

# define plot parameters
fig = plt.figure(2, figsize=(5, 3))             # figuresize in inches
gs = gridspec.GridSpec(2,3)                     # sets up subplotgrid rows by columns
gs.update(wspace=0.6, hspace=0.7)              # adjusts space in between the boxes in the grid
colors = [colors_parent[1],colors_parent[2]];   # defines colors
sns.set_palette(sns.color_palette(colors))      # sets colors
linewidth_bp = 0.7                              # linewidth of boxplot borders
width = 0.3                                     # width of boxplots
dotsize = 2                                     # size of datapoints in swarmplot
linewidth_sw = 0.3                              # linewidth of boxplot borders
alpha_sw = 1                                    # transparency of dots in swarmplot
alpha_bp = 0.8                                  # transparency of boxplots
ylabeloffset = 1                                # adjusts distance of ylabel to the plot
xlabeloffset = 1                                # adjusts distance of ylabel to the plot
titleoffset = 5                                 # adjusts distance of title to the plot

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
fig_ax.set_title(label='relative $\mathrm{E_s}$', pad=titleoffset)
fig_ax.set()

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
fig_ax.set_ylabel(ylabel='doublet', labelpad=ylabeloffset)
fig_ax.set_title(label='relative $\mathrm{E_s}$', pad=titleoffset)
fig_ax.set()

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
fig_ax.set_title(label='relative $\mathrm{E_s}$', pad=titleoffset)
fig_ax.set()

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
fig_ax.set_ylabel(ylabel='singlet', labelpad=ylabeloffset)
fig_ax.set_title(label='relative $\mathrm{E_s}$', pad=titleoffset)
fig_ax.set()

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
ylabeloffset = -1

# extract data from dataframe to test if their distribution is gaussian
data_1to1d = df_fs[df_fs["keys"]=="AR1to1d_fs"]["REI"].to_numpy()
data_1to1s = df_fs[df_fs["keys"]=="AR1to1s_fs"]["REI"].to_numpy()
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
sns.swarmplot(x=x, y=y, data=df_fs, ax=fig_ax,alpha=alpha_sw,linewidth=linewidth_sw, zorder=0, size=dotsize)
bp = sns.boxplot(x=x, y=y, data=df_fs, ax=fig_ax,linewidth=linewidth_bp,notch=True, showfliers = False, width=width)

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
fig_ax.set_xticklabels(['doublet', 'singlet'])
fig_ax.set_xlabel(xlabel=None)
fig_ax.set_ylabel(ylabel='REI', labelpad=ylabeloffset)
fig_ax.set_title(label='Relative energy increase', pad=titleoffset)
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
# extract data from dataframe to test if their distribution is gaussian
data_1to1d = df_hs[df_hs["keys"]=="AR1to1d_hs"]["REI"].to_numpy()
data_1to1s = df_hs[df_hs["keys"]=="AR1to1s_hs"]["REI"].to_numpy()
# if test_if_gaussian(data_1to1d,data_1to1s,'REI'):
#     test = 't-test_ind'
# else:
test = 'Mann-Whitney'

ymin = -0.2
ymax = 0.8
stat_annotation_offset = 0.7

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[1,2])

# set plot variables
x = 'keys'
y = 'REI'

# create box- and swarmplots
sns.swarmplot(x=x, y=y, data=df_hs, ax=fig_ax,alpha=alpha_sw,linewidth=linewidth_sw, zorder=0, size=dotsize)
bp = sns.boxplot(x=x, y=y, data=df_hs, ax=fig_ax,linewidth=linewidth_bp,notch=True, showfliers = False, width=width)

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
fig_ax.set_xticklabels(['doublet', 'singlet'])
fig_ax.set_xlabel(xlabel=None)
fig_ax.set_ylabel(ylabel='REI', labelpad=ylabeloffset)
fig_ax.set_title(label='Relative energy increase', pad=titleoffset)
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

# save plot to file
plt.savefig(figfolder+'C.png', dpi=300, bbox_inches="tight")
plt.close()

#%% plot figure 3D

# define plot parameters
fig = plt.figure(2, figsize=(5, 3))             # figuresize in inches
gs = gridspec.GridSpec(2,3)                     # sets up subplotgrid rows by columns
gs.update(wspace=0.6, hspace=0.7)              # adjusts space in between the boxes in the grid
colors = [colors_parent[1],colors_parent[2]];   # defines colors
sns.set_palette(sns.color_palette(colors))      # sets colors
linewidth_bp = 0.7                              # linewidth of boxplot borders
width = 0.3                                     # width of boxplots
dotsize = 2                                     # size of datapoints in swarmplot
linewidth_sw = 0.3                              # linewidth of boxplot borders
alpha_sw = 1                                    # transparency of dots in swarmplot
alpha_bp = 0.8                                  # transparency of boxplots
ylabeloffset = 1                                # adjusts distance of ylabel to the plot
xlabeloffset = 1                                # adjusts distance of ylabel to the plot
titleoffset = 5                                 # adjusts distance of title to the plot

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
fig_ax.set_title(label='relative $\mathrm{\sigma_{xx}}$', pad=titleoffset)
fig_ax.set()

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
y = AR1to1d_halfstim["MSM_data"]["relsigma_xx_average"]
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
fig_ax.set_title(label='relative $\mathrm{\sigma_{xx}}$', pad=titleoffset)
fig_ax.set()

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
fig_ax.set_title(label='relative $\mathrm{\sigma_{xx}}$', pad=titleoffset)
fig_ax.set()

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
y = AR1to1s_halfstim["MSM_data"]["relsigma_xx_average"]
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
fig_ax.set_title(label='relative $\mathrm{\sigma_{xx}}$', pad=titleoffset)
fig_ax.set()

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

# save plot to file
plt.savefig(figfolder+'D.png', dpi=300, bbox_inches="tight")
plt.close()
#%% plot figure 3E

# define plot parameters
fig = plt.figure(2, figsize=(5, 3))             # figuresize in inches
gs = gridspec.GridSpec(2,3)                     # sets up subplotgrid rows by columns
gs.update(wspace=0.6, hspace=0.7)              # adjusts space in between the boxes in the grid
colors = [colors_parent[1],colors_parent[2]];   # defines colors
sns.set_palette(sns.color_palette(colors))      # sets colors
linewidth_bp = 0.7                              # linewidth of boxplot borders
width = 0.3                                     # width of boxplots
dotsize = 2                                     # size of datapoints in swarmplot
linewidth_sw = 0.3                              # linewidth of boxplot borders
alpha_sw = 1                                    # transparency of dots in swarmplot
alpha_bp = 0.8                                  # transparency of boxplots
ylabeloffset = 1                                # adjusts distance of ylabel to the plot
xlabeloffset = 1                                # adjusts distance of ylabel to the plot
titleoffset = 5                                 # adjusts distance of title to the plot

##############################################################################
#Generate first panel
##############################################################################

ymin = 0.9
ymax = 1.3

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[0,0])

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
fig_ax.set_ylabel(ylabel='doublet', labelpad=ylabeloffset)
fig_ax.set_title(label='relative $\mathrm{\sigma_{yy}}$', pad=titleoffset)
fig_ax.set()

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
y = AR1to1d_halfstim["MSM_data"]["relsigma_yy_average"]
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
fig_ax.set_title(label='relative $\mathrm{\sigma_{yy}}$', pad=titleoffset)
fig_ax.set()

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
fig_ax.set_ylabel(ylabel='singlet', labelpad=ylabeloffset)
fig_ax.set_title(label='relative $\mathrm{\sigma_{yy}}$', pad=titleoffset)
fig_ax.set()

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
y = AR1to1s_halfstim["MSM_data"]["relsigma_yy_average"]
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
fig_ax.set_title(label='relative $\mathrm{\sigma_{yy}}$', pad=titleoffset)
fig_ax.set()

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
# ##############################################################################
# #Generate fifth panel
# ##############################################################################
# ylabeloffset = -1

# # extract data from dataframe to test if their distribution is gaussian
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
plt.savefig(figfolder+'E.png', dpi=300, bbox_inches="tight")
plt.close()

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
# fig.savefig(folder+'a.png', dpi=300, bbox_inches="tight")
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
# fig.savefig(folder+'b.png', dpi=300, bbox_inches="tight")
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
# fig.savefig(folder+'c.png', dpi=300, bbox_inches="tight")
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
# fig.savefig(folder+'d.png', dpi=300, bbox_inches="tight")
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
# fig.savefig(folder+'e.png', dpi=300, bbox_inches="tight")
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
# fig.savefig(folder+'f.png', dpi=300, bbox_inches="tight")
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