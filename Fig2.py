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
from scipy.stats import normaltest, shapiro, pearsonr
from statannot import add_stat_annotation

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
AR1to1d_fullstim_short =  pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_short.dat", "rb"))
AR1to1s_fullstim_short =  pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_short.dat", "rb"))
# AR1to2d_halfstim =        pickle.load(open(folder + "analysed_data/AR1to2d_halfstim.dat", "rb"))
AR1to1d_halfstim =        pickle.load(open(folder + "analysed_data/AR1to1d_halfstim.dat", "rb"))
AR1to1s_halfstim =        pickle.load(open(folder + "analysed_data/AR1to1s_halfstim.dat", "rb"))
# AR2to1d_halfstim =        pickle.load(open(folder + "analysed_data/AR2to1d_halfstim.dat", "rb"))

AR1to1d_fullstim_long_CM =   pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_long_CM_data.dat", "rb"))
AR1to1s_fullstim_long_CM =   pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_long_CM_data.dat", "rb"))
AR1to1d_fullstim_short_CM =  pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_short_CM_data.dat", "rb"))
AR1to1s_fullstim_short_CM =  pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_short_CM_data.dat", "rb"))
# AR1to2d_halfstim_CM =        pickle.load(open(folder + "analysed_data/AR1to2d_halfstim_CM_data.dat", "rb"))
AR1to1d_halfstim_CM =        pickle.load(open(folder + "analysed_data/AR1to1d_halfstim_CM_data.dat", "rb"))
AR1to1s_halfstim_CM =        pickle.load(open(folder + "analysed_data/AR1to1s_halfstim_CM_data.dat", "rb"))
# AR2to1d_halfstim_CM =        pickle.load(open(folder + "analysed_data/AR2to1d_halfstim_CM_data.dat", "rb"))

colors_parent = ['#026473','#E3CC69','#77C8A6','#D96248'];


#%%

# prepare data first

# concatenate data from different experiments for boxplots
linetension_baseline_1to1d = np.concatenate((AR1to1d_halfstim_CM["TEM_data"]["line tension baseline [nN]"], AR1to1d_fullstim_short_CM["TEM_data"]["line tension baseline [nN]"], AR1to1d_fullstim_long_CM["TEM_data"]["line tension baseline [nN]"]))
linetension_baseline_1to1s = np.concatenate((AR1to1s_halfstim_CM["TEM_data"]["line tension baseline [nN]"], AR1to1s_fullstim_short_CM["TEM_data"]["line tension baseline [nN]"], AR1to1s_fullstim_long_CM["TEM_data"]["line tension baseline [nN]"]))
f_adherent_baseline_1to1d = np.concatenate((AR1to1d_halfstim_CM["TEM_data"]["f adherent baseline [nN]"], AR1to1d_fullstim_short_CM["TEM_data"]["f adherent baseline [nN]"], AR1to1d_fullstim_long_CM["TEM_data"]["f adherent baseline [nN]"]))
f_adherent_baseline_1to1s = np.concatenate((AR1to1s_halfstim_CM["TEM_data"]["f adherent baseline [nN]"], AR1to1s_fullstim_short_CM["TEM_data"]["f adherent baseline [nN]"], AR1to1s_fullstim_long_CM["TEM_data"]["f adherent baseline [nN]"]))

sigma_x_CM_baseline_1to1d = np.concatenate((AR1to1d_halfstim_CM["ellipse_data"]["sigma_x_baseline"], AR1to1d_fullstim_short_CM["ellipse_data"]["sigma_x_baseline"], AR1to1d_fullstim_long_CM["ellipse_data"]["sigma_x_baseline"]))
sigma_x_CM_baseline_1to1s = np.concatenate((AR1to1s_halfstim_CM["ellipse_data"]["sigma_x_baseline"], AR1to1s_fullstim_short_CM["ellipse_data"]["sigma_x_baseline"], AR1to1s_fullstim_long_CM["ellipse_data"]["sigma_x_baseline"]))
sigma_y_CM_baseline_1to1d = np.concatenate((AR1to1d_halfstim_CM["ellipse_data"]["sigma_y_baseline"], AR1to1d_fullstim_short_CM["ellipse_data"]["sigma_y_baseline"], AR1to1d_fullstim_long_CM["ellipse_data"]["sigma_y_baseline"]))
sigma_y_CM_baseline_1to1s = np.concatenate((AR1to1s_halfstim_CM["ellipse_data"]["sigma_y_baseline"], AR1to1s_fullstim_short_CM["ellipse_data"]["sigma_y_baseline"], AR1to1s_fullstim_long_CM["ellipse_data"]["sigma_y_baseline"]))

sigma_x_MSM_baseline_1to1d = np.concatenate((AR1to1d_halfstim["MSM_data"]["sigma_xx_baseline"], AR1to1d_fullstim_short["MSM_data"]["sigma_xx_baseline"], AR1to1d_fullstim_long["MSM_data"]["sigma_xx_baseline"]))
sigma_x_MSM_baseline_1to1s = np.concatenate((AR1to1s_halfstim["MSM_data"]["sigma_xx_baseline"], AR1to1s_fullstim_short["MSM_data"]["sigma_xx_baseline"], AR1to1s_fullstim_long["MSM_data"]["sigma_xx_baseline"]))
sigma_y_MSM_baseline_1to1d = np.concatenate((AR1to1d_halfstim["MSM_data"]["sigma_yy_baseline"], AR1to1d_fullstim_short["MSM_data"]["sigma_yy_baseline"], AR1to1d_fullstim_long["MSM_data"]["sigma_yy_baseline"]))
sigma_y_MSM_baseline_1to1s = np.concatenate((AR1to1s_halfstim["MSM_data"]["sigma_yy_baseline"], AR1to1s_fullstim_short["MSM_data"]["sigma_yy_baseline"], AR1to1s_fullstim_long["MSM_data"]["sigma_yy_baseline"]))

# set up pandas data frame to use with seaborn for box- and swarmplots
linetension_baseline = np.concatenate((linetension_baseline_1to1d,linetension_baseline_1to1s))#*1e9 # convert to nN for plotting
f_adherent_baseline = np.concatenate((f_adherent_baseline_1to1d,f_adherent_baseline_1to1s))#*1e9

sigma_x_CM_baseline = np.concatenate((sigma_x_CM_baseline_1to1d,sigma_x_CM_baseline_1to1s))
sigma_y_CM_baseline = np.concatenate((sigma_y_CM_baseline_1to1d,sigma_y_CM_baseline_1to1s))
sigma_x_MSM_baseline = np.concatenate((sigma_x_MSM_baseline_1to1d,sigma_x_MSM_baseline_1to1s))*1e3 # convert to mN/m for plotting
sigma_y_MSM_baseline = np.concatenate((sigma_y_MSM_baseline_1to1d,sigma_y_MSM_baseline_1to1s))*1e3

n_doublets = linetension_baseline_1to1d.shape[0]
n_singlets = linetension_baseline_1to1s.shape[0]

keys1to1d = ['AR1to1d' for i in range(n_doublets)]
keys1to1s = ['AR1to1s' for i in range(n_singlets)]
keys = np.concatenate((keys1to1d,keys1to1s))

data = {'keys': keys, 'linetension': linetension_baseline, 'f_adherent': f_adherent_baseline, 'sigma_x_CM': sigma_x_CM_baseline, 'sigma_x_MSM': sigma_x_MSM_baseline, 'sigma_y_CM': sigma_y_CM_baseline, 'sigma_y_MSM': sigma_y_MSM_baseline}
# Creates DataFrame.
df = pd.DataFrame(data)
#%%
sns.scatterplot(data=df, x='sigma_x_MSM', y='sigma_x_CM',hue='keys')
plt.plot([0,10],[0,10])
plt.show()

sns.scatterplot(data=df, x='sigma_y_MSM', y='sigma_y_CM',hue='keys')
plt.plot([0,10],[0,10])
plt.show()

#%% plot figure 6plus, Correlation plots

# set plot parameters
ylabeloffset = -5
colors = [colors_parent[1],colors_parent[2]];   # defines colors for scatterplot
colors_regplot = ['#000000','#000000','#000000','#000000']; # defines colors for linear regression plot

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4, 1.8))


############# first panel ####################x
x = 'sigma_x_MSM'
y = 'sigma_x_CM'
hue = 'keys'
xmin = 0
xmax = 20
ymin = 0
ymax = 20
xticks = np.arange(0,20.1,5)
yticks = np.arange(0,20.1,5)
xlabel = '$\mathrm{\sigma_{x, MSM}}$'
ylabel = '$\mathrm{\sigma_{x, CM}}$'

sns.set_palette(sns.color_palette(colors))  
sns.scatterplot(data=df, x=x, y=y,hue=hue,ax=axes[0])
sns.set_palette(sns.color_palette(colors_regplot))      # sets colors
sns.regplot(data=df, x=x, y=y, scatter=False,ax=axes[0])

# set labels
axes[0].set_xlabel(xlabel=xlabel)
axes[0].set_ylabel(ylabel=ylabel, labelpad=ylabeloffset)

# remove legend
axes[0].get_legend().remove()

# set limits
axes[0].set_xlim(xmin=xmin,xmax=xmax)
axes[0].set_ylim(ymin=ymin,ymax=ymax)


# Define where you want ticks
plt.sca(axes[0])
plt.xticks(xticks)
plt.yticks(yticks)


# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=True, top=True, left=True, right=True)

corr,p = pearsonr(df['sigma_x_MSM'], df['sigma_x_CM'])

corr = np.round(corr,decimals=4)
# p = np.round(p,decimals=6)

plt.text(5.2,2.5,'Pearson R = ' + str(corr))
plt.text(5.2,1,'p = ' + '{:0.2e}'.format(p))

############# second panel ####################
x = 'sigma_y_MSM'
y = 'sigma_y_CM'
hue = 'keys'
xmin = 0
xmax = 20
ymin = 0
ymax = 20
xticks = np.arange(0,20.1,5)
yticks = np.arange(0,20.1,5)
xlabel = '$\mathrm{\sigma_{y, MSM}}$'
ylabel = '$\mathrm{\sigma_{y, CM}}$'

sns.set_palette(sns.color_palette(colors))  
sns.scatterplot(data=df, x=x, y=y,hue=hue,ax=axes[1])
sns.set_palette(sns.color_palette(colors_regplot))      # sets colors
sns.regplot(data=df, x=x, y=y, scatter=False,ax=axes[1])

# set labels
axes[1].set_xlabel(xlabel=xlabel)
axes[1].set_ylabel(ylabel=ylabel, labelpad=ylabeloffset)

# remove legend
axes[1].get_legend().remove()

# set limits
axes[1].set_xlim(xmin=xmin,xmax=xmax)
axes[1].set_ylim(ymin=ymin,ymax=ymax)


# Define where you want ticks
plt.sca(axes[1])
plt.xticks(xticks)
plt.yticks(yticks)


# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=True, top=True, left=True, right=True)

corr,p = pearsonr(df['sigma_y_MSM'], df['sigma_y_CM'])

corr = np.round(corr,decimals=4)
# p = np.round(p,decimals=6)

plt.text(5.2,2.5,'Pearson R = ' + str(corr))
plt.text(5.2,1,'p = ' + '{:0.2e}'.format(p))

plt.subplots_adjust(wspace=0.4, hspace=0.4)

plt.savefig(folder+'sup.png', dpi=300, bbox_inches="tight")
plt.show()

#%%
# define plot parameters
fig = plt.figure(2, figsize=(5.5, 2))          # figuresize in inches
gs = gridspec.GridSpec(1,3)                     # sets up subplotgrid rows by columns
gs.update(wspace=0.4, hspace=0.25)              # adjusts space in between the boxes in the grid
colors = [colors_parent[1],colors_parent[2]];   # defines colors
sns.set_palette(sns.color_palette(colors))      # sets colors
linewidth_bp = 0.7                              # linewidth of boxplot borders
width = 0.3                                     # width of boxplots
dotsize = 2                                     # size of datapoints in swarmplot
linewidth_sw = 0.3                              # linewidth of boxplot borders
alpha_sw = 1                                    # transparency of dots in swarmplot
alpha_bp = 0.8                                  # transparency of boxplots
ylabeloffset = 1                                # adjusts distance of ylabel to the plot
titleoffset = 3                                 # adjusts distance of title to the plot

##############################################################################
#Generate first panel
##############################################################################
ymin = -100
ymax = 500
stat_annotation_offset = 0.2

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[0,0])

# set plot variables
x = 'keys'
y = 'linetension'
# if test_if_gaussian(spreadingsize_baseline_1to1d,spreadingsize_baseline_1to1s,'Spreading size'):
#     test = 't-test_ind'
# else:
#     test = 'Mann-Whitney'

# create box- and swarmplots
sns.swarmplot(x=x, y=y, data=df, ax=fig_ax,alpha=alpha_sw,linewidth=linewidth_sw, zorder=0, size=dotsize)
bp = sns.boxplot(x=x, y=y, data=df, ax=fig_ax,linewidth=linewidth_bp,notch=True, showfliers = False, width=width)

order = ['AR1to1d', 'AR1to1s']
# test_results = add_stat_annotation(bp, data=df, x=x, y=y, order=order, line_offset_to_box=stat_annotation_offset, box_pairs=[('AR1to1d', 'AR1to1s')],                      
#                                    test=test, text_format='star', loc='inside', verbose=2)

# make boxplots transparent
for patch in bp.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, alpha_bp))

plt.setp(bp.artists, edgecolor = 'k')
plt.setp(bp.lines, color='k')
     
# set labels
fig_ax.set_xticklabels(['doublet', 'singlet'])
fig_ax.set_xlabel(xlabel=None)
fig_ax.set_ylabel(ylabel='A [$\mathrm{\lambda nN}$]', labelpad=ylabeloffset)
fig_ax.set_title(label='Line tension', pad=titleoffset)
fig_ax.set()

# Define where you want ticks
yticks = np.arange(0,501,100)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=False, top=False, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=False, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
#Generate second panel
##############################################################################
ymin = -100
ymax = 500
stat_annotation_offset = -0.1 # adjust y-position of statistical annotation

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[0,1])

# set plot variables
x = 'keys'
y = 'f_adherent'
# if test_if_gaussian(Es_baseline_1to1d,Es_baseline_1to1s,'Force of adherent fiber'):
#     test = 't-test_ind'
# else:
#     test = 'Mann-Whitney'

sns.swarmplot(x=x, y=y, data=df, ax=fig_ax,alpha=alpha_sw,linewidth=linewidth_sw, zorder=0, size=dotsize)
bp = sns.boxplot(x=x, y=y, data=df, ax=fig_ax,linewidth=linewidth_bp,notch=True, showfliers = False, width=width)

order = ['AR1to1d', 'AR1to1s']
# test_results = add_stat_annotation(bp, data=df, x=x, y=y, order=order, line_offset_to_box=stat_annotation_offset, box_pairs=[('AR1to1d', 'AR1to1s')],                      
#                                    test=test, text_format='star', loc='inside', verbose=2)

# make boxplots transparent
for patch in bp.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, alpha_bp))

plt.setp(bp.artists, edgecolor = 'k')
plt.setp(bp.lines, color='k')
     
# set labels
fig_ax.set_xticklabels(['doublet', 'singlet'])
fig_ax.set_xlabel(xlabel=None)
fig_ax.set_ylabel(ylabel='$\mathrm{f_a}$ [nN]', labelpad=ylabeloffset)
fig_ax.set_title(label='Force of adherent fiber', pad=titleoffset)
fig_ax.set()

# Define where you want ticks
yticks = np.arange(0,501,100)
plt.yticks(yticks)

#provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=False, top=False, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=False, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

# ##############################################################################
# #Generate third panel
# ##############################################################################
# ymin = 0
# ymax = 90
# stat_annotation_offset = 0.037 # adjust y-position of statistical annotation

# # the grid spec is rows, then columns
# fig_ax = fig.add_subplot(gs[0,2])

# # set plot variables
# x = 'keys'
# y = 'force_angle'
# if test_if_gaussian(force_angle_baseline_1to1d,force_angle_baseline_1to1s,'Force angles'):
#     test = 't-test_ind'
# else:
#     test = 'Mann-Whitney'

# # create box- and swarmplots
# sns.swarmplot(x=x, y=y, data=df, ax=fig_ax,alpha=alpha_sw,linewidth=linewidth_sw, zorder=0, size=dotsize)
# bp = sns.boxplot(x=x, y=y, data=df, ax=fig_ax,linewidth=linewidth_bp,notch=True, showfliers = False, width=width)

# order = ['AR1to1d', 'AR1to1s']
# test_results = add_stat_annotation(bp, data=df, x=x, y=y, order=order, line_offset_to_box=stat_annotation_offset, box_pairs=[('AR1to1d', 'AR1to1s')],                      
#                                    test=test, text_format='star', loc='inside', verbose=2)

# # make boxplots transparent
# for patch in bp.artists:
#     r, g, b, a = patch.get_facecolor()
#     patch.set_facecolor((r, g, b, alpha_bp))

# plt.setp(bp.artists, edgecolor = 'k')
# plt.setp(bp.lines, color='k')

     
# # set labels
# fig_ax.set_xticklabels(['doublet', 'singlet'])
# fig_ax.set_xlabel(xlabel=None)
# fig_ax.set_ylabel(ylabel=r'$\mathrm{\vartheta}$ [°]', labelpad=ylabeloffset)
# fig_ax.set_title(label='Force angles', pad=titleoffset)
# fig_ax.set()

# # Define where you want ticks
# yticks = np.arange(0,91,15)
# plt.yticks(yticks)

# #provide info on tick parameters
# plt.minorticks_on()
# plt.tick_params(direction='in',which='minor', length=3, bottom=False, top=False, left=True, right=True)
# plt.tick_params(direction='in',which='major', length=6, bottom=False, top=False, left=True, right=True)

# # set limits
# fig_ax.set_ylim(ymin=ymin)
# fig_ax.set_ylim(ymax=ymax)


# # save plot to file
plt.savefig(folder+'fig2C.png', dpi=300, bbox_inches="tight")
plt.show()
#%% plot figure 1E

# prepare data first

# concatenate MSM maps from different experiments and calculate average maps over first 20 frames and all cells to get average maps
sigma_xx_1to1d_average = np.nanmean(np.concatenate((AR1to1d_halfstim["MSM_data"]["sigma_xx"][:,:,0:20,:],AR1to1d_fullstim_short["MSM_data"]["sigma_xx"][:,:,0:20,:],AR1to1d_fullstim_long["MSM_data"]["sigma_xx"][:,:,0:20,:]),axis=3),axis=(2,3))
sigma_xx_1to1s_average = np.nanmean(np.concatenate((AR1to1s_halfstim["MSM_data"]["sigma_xx"][:,:,0:20,:],AR1to1s_fullstim_short["MSM_data"]["sigma_xx"][:,:,0:20,:],AR1to1s_fullstim_long["MSM_data"]["sigma_xx"][:,:,0:20,:]),axis=3),axis=(2,3))

# get one example
sigma_xx_1to1d_example = AR1to1d_halfstim["MSM_data"]["sigma_xx"][:,:,0,1]
sigma_xx_1to1s_example = AR1to1s_halfstim["MSM_data"]["sigma_xx"][:,:,0,3]

# crop maps 
crop_start = 8
crop_end = 84

sigma_xx_1to1d_average_crop = sigma_xx_1to1d_average[crop_start:crop_end,crop_start:crop_end]*1e3 # convert to mN/m
sigma_xx_1to1d_example_crop = sigma_xx_1to1d_example[crop_start:crop_end,crop_start:crop_end]*1e3
sigma_xx_1to1s_average_crop = sigma_xx_1to1s_average[crop_start:crop_end,crop_start:crop_end]*1e3

sigma_xx_1to1s_example_crop = sigma_xx_1to1s_example[crop_start:crop_end,crop_start:crop_end]*1e3

# set up plot parameters
#*****************************************************************************
n=4 # every nth arrow will be plotted
pixelsize = 0.864 # in µm 
pmax = 10 # mN/m


# create x- and y-axis for plotting maps 
x_end = np.shape(sigma_xx_1to1d_average_crop)[1]
y_end = np.shape(sigma_xx_1to1d_average_crop)[0]
extent = [0, x_end*pixelsize, 0, y_end*pixelsize] 

# create mesh for vectorplot    
xq, yq = np.meshgrid(np.linspace(0,extent[1],x_end), np.linspace(0,extent[3],y_end)) 

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(3, 2.5))

im = axes[0,0].imshow(sigma_xx_1to1d_example_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0, vmax=pmax, aspect='auto')

axes[0,1].imshow(sigma_xx_1to1d_average_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0, vmax=pmax, aspect='auto')

axes[1,0].imshow(sigma_xx_1to1s_example_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0, vmax=pmax, aspect='auto')

axes[1,1].imshow(sigma_xx_1to1s_average_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0, vmax=pmax, aspect='auto')

# adjust space in between plots
plt.subplots_adjust(wspace=0.02, hspace=-0.02)

# add annotations
plt.text(-40.5,55.5,'n=1',color = 'white')
plt.text(-40.5,119,'n=1',color = 'white')
plt.text(23,55.5,'n=101',color = 'white')
plt.text(23.5,119,'n=66',color = 'white')

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
plt.suptitle('$\mathrm{\sigma _{xx}}$',y=0.94, x=0.44)

# save figure
fig.savefig(folder+'fig1E.png', dpi=300, bbox_inches="tight")
plt.show()

#%% plot figure 1F

# prepare data first

# concatenate MSM maps from different experiments and calculate average maps over first 20 frames and all cells to get average maps
sigma_yy_1to1d_average = np.nanmean(np.concatenate((AR1to1d_halfstim["MSM_data"]["sigma_yy"][:,:,0:20,:],AR1to1d_fullstim_short["MSM_data"]["sigma_yy"][:,:,0:20,:],AR1to1d_fullstim_long["MSM_data"]["sigma_yy"][:,:,0:20,:]),axis=3),axis=(2,3))
sigma_yy_1to1s_average = np.nanmean(np.concatenate((AR1to1s_halfstim["MSM_data"]["sigma_yy"][:,:,0:20,:],AR1to1s_fullstim_short["MSM_data"]["sigma_yy"][:,:,0:20,:],AR1to1s_fullstim_long["MSM_data"]["sigma_yy"][:,:,0:20,:]),axis=3),axis=(2,3))

# get one example
sigma_yy_1to1d_example = AR1to1d_halfstim["MSM_data"]["sigma_yy"][:,:,0,1]
sigma_yy_1to1s_example = AR1to1s_halfstim["MSM_data"]["sigma_yy"][:,:,0,3]

# crop maps 
crop_start = 8
crop_end = 84

sigma_yy_1to1d_average_crop = sigma_yy_1to1d_average[crop_start:crop_end,crop_start:crop_end]*1e3
sigma_yy_1to1d_example_crop = sigma_yy_1to1d_example[crop_start:crop_end,crop_start:crop_end]*1e3
sigma_yy_1to1s_average_crop = sigma_yy_1to1s_average[crop_start:crop_end,crop_start:crop_end]*1e3
sigma_yy_1to1s_example_crop = sigma_yy_1to1s_example[crop_start:crop_end,crop_start:crop_end]*1e3

# set up plot parameters
#*****************************************************************************
n=4 # every nth arrow will be plotted
pixelsize = 0.864 # in µm 
pmax = 10 # mN/m


# create x- and y-axis for plotting maps 
x_end = np.shape(sigma_xx_1to1d_average_crop)[1]
y_end = np.shape(sigma_xx_1to1d_average_crop)[0]
extent = [0, x_end*pixelsize, 0, y_end*pixelsize] 

# create mesh for vectorplot    
xq, yq = np.meshgrid(np.linspace(0,extent[1],x_end), np.linspace(0,extent[3],y_end)) 

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(3, 2.5))

im = axes[0,0].imshow(sigma_yy_1to1d_example_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0, vmax=pmax, aspect='auto')

axes[0,1].imshow(sigma_yy_1to1d_average_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0, vmax=pmax, aspect='auto')

axes[1,0].imshow(sigma_yy_1to1s_example_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0, vmax=pmax, aspect='auto')

axes[1,1].imshow(sigma_yy_1to1s_average_crop, cmap=plt.get_cmap("turbo"), interpolation="bilinear", extent=extent, vmin=0, vmax=pmax, aspect='auto')

# adjust space in between plots
plt.subplots_adjust(wspace=0.02, hspace=-0.02)

# add annotations
plt.text(-40.5,55.5,'n=1',color = 'white')
plt.text(-40.5,119,'n=1',color = 'white')
plt.text(23,55.5,'n=101',color = 'white')
plt.text(23.5,119,'n=66',color = 'white')

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
plt.suptitle('$\mathrm{\sigma _{yy}}$',y=0.94, x=0.44)

# save figure
fig.savefig(folder+'fig1F.png', dpi=300, bbox_inches="tight")
plt.close()

#%% plot figure 1G
# prepare data first

# concatenate data from different experiments for boxplots
spreadingsize_baseline_1to1d = np.concatenate((AR1to1d_halfstim["shape_data"]["spreadingsize_baseline"], AR1to1d_fullstim_short["shape_data"]["spreadingsize_baseline"], AR1to1d_fullstim_long["shape_data"]["spreadingsize_baseline"]))
spreadingsize_baseline_1to1s = np.concatenate((AR1to1s_halfstim["shape_data"]["spreadingsize_baseline"], AR1to1s_fullstim_short["shape_data"]["spreadingsize_baseline"], AR1to1s_fullstim_long["shape_data"]["spreadingsize_baseline"]))
Es_baseline_1to1d = np.concatenate((AR1to1d_halfstim["TFM_data"]["Es_baseline"], AR1to1d_fullstim_short["TFM_data"]["Es_baseline"], AR1to1d_fullstim_long["TFM_data"]["Es_baseline"]))
Es_baseline_1to1s = np.concatenate((AR1to1s_halfstim["TFM_data"]["Es_baseline"], AR1to1s_fullstim_short["TFM_data"]["Es_baseline"], AR1to1s_fullstim_long["TFM_data"]["Es_baseline"]))
force_angle_baseline_1to1d = np.concatenate((AR1to1d_halfstim["TFM_data"]["force_angle_baseline"], AR1to1d_fullstim_short["TFM_data"]["force_angle_baseline"], AR1to1d_fullstim_long["TFM_data"]["force_angle_baseline"]))
force_angle_baseline_1to1s = np.concatenate((AR1to1s_halfstim["TFM_data"]["force_angle_baseline"], AR1to1s_fullstim_short["TFM_data"]["force_angle_baseline"], AR1to1s_fullstim_long["TFM_data"]["force_angle_baseline"]))
sigma_xx_baseline_1to1d = np.concatenate((AR1to1d_halfstim["MSM_data"]["sigma_xx_baseline"], AR1to1d_fullstim_short["MSM_data"]["sigma_xx_baseline"], AR1to1d_fullstim_long["MSM_data"]["sigma_xx_baseline"]))
sigma_xx_baseline_1to1s = np.concatenate((AR1to1s_halfstim["MSM_data"]["sigma_xx_baseline"], AR1to1s_fullstim_short["MSM_data"]["sigma_xx_baseline"], AR1to1s_fullstim_long["MSM_data"]["sigma_xx_baseline"]))
sigma_yy_baseline_1to1d = np.concatenate((AR1to1d_halfstim["MSM_data"]["sigma_yy_baseline"], AR1to1d_fullstim_short["MSM_data"]["sigma_yy_baseline"], AR1to1d_fullstim_long["MSM_data"]["sigma_yy_baseline"]))
sigma_yy_baseline_1to1s = np.concatenate((AR1to1s_halfstim["MSM_data"]["sigma_yy_baseline"], AR1to1s_fullstim_short["MSM_data"]["sigma_yy_baseline"], AR1to1s_fullstim_long["MSM_data"]["sigma_yy_baseline"]))

# set up pandas data frame to use with seaborn for box- and swarmplots
spreadingsize_baseline = np.concatenate((spreadingsize_baseline_1to1d,spreadingsize_baseline_1to1s))*1e12 # convert to µm²
Es_baseline = np.concatenate((Es_baseline_1to1d,Es_baseline_1to1s))*1e12 # convert to pJ
force_angle_baseline = np.concatenate((force_angle_baseline_1to1d,force_angle_baseline_1to1s))
sigma_xx_baseline = np.concatenate((sigma_xx_baseline_1to1d,sigma_xx_baseline_1to1s))*1e3 # convert to mN/m
sigma_yy_baseline = np.concatenate((sigma_yy_baseline_1to1d,sigma_yy_baseline_1to1s))*1e3

n_doublets = Es_baseline_1to1d.shape[0]
n_singlets = Es_baseline_1to1s.shape[0]

keys1to1d = ['AR1to1d' for i in range(n_doublets)]
keys1to1s = ['AR1to1s' for i in range(n_singlets)]
keys = np.concatenate((keys1to1d,keys1to1s))

data = {'keys': keys, 'spreadingsize': spreadingsize_baseline, 'strain_energy': Es_baseline, 'force_angle': force_angle_baseline, 'sigma_xx_baseline': sigma_xx_baseline, 'sigma_yy_baseline': sigma_yy_baseline}
# Creates DataFrame.
df = pd.DataFrame(data)

# define plot parameters
fig = plt.figure(2, figsize=(9, 2))          # figuresize in inches
gs = gridspec.GridSpec(1,5)                     # sets up subplotgrid rows by columns
gs.update(wspace=0.4, hspace=0.25)              # adjusts space in between the boxes in the grid
colors = [colors_parent[1],colors_parent[2]];   # defines colors
sns.set_palette(sns.color_palette(colors))      # sets colors
linewidth_bp = 0.7                              # linewidth of boxplot borders
width = 0.3                                     # width of boxplots
dotsize = 2                                     # size of datapoints in swarmplot
linewidth_sw = 0.3                              # linewidth of boxplot borders
alpha_sw = 1                                    # transparency of dots in swarmplot
alpha_bp = 0.8                                  # transparency of boxplots
ylabeloffset = 1                                # adjusts distance of ylabel to the plot
titleoffset = 3                                 # adjusts distance of title to the plot

##############################################################################
#Generate first panel
##############################################################################
ymin = 500
ymax = 2000
stat_annotation_offset = .265

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[0,0])

# set plot variables
x = 'keys'
y = 'spreadingsize'
if test_if_gaussian(spreadingsize_baseline_1to1d,spreadingsize_baseline_1to1s,'Spreading size'):
    test = 't-test_ind'
else:
    test = 'Mann-Whitney'

# create box- and swarmplots
sns.swarmplot(x=x, y=y, data=df, ax=fig_ax,alpha=alpha_sw,linewidth=linewidth_sw, zorder=0, size=dotsize)
bp = sns.boxplot(x=x, y=y, data=df, ax=fig_ax,linewidth=linewidth_bp,notch=True, showfliers = False, width=width)

order = ['AR1to1d', 'AR1to1s']
# test_results = add_stat_annotation(bp, data=df, x=x, y=y, order=order, line_offset_to_box=stat_annotation_offset, box_pairs=[('AR1to1d', 'AR1to1s')],                      
#                                    test=test, text_format='star', loc='inside', verbose=2)

# make boxplots transparent
for patch in bp.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, alpha_bp))

plt.setp(bp.artists, edgecolor = 'k')
plt.setp(bp.lines, color='k')
     
# set labels
fig_ax.set_xticklabels(['doublet', 'singlet'])
fig_ax.set_xlabel(xlabel=None)
fig_ax.set_ylabel(ylabel='A [$\mathrm{\mu m^2}$]', labelpad=ylabeloffset)
fig_ax.set_title(label='Spreading size', pad=titleoffset)
fig_ax.set()

# Define where you want ticks
yticks = np.arange(500,2001,250)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=False, top=False, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=False, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
#Generate second panel
##############################################################################
ymin = 0
ymax = 2
stat_annotation_offset = 0.03 # adjust y-position of statistical annotation

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[0,1])

# set plot variables
x = 'keys'
y = 'strain_energy'
if test_if_gaussian(Es_baseline_1to1d,Es_baseline_1to1s,'Strain energy'):
    test = 't-test_ind'
else:
    test = 'Mann-Whitney'

sns.swarmplot(x=x, y=y, data=df, ax=fig_ax,alpha=alpha_sw,linewidth=linewidth_sw, zorder=0, size=dotsize)
bp = sns.boxplot(x=x, y=y, data=df, ax=fig_ax,linewidth=linewidth_bp,notch=True, showfliers = False, width=width)

order = ['AR1to1d', 'AR1to1s']
# test_results = add_stat_annotation(bp, data=df, x=x, y=y, order=order, line_offset_to_box=stat_annotation_offset, box_pairs=[('AR1to1d', 'AR1to1s')],                      
#                                    test=test, text_format='star', loc='inside', verbose=2)

# make boxplots transparent
for patch in bp.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, alpha_bp))

plt.setp(bp.artists, edgecolor = 'k')
plt.setp(bp.lines, color='k')
     
# set labels
fig_ax.set_xticklabels(['doublet', 'singlet'])
fig_ax.set_xlabel(xlabel=None)
fig_ax.set_ylabel(ylabel='$\mathrm{E_s}$ [pJ]', labelpad=ylabeloffset)
fig_ax.set_title(label='Strain energy', pad=titleoffset)
fig_ax.set()

# Define where you want ticks
yticks = np.arange(0,2.1,0.5)
plt.yticks(yticks)

#provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=False, top=False, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=False, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
#Generate third panel
##############################################################################
ymin = 0
ymax = 90
stat_annotation_offset = 0.057 # adjust y-position of statistical annotation

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[0,2])

# set plot variables
x = 'keys'
y = 'force_angle'
if test_if_gaussian(force_angle_baseline_1to1d,force_angle_baseline_1to1s,'Force angles'):
    test = 't-test_ind'
else:
    test = 'Mann-Whitney'

# create box- and swarmplots
sns.swarmplot(x=x, y=y, data=df, ax=fig_ax,alpha=alpha_sw,linewidth=linewidth_sw, zorder=0, size=dotsize)
bp = sns.boxplot(x=x, y=y, data=df, ax=fig_ax,linewidth=linewidth_bp,notch=True, showfliers = False, width=width)

order = ['AR1to1d', 'AR1to1s']
# test_results = add_stat_annotation(bp, data=df, x=x, y=y, order=order, line_offset_to_box=stat_annotation_offset, box_pairs=[('AR1to1d', 'AR1to1s')],                      
#                                    test=test, text_format='star', loc='inside', verbose=2)

# make boxplots transparent
for patch in bp.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, alpha_bp))

plt.setp(bp.artists, edgecolor = 'k')
plt.setp(bp.lines, color='k')

     
# set labels
fig_ax.set_xticklabels(['doublet', 'singlet'])
fig_ax.set_xlabel(xlabel=None)
fig_ax.set_ylabel(ylabel=r'$\mathrm{\vartheta}$ [°]', labelpad=ylabeloffset)
fig_ax.set_title(label='Force angles', pad=titleoffset)
fig_ax.set()

# Define where you want ticks
yticks = np.arange(0,91,15)
plt.yticks(yticks)

#provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=False, top=False, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=False, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
#Generate fourth panel
##############################################################################
ymin = 0
ymax = 14
stat_annotation_offset = 0.21 # adjust y-position of statistical annotation

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[0,3])

# set plot variables
x = 'keys'
y = 'sigma_xx_baseline'
if test_if_gaussian(sigma_xx_baseline_1to1d,sigma_xx_baseline_1to1s,'sigma_xx_baseline'):
    test = 't-test_ind'
else:
    test = 'Mann-Whitney'

sns.swarmplot(x=x, y=y, data=df, ax=fig_ax,alpha=alpha_sw,linewidth=linewidth_sw, zorder=0, size=dotsize)
bp = sns.boxplot(x=x, y=y, data=df, ax=fig_ax,linewidth=linewidth_bp,notch=True, showfliers = False, width=width)

order = ['AR1to1d', 'AR1to1s']
# test_results = add_stat_annotation(bp, data=df, x=x, y=y, order=order, line_offset_to_box=stat_annotation_offset, box_pairs=[('AR1to1d', 'AR1to1s')],                      
                                    # test=test, text_format='star', loc='inside', verbose=2)

# make boxplots transparent
for patch in bp.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, alpha_bp))

plt.setp(bp.artists, edgecolor = 'k')
plt.setp(bp.lines, color='k')
     
# set labels
fig_ax.set_xticklabels(['doublet', 'singlet'])
fig_ax.set_xlabel(xlabel=None)
fig_ax.set_ylabel(ylabel='$\mathrm{\sigma _{xx}}$ [mN/m]', labelpad=ylabeloffset)
fig_ax.set_title(label='xx-Stress', pad=titleoffset)
fig_ax.set()

# Define where you want ticks
yticks = np.arange(0,15,2)
plt.yticks(yticks)

#provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=False, top=False, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=False, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)


##############################################################################
#Generate fifth panel
##############################################################################
ymin = 0
ymax = 14
stat_annotation_offset = 1.55 # adjust y-position of statistical annotation

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[0,4])

# set plot variables
x = 'keys'
y = 'sigma_yy_baseline'
if test_if_gaussian(sigma_xx_baseline_1to1d,sigma_xx_baseline_1to1s,'sigma_yy_baseline'):
    test = 't-test_ind'
else:
    test = 'Mann-Whitney'

sns.swarmplot(x=x, y=y, data=df, ax=fig_ax,alpha=alpha_sw,linewidth=linewidth_sw, zorder=0, size=dotsize)
bp = sns.boxplot(x=x, y=y, data=df, ax=fig_ax,linewidth=linewidth_bp,notch=True, showfliers = False, width=width)

order = ['AR1to1d', 'AR1to1s']
# test_results = add_stat_annotation(bp, data=df, x=x, y=y, order=order, line_offset_to_box=stat_annotation_offset, box_pairs=[('AR1to1d', 'AR1to1s')],                      
#                                     test=test, text_format='star', loc='inside', verbose=2)

# make boxplots transparent
for patch in bp.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, alpha_bp))

plt.setp(bp.artists, edgecolor = 'k')
plt.setp(bp.lines, color='k')
     
# set labels
fig_ax.set_xticklabels(['doublet', 'singlet'])
fig_ax.set_xlabel(xlabel=None)
fig_ax.set_ylabel(ylabel='$\mathrm{\sigma _{yy}}$ [mN/m]', labelpad=ylabeloffset)
fig_ax.set_title(label='yy-Stress', pad=titleoffset)
fig_ax.set()

# Define where you want ticks
yticks = np.arange(0,15,2)
plt.yticks(yticks)

#provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=False, top=False, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=False, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

# save plot to file
plt.savefig(folder+'fig1G.png', dpi=300, bbox_inches="tight")
plt.close()