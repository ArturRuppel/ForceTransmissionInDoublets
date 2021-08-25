# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 18:22:07 2021

@author: Balland
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


colors_parent = ['#026473','#E3CC69','#77C8A6','#D96248'];
#%% load data for plotting
folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"
    
AR1to1d_fullstim_long =   pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_long.dat", "rb"))
AR1to1s_fullstim_long =   pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_long.dat", "rb"))
AR1to1d_fullstim_short =  pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_short.dat", "rb"))
AR1to1s_fullstim_short =  pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_short.dat", "rb"))
AR1to2d_halfstim =        pickle.load(open(folder + "analysed_data/AR1to2d_halfstim.dat", "rb"))
AR1to1d_halfstim =        pickle.load(open(folder + "analysed_data/AR1to1d_halfstim.dat", "rb"))
AR1to1s_halfstim =        pickle.load(open(folder + "analysed_data/AR1to1s_halfstim.dat", "rb"))
AR2to1d_halfstim =        pickle.load(open(folder + "analysed_data/AR2to1d_halfstim.dat", "rb"))

AR1to1d_fullstim_long_CM =   pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_long_CM.dat", "rb"))
AR1to1s_fullstim_long_CM =   pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_long_CM.dat", "rb"))
AR1to1d_fullstim_short_CM =  pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_short_CM.dat", "rb"))
AR1to1s_fullstim_short_CM =  pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_short_CM.dat", "rb"))
AR1to2d_halfstim_CM =        pickle.load(open(folder + "analysed_data/AR1to2d_halfstim_CM.dat", "rb"))
AR1to1d_halfstim_CM =        pickle.load(open(folder + "analysed_data/AR1to1d_halfstim_CM.dat", "rb"))
AR1to1s_halfstim_CM =        pickle.load(open(folder + "analysed_data/AR1to1s_halfstim_CM.dat", "rb"))
AR2to1d_halfstim_CM =        pickle.load(open(folder + "analysed_data/AR2to1d_halfstim_CM.dat", "rb"))


colors_parent = ['#026473','#E3CC69','#77C8A6','#D96248'];

#%% Plot panel X1

# prepare data first

# concatenate data from different experiments for boxplots
sigma_x_MSM_1to1d   = np.concatenate((AR1to1d_halfstim["MSM_data"]["sigma_xx_baseline"], AR1to1d_fullstim_short["MSM_data"]["sigma_xx_baseline"], AR1to1d_fullstim_long["MSM_data"]["sigma_xx_baseline"]))
sigma_x_CM_1to1d    = np.concatenate((AR1to1d_halfstim_CM["ellipse_data"]["sigma_x_baseline"], AR1to1d_fullstim_short_CM["ellipse_data"]["sigma_x_baseline"], AR1to1d_fullstim_long_CM["ellipse_data"]["sigma_x_baseline"]))
sigma_x_MSM_1to1s   = np.concatenate((AR1to1s_halfstim["MSM_data"]["sigma_xx_baseline"], AR1to1s_fullstim_short["MSM_data"]["sigma_xx_baseline"], AR1to1s_fullstim_long["MSM_data"]["sigma_xx_baseline"]))
sigma_x_CM_1to1s    = np.concatenate((AR1to1s_halfstim_CM["ellipse_data"]["sigma_x_baseline"], AR1to1s_fullstim_short_CM["ellipse_data"]["sigma_x_baseline"], AR1to1s_fullstim_long_CM["ellipse_data"]["sigma_x_baseline"]))
sigma_x_MSM_1to2d   = AR1to2d_halfstim["MSM_data"]["sigma_xx_baseline"]
sigma_x_CM_1to2d    = AR1to2d_halfstim_CM["ellipse_data"]["sigma_x_baseline"]
sigma_x_MSM_2to1d   = AR2to1d_halfstim["MSM_data"]["sigma_xx_baseline"]
sigma_x_CM_2to1d    = AR2to1d_halfstim_CM["ellipse_data"]["sigma_x_baseline"]


# set up pandas data frame to use with seaborn for box- and swarmplots
sigma_x_MSM_baseline = np.concatenate((sigma_x_MSM_1to1d,sigma_x_MSM_1to1s,sigma_x_MSM_1to2d,sigma_x_MSM_2to1d))*1e3
sigma_x_CM_baseline = np.concatenate((sigma_x_CM_1to1d,sigma_x_CM_1to1s,sigma_x_CM_1to2d,sigma_x_CM_2to1d))

n_1to1d = sigma_x_MSM_1to1d.shape[0]
n_1to1s = sigma_x_MSM_1to1s.shape[0]
n_1to2d = sigma_x_MSM_1to2d.shape[0]
n_2to1d = sigma_x_MSM_2to1d.shape[0]

keys1to1d = ['AR1to1d' for i in range(n_1to1d)]
keys1to1s = ['AR1to1s' for i in range(n_1to1s)]
keys1to2d = ['AR1to2d' for i in range(n_1to2d)]
keys2to1d = ['AR2to1d' for i in range(n_2to1d)]

keys = np.concatenate((keys1to1d,keys1to1s,keys1to2d,keys2to1d))

data = {'keys': keys, 'sigma_x_MSM_baseline': sigma_x_MSM_baseline, 'sigma_x_CM_baseline': sigma_x_CM_baseline}
# Creates DataFrame.
df = pd.DataFrame(data)
# df=df.drop([174])
# df=df.drop([183])
sns.scatterplot(data=df, x='sigma_x_MSM_baseline', y='sigma_x_CM_baseline',hue='keys')
corr, _ = pearsonr(df['sigma_x_MSM_baseline'], df['sigma_x_CM_baseline'])
plt.savefig(folder+'fig_xxcorrleation.png', dpi=300, bbox_inches="tight")
plt.show()

#%% Plot panel X1

# prepare data first

# concatenate data from different experiments for boxplots
sigma_y_MSM_1to1d   = np.concatenate((AR1to1d_halfstim["MSM_data"]["sigma_yy_baseline"], AR1to1d_fullstim_short["MSM_data"]["sigma_yy_baseline"], AR1to1d_fullstim_long["MSM_data"]["sigma_yy_baseline"]))
sigma_y_CM_1to1d    = np.concatenate((AR1to1d_halfstim_CM["ellipse_data"]["sigma_y_baseline"], AR1to1d_fullstim_short_CM["ellipse_data"]["sigma_y_baseline"], AR1to1d_fullstim_long_CM["ellipse_data"]["sigma_y_baseline"]))
sigma_y_MSM_1to1s   = np.concatenate((AR1to1s_halfstim["MSM_data"]["sigma_yy_baseline"], AR1to1s_fullstim_short["MSM_data"]["sigma_yy_baseline"], AR1to1s_fullstim_long["MSM_data"]["sigma_yy_baseline"]))
sigma_y_CM_1to1s    = np.concatenate((AR1to1s_halfstim_CM["ellipse_data"]["sigma_y_baseline"], AR1to1s_fullstim_short_CM["ellipse_data"]["sigma_y_baseline"], AR1to1s_fullstim_long_CM["ellipse_data"]["sigma_y_baseline"]))
sigma_y_MSM_1to2d   = AR1to2d_halfstim["MSM_data"]["sigma_yy_baseline"]
sigma_y_CM_1to2d    = AR1to2d_halfstim_CM["ellipse_data"]["sigma_y_baseline"]
sigma_y_MSM_2to1d   = AR2to1d_halfstim["MSM_data"]["sigma_yy_baseline"]
sigma_y_CM_2to1d    = AR2to1d_halfstim_CM["ellipse_data"]["sigma_y_baseline"]


# set up pandas data frame to use with seaborn for box- and swarmplots
sigma_y_MSM_baseline = np.concatenate((sigma_y_MSM_1to1d,sigma_y_MSM_1to1s,sigma_y_MSM_1to2d,sigma_y_MSM_2to1d))*1e3
sigma_y_CM_baseline = np.concatenate((sigma_y_CM_1to1d,sigma_y_CM_1to1s,sigma_y_CM_1to2d,sigma_y_CM_2to1d))

n_1to1d = sigma_y_MSM_1to1d.shape[0]
n_1to1s = sigma_y_MSM_1to1s.shape[0]
n_1to2d = sigma_y_MSM_1to2d.shape[0]
n_2to1d = sigma_y_MSM_2to1d.shape[0]

keys1to1d = ['AR1to1d' for i in range(n_1to1d)]
keys1to1s = ['AR1to1s' for i in range(n_1to1s)]
keys1to2d = ['AR1to2d' for i in range(n_1to2d)]
keys2to1d = ['AR2to1d' for i in range(n_2to1d)]

keys = np.concatenate((keys1to1d,keys1to1s,keys1to2d,keys2to1d))

data = {'keys': keys, 'sigma_y_MSM_baseline': sigma_y_MSM_baseline, 'sigma_y_CM_baseline': sigma_y_CM_baseline}
# Creates DataFrame.
df = pd.DataFrame(data)
df=df.drop([174])
df=df.drop([183])
sns.scatterplot(data=df, x='sigma_y_MSM_baseline', y='sigma_y_CM_baseline',hue='keys')
corr, _ = pearsonr(df['sigma_y_MSM_baseline'], df['sigma_y_CM_baseline'])
plt.savefig(folder+'fig_yycorrleation.png', dpi=300, bbox_inches="tight")
plt.show()

#%% plot std

# prepare data first

# concatenate data from different experiments for boxplots
std_circle_1to1d   = np.concatenate((AR1to1d_halfstim_CM["circle_fit_data"]["std_baseline [px]"], AR1to1d_fullstim_short_CM["circle_fit_data"]["std_baseline [px]"], AR1to1d_fullstim_long_CM["circle_fit_data"]["std_baseline [px]"]))
std_ellipse_1to1d   = np.concatenate((AR1to1d_halfstim_CM["ellipse_data"]["std_baseline"], AR1to1d_fullstim_short_CM["ellipse_data"]["std_baseline"], AR1to1d_fullstim_long_CM["ellipse_data"]["std_baseline"]))# sigma_y_MSM_1to1s   = np.concatenate((AR1to1s_halfstim["MSM_data"]["sigma_yy_baseline"], AR1to1s_fullstim_short["MSM_data"]["sigma_yy_baseline"], AR1to1s_fullstim_long["MSM_data"]["sigma_yy_baseline"]))
std_circle_1to1s   = np.concatenate((AR1to1s_halfstim_CM["circle_fit_data"]["std_baseline [px]"], AR1to1s_fullstim_short_CM["circle_fit_data"]["std_baseline [px]"], AR1to1s_fullstim_long_CM["circle_fit_data"]["std_baseline [px]"]))
std_ellipse_1to1s   = np.concatenate((AR1to1s_halfstim_CM["ellipse_data"]["std_baseline"], AR1to1s_fullstim_short_CM["ellipse_data"]["std_baseline"], AR1to1s_fullstim_long_CM["ellipse_data"]["std_baseline"]))# sigma_y_MSM_1to1s   = np.concatenate((AR1to1s_halfstim["MSM_data"]["sigma_yy_baseline"], AR1to1s_fullstim_short["MSM_data"]["sigma_yy_baseline"], AR1to1s_fullstim_long["MSM_data"]["sigma_yy_baseline"]))


# set up pandas data frame to use with seaborn for box- and swarmplots
std_circle = np.concatenate((std_circle_1to1d,std_circle_1to1s))*0.108 #convert to µm for plotting
std_ellipse = np.concatenate((std_ellipse_1to1d,std_ellipse_1to1s))

n_1to1d = std_circle_1to1d.shape[0]
n_1to1s = std_circle_1to1s.shape[0]

keys1to1d = ['AR1to1d' for i in range(n_1to1d)]
keys1to1s = ['AR1to1s' for i in range(n_1to1s)]

keys = np.concatenate((keys1to1d,keys1to1s))

data = {'keys': keys, 'std_circle': std_circle, 'std_ellipse': std_ellipse}
# Creates DataFrame.
df = pd.DataFrame(data)



# define plot parameters
fig = plt.figure(2, figsize=(5.5, 2))          # figuresize in inches
gs = gridspec.GridSpec(1,2)                     # sets up subplotgrid rows by columns
gs.update(wspace=0.4, hspace=0.25)              # adjusts space in between the boxes in the grid
colors = [colors_parent[1],colors_parent[1]];   # defines colors
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
ymin = 0
ymax = 1
# stat_annotation_offset = 0.2

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[0,0])

# set plot variables
# x = 'keys'
# y = 'spreadingsize'
# if test_if_gaussian(spreadingsize_baseline_1to1d,spreadingsize_baseline_1to1s,'Spreading size'):
#     test = 't-test_ind'
# else:
#     test = 'Mann-Whitney'

# create box- and swarmplots
sns.swarmplot(data=df, ax=fig_ax,alpha=alpha_sw,linewidth=linewidth_sw, zorder=0, size=dotsize)
bp = sns.boxplot(data=df, ax=fig_ax,linewidth=linewidth_bp,notch=True, showfliers = False, width=width)

# order = ['std_circle', 'st_ellipse']
# test_results = add_stat_annotation(bp, data=df, x=x, y=y, order=order, line_offset_to_box=stat_annotation_offset, box_pairs=[('AR1to1d', 'AR1to1s')],                      
#                                    test=test, text_format='star', loc='inside', verbose=2)

# make boxplots transparent
for patch in bp.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, alpha_bp))

plt.setp(bp.artists, edgecolor = 'k')
plt.setp(bp.lines, color='k')
     
# set labels
# fig_ax.set_xticklabels(['doublet', 'singlet'])
# fig_ax.set_xlabel(xlabel=None)
fig_ax.set_ylabel(ylabel='std [$\mathrm{\mu m}$]', labelpad=ylabeloffset)
fig_ax.set_title(label='Standard deviation', pad=titleoffset)
fig_ax.set()

# Define where you want ticks
yticks = np.arange(0,1.1,0.5)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3, bottom=False, top=False, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, bottom=False, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

plt.savefig(folder+'fig_std.png', dpi=300, bbox_inches="tight")
plt.show()
# ##############################################################################
# #Generate second panel
# ##############################################################################
# ymin = 0
# ymax = 2
# stat_annotation_offset = -0.1 # adjust y-position of statistical annotation

# # the grid spec is rows, then columns
# fig_ax = fig.add_subplot(gs[0,1])

# # set plot variables
# x = 'keys'
# y = 'strain_energy'
# # if test_if_gaussian(Es_baseline_1to1d,Es_baseline_1to1s,'Strain energy'):
# #     test = 't-test_ind'
# # else:
# #     test = 'Mann-Whitney'

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
# fig_ax.set_ylabel(ylabel='$\mathrm{E_s}$ [pJ]', labelpad=ylabeloffset)
# fig_ax.set_title(label='Strain energy', pad=titleoffset)
# fig_ax.set()

# # Define where you want ticks
# yticks = np.arange(0,2.1,0.5)
# plt.yticks(yticks)

# #provide info on tick parameters
# plt.minorticks_on()
# plt.tick_params(direction='in',which='minor', length=3, bottom=False, top=False, left=True, right=True)
# plt.tick_params(direction='in',which='major', length=6, bottom=False, top=False, left=True, right=True)

# # set limits
# fig_ax.set_ylim(ymin=ymin)
# fig_ax.set_ylim(ymax=ymax)
#%% Plot panel X2