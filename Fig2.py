# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 21:56:01 2021

@author: Artur Ruppel

"""
import os
import pickle

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from statannot import add_stat_annotation

# mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 8

# %% load data for plotting
folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"

AR1to1d_fullstim_long = pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_long.dat", "rb"))
AR1to1s_fullstim_long = pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_long.dat", "rb"))
AR1to1d_fullstim_short = pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_short.dat", "rb"))
AR1to1s_fullstim_short = pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_short.dat", "rb"))
# AR1to2d_halfstim =        pickle.load(open(folder + "analysed_data/AR1to2d_halfstim.dat", "rb"))
AR1to1d_halfstim = pickle.load(open(folder + "analysed_data/AR1to1d_halfstim.dat", "rb"))
AR1to1s_halfstim = pickle.load(open(folder + "analysed_data/AR1to1s_halfstim.dat", "rb"))
# AR2to1d_halfstim =        pickle.load(open(folder + "analysed_data/AR2to1d_halfstim.dat", "rb"))

AR1to1d_fullstim_long_CM = pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_long_CM_data.dat", "rb"))
AR1to1s_fullstim_long_CM = pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_long_CM_data.dat", "rb"))
AR1to1d_fullstim_short_CM = pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_short_CM_data.dat", "rb"))
AR1to1s_fullstim_short_CM = pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_short_CM_data.dat", "rb"))
# AR1to2d_halfstim_CM =        pickle.load(open(folder + "analysed_data/AR1to2d_halfstim_CM_data.dat", "rb"))
AR1to1d_halfstim_CM = pickle.load(open(folder + "analysed_data/AR1to1d_halfstim_CM_data.dat", "rb"))
AR1to1s_halfstim_CM = pickle.load(open(folder + "analysed_data/AR1to1s_halfstim_CM_data.dat", "rb"))
# AR2to1d_halfstim_CM =        pickle.load(open(folder + "analysed_data/AR2to1d_halfstim_CM_data.dat", "rb"))

colors_parent = ['#026473', '#E3CC69', '#77C8A6', '#D96248']

figfolder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/_Figure2/"
if not os.path.exists(figfolder):
    os.mkdir(figfolder)
# %%

# prepare data first

# concatenate data from different experiments for boxplots
linetension_baseline_1to1d = np.concatenate((AR1to1d_halfstim_CM["TEM_data"]["line tension baseline [nN]"],
                                             AR1to1d_fullstim_short_CM["TEM_data"]["line tension baseline [nN]"],
                                             AR1to1d_fullstim_long_CM["TEM_data"]["line tension baseline [nN]"]))
linetension_baseline_1to1s = np.concatenate((AR1to1s_halfstim_CM["TEM_data"]["line tension baseline [nN]"],
                                             AR1to1s_fullstim_short_CM["TEM_data"]["line tension baseline [nN]"],
                                             AR1to1s_fullstim_long_CM["TEM_data"]["line tension baseline [nN]"]))
f_adherent_baseline_1to1d = np.concatenate((AR1to1d_halfstim_CM["TEM_data"]["f adherent baseline [nN]"],
                                            AR1to1d_fullstim_short_CM["TEM_data"]["f adherent baseline [nN]"],
                                            AR1to1d_fullstim_long_CM["TEM_data"]["f adherent baseline [nN]"]))
f_adherent_baseline_1to1s = np.concatenate((AR1to1s_halfstim_CM["TEM_data"]["f adherent baseline [nN]"],
                                            AR1to1s_fullstim_short_CM["TEM_data"]["f adherent baseline [nN]"],
                                            AR1to1s_fullstim_long_CM["TEM_data"]["f adherent baseline [nN]"]))

sigma_x_CM_baseline_1to1d = np.concatenate((AR1to1d_halfstim_CM["ellipse_data"]["sigma_x_baseline"],
                                            AR1to1d_fullstim_short_CM["ellipse_data"]["sigma_x_baseline"],
                                            AR1to1d_fullstim_long_CM["ellipse_data"]["sigma_x_baseline"]))
sigma_x_CM_baseline_1to1s = np.concatenate((AR1to1s_halfstim_CM["ellipse_data"]["sigma_x_baseline"],
                                            AR1to1s_fullstim_short_CM["ellipse_data"]["sigma_x_baseline"],
                                            AR1to1s_fullstim_long_CM["ellipse_data"]["sigma_x_baseline"]))
sigma_y_CM_baseline_1to1d = np.concatenate((AR1to1d_halfstim_CM["ellipse_data"]["sigma_y_baseline"],
                                            AR1to1d_fullstim_short_CM["ellipse_data"]["sigma_y_baseline"],
                                            AR1to1d_fullstim_long_CM["ellipse_data"]["sigma_y_baseline"]))
sigma_y_CM_baseline_1to1s = np.concatenate((AR1to1s_halfstim_CM["ellipse_data"]["sigma_y_baseline"],
                                            AR1to1s_fullstim_short_CM["ellipse_data"]["sigma_y_baseline"],
                                            AR1to1s_fullstim_long_CM["ellipse_data"]["sigma_y_baseline"]))

sigma_x_MSM_baseline_1to1d = np.concatenate((AR1to1d_halfstim["MSM_data"]["sigma_xx_baseline"],
                                             AR1to1d_fullstim_short["MSM_data"]["sigma_xx_baseline"],
                                             AR1to1d_fullstim_long["MSM_data"]["sigma_xx_baseline"]))
sigma_x_MSM_baseline_1to1s = np.concatenate((AR1to1s_halfstim["MSM_data"]["sigma_xx_baseline"],
                                             AR1to1s_fullstim_short["MSM_data"]["sigma_xx_baseline"],
                                             AR1to1s_fullstim_long["MSM_data"]["sigma_xx_baseline"]))
sigma_y_MSM_baseline_1to1d = np.concatenate((AR1to1d_halfstim["MSM_data"]["sigma_yy_baseline"],
                                             AR1to1d_fullstim_short["MSM_data"]["sigma_yy_baseline"],
                                             AR1to1d_fullstim_long["MSM_data"]["sigma_yy_baseline"]))
sigma_y_MSM_baseline_1to1s = np.concatenate((AR1to1s_halfstim["MSM_data"]["sigma_yy_baseline"],
                                             AR1to1s_fullstim_short["MSM_data"]["sigma_yy_baseline"],
                                             AR1to1s_fullstim_long["MSM_data"]["sigma_yy_baseline"]))

# set up pandas data frame to use with seaborn for box- and swarmplots
linetension_baseline = np.concatenate(
    (linetension_baseline_1to1d, linetension_baseline_1to1s))  # *1e9 # convert to nN for plotting
f_adherent_baseline = np.concatenate((f_adherent_baseline_1to1d, f_adherent_baseline_1to1s))  # *1e9

sigma_x_CM_baseline = np.concatenate((sigma_x_CM_baseline_1to1d, sigma_x_CM_baseline_1to1s))
sigma_y_CM_baseline = np.concatenate((sigma_y_CM_baseline_1to1d, sigma_y_CM_baseline_1to1s))
sigma_x_MSM_baseline = np.concatenate(
    (sigma_x_MSM_baseline_1to1d, sigma_x_MSM_baseline_1to1s)) * 1e3  # convert to mN/m for plotting
sigma_y_MSM_baseline = np.concatenate((sigma_y_MSM_baseline_1to1d, sigma_y_MSM_baseline_1to1s)) * 1e3

n_doublets = linetension_baseline_1to1d.shape[0]
n_singlets = linetension_baseline_1to1s.shape[0]

keys1to1d = ['AR1to1d' for i in range(n_doublets)]
keys1to1s = ['AR1to1s' for i in range(n_singlets)]
keys = np.concatenate((keys1to1d, keys1to1s))

data = {'keys': keys, 'linetension': linetension_baseline, 'f_adherent': f_adherent_baseline,
        'sigma_x_CM': sigma_x_CM_baseline, 'sigma_x_MSM': sigma_x_MSM_baseline, 'sigma_y_CM': sigma_y_CM_baseline,
        'sigma_y_MSM': sigma_y_MSM_baseline}
# Creates DataFrame.
df = pd.DataFrame(data)
# %% plot figure 2D, line tension and force of adherent fiber

# define plot parameters
fig = plt.figure(2, figsize=(5.5, 2))  # figuresize in inches
gs = gridspec.GridSpec(1, 3)  # sets up subplotgrid rows by columns
gs.update(wspace=0.4, hspace=0.25)  # adjusts space in between the boxes in the grid
colors = [colors_parent[1], colors_parent[2]]  # defines colors
sns.set_palette(sns.color_palette(colors))  # sets colors
linewidth_bp = 0.7  # linewidth of boxplot borders
width = 0.3  # width of boxplots
dotsize = 1.8  # size of datapoints in swarmplot
linewidth_sw = 0.3  # linewidth of boxplot borders
alpha_sw = 1  # transparency of dots in swarmplot
alpha_bp = 0.8  # transparency of boxplots
ylabeloffset = 1  # adjusts distance of ylabel to the plot
titleoffset = 3  # adjusts distance of title to the plot

##############################################################################
# Generate first panel
##############################################################################
ymin = -100
ymax = 500
stat_annotation_offset = 0.05

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[0, 0])

# set plot variables
x = 'keys'
y = 'linetension'

test = 'Mann-Whitney'

# create box- and swarmplots
sns.swarmplot(x=x, y=y, data=df, ax=fig_ax, alpha=alpha_sw, linewidth=linewidth_sw, zorder=0, size=dotsize)
bp = sns.boxplot(x=x, y=y, data=df, ax=fig_ax, linewidth=linewidth_bp, notch=True, showfliers=False, width=width)

order = ['AR1to1d', 'AR1to1s']
test_results = add_stat_annotation(bp, data=df, x=x, y=y, order=order, line_offset_to_box=stat_annotation_offset, box_pairs=[('AR1to1d', 'AR1to1s')],
                                    test=test, text_format='star', loc='inside', verbose=2)

# make boxplots transparent
for patch in bp.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, alpha_bp))

plt.setp(bp.artists, edgecolor='k')
plt.setp(bp.lines, color='k')

# set labels
fig_ax.set_xticklabels(['doublet', 'singlet'])
fig_ax.set_xlabel(xlabel=None)
fig_ax.set_ylabel(ylabel='A [$\mathrm{\lambda nN}$]', labelpad=ylabeloffset)
fig_ax.set_title(label='Line tension', pad=titleoffset)
fig_ax.set()

# Define where you want ticks
yticks = np.arange(0, 501, 100)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in', which='minor', length=3, bottom=False, top=False, left=True, right=True)
plt.tick_params(direction='in', which='major', length=6, bottom=False, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)

##############################################################################
# Generate second panel
##############################################################################
ymin = -100
ymax = 500
stat_annotation_offset = 0.55  # adjust y-position of statistical annotation

# the grid spec is rows, then columns
fig_ax = fig.add_subplot(gs[0, 1])

# set plot variables
x = 'keys'
y = 'f_adherent'
# if test_if_gaussian(Es_baseline_1to1d,Es_baseline_1to1s,'Force of adherent fiber'):
#     test = 't-test_ind'
# else:
#     test = 'Mann-Whitney'

sns.swarmplot(x=x, y=y, data=df, ax=fig_ax, alpha=alpha_sw, linewidth=linewidth_sw, zorder=0, size=dotsize)
bp = sns.boxplot(x=x, y=y, data=df, ax=fig_ax, linewidth=linewidth_bp, notch=True, showfliers=False, width=width)

order = ['AR1to1d', 'AR1to1s']
test_results = add_stat_annotation(bp, data=df, x=x, y=y, order=order, line_offset_to_box=stat_annotation_offset, box_pairs=[('AR1to1d', 'AR1to1s')],
                                   test=test, text_format='star', loc='inside', verbose=2)

# make boxplots transparent
for patch in bp.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, alpha_bp))

plt.setp(bp.artists, edgecolor='k')
plt.setp(bp.lines, color='k')

# set labels
fig_ax.set_xticklabels(['doublet', 'singlet'])
fig_ax.set_xlabel(xlabel=None)
fig_ax.set_ylabel(ylabel='$\mathrm{f_a}$ [nN]', labelpad=ylabeloffset)
fig_ax.set_title(label='Force of adherent fiber', pad=titleoffset)
fig_ax.set()

# Define where you want ticks
yticks = np.arange(0, 501, 100)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in', which='minor', length=3, bottom=False, top=False, left=True, right=True)
plt.tick_params(direction='in', which='major', length=6, bottom=False, top=False, left=True, right=True)

# set limits
fig_ax.set_ylim(ymin=ymin)
fig_ax.set_ylim(ymax=ymax)



# # save plot to file
plt.savefig(figfolder + 'D.png', dpi=300, bbox_inches="tight")
plt.show()

# %% plot figure 2E, Correlation plots of sigma_MSM and sigma_CM

# set plot parameters
ylabeloffset = 0
xlabeloffset = 0
colors = [colors_parent[1], colors_parent[2]]  # defines colors for scatterplot
colors_regplot = ['#000000', '#000000', '#000000', '#000000']  # defines colors for linear regression plot

dotsize = 1.8  # size of datapoints in scatterplot
linewidth_sw = 0.3  # linewidth of dots in scatterplot
alpha_sw = 1  # transparency of dots in scatterplot

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4, 1.8))

##############################################################################
# Generate first panel
##############################################################################
x = 'sigma_x_MSM'
y = 'sigma_x_CM'
hue = 'keys'
xmin = 0
xmax = 20
ymin = 0
ymax = 20
xticks = np.arange(0, 20.1, 5)
yticks = np.arange(0, 20.1, 5)
xlabel = '$\mathrm{\sigma_{x, MSM}}$'
ylabel = '$\mathrm{\sigma_{x, CM}}$'

sns.set_palette(sns.color_palette(colors))
sns.scatterplot(data=df, x=x, y=y, hue=hue, style=hue, ax=axes[0], alpha=alpha_sw, linewidth=linewidth_sw, size=dotsize)
sns.set_palette(sns.color_palette(colors_regplot))  # sets colors
sns.regplot(data=df, x=x, y=y, scatter=False, ax=axes[0])

# add line with slope 1 for visualisation
axes[0].plot([0, xmax], [0, ymax], linewidth=0.5, linestyle=':', color='grey')

# set labels
axes[0].set_xlabel(xlabel=xlabel, labelpad=xlabeloffset)
axes[0].set_ylabel(ylabel=ylabel, labelpad=ylabeloffset)

# remove legend
axes[0].get_legend().remove()

# set limits
axes[0].set_xlim(xmin=xmin, xmax=xmax)
axes[0].set_ylim(ymin=ymin, ymax=ymax)

# Define where you want ticks
plt.sca(axes[0])
plt.xticks(xticks)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in', which='minor', length=3, bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in', which='major', length=6, bottom=True, top=True, left=True, right=True)

corr, p = pearsonr(df['sigma_x_MSM'], df['sigma_x_CM'])

corr = np.round(corr, decimals=3)
# p = np.round(p,decimals=6)

plt.text(8, 2.5, 'R = ' + str(corr))
plt.text(8, 1, 'p = ' + '{:0.2e}'.format(p))

##############################################################################
# Generate second panel
##############################################################################
x = 'sigma_y_MSM'
y = 'sigma_y_CM'
hue = 'keys'
xmin = 0
xmax = 10
ymin = 0
ymax = 10
xticks = np.arange(0, 10.1, 5)
yticks = np.arange(0, 10.1, 5)
xlabel = '$\mathrm{\sigma_{y, MSM}}$'
ylabel = '$\mathrm{\sigma_{y, CM}}$'

sns.set_palette(sns.color_palette(colors))
sns.scatterplot(data=df, x=x, y=y, hue=hue, style=hue, ax=axes[1], alpha=alpha_sw, linewidth=linewidth_sw, size=dotsize)
sns.set_palette(sns.color_palette(colors_regplot))  # sets colors
sns.regplot(data=df, x=x, y=y, scatter=False, ax=axes[1])

# add line with slope 1 for visualisation
axes[1].plot([0, xmax], [0, ymax], linewidth=0.5, linestyle=':', color='grey')

# set labels
axes[1].set_xlabel(xlabel=xlabel, labelpad=xlabeloffset)
axes[1].set_ylabel(ylabel=ylabel, labelpad=ylabeloffset)

# adjust legend
L = axes[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
L.get_texts()[0].set_text('doublet')
L.get_texts()[1].set_text('singlet')
L.get_texts()[2].set_text('regression')
# L.get_texts()[3].set_text('$\mathrm{\sigma_{MSM}}$ = $\mathrm{\sigma_{CM}}$')
# axes[1].legend()


# set limits
axes[1].set_xlim(xmin=xmin, xmax=xmax)
axes[1].set_ylim(ymin=ymin, ymax=ymax)

# Define where you want ticks
plt.sca(axes[1])
plt.xticks(xticks)
plt.yticks(yticks)

# provide info on tick parameters
plt.minorticks_on()
plt.tick_params(direction='in', which='minor', length=3, bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in', which='major', length=6, bottom=True, top=True, left=True, right=True)

corr, p = pearsonr(df['sigma_y_MSM'], df['sigma_y_CM'])

corr = np.round(corr, decimals=3)
# p = np.round(p,decimals=6)

plt.text(4, 1.25, 'R = ' + str(corr))
plt.text(4, 0.5, 'p = ' + '{:0.2e}'.format(p))

plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.savefig(figfolder + 'E.png', dpi=300, bbox_inches="tight")
plt.show()

