import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline, UnivariateSpline
# from matplotlib import gridspec
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
import sys
sys.path.append('/Users/dennis/customPyModules')
from pubproplot import *

save_directory = 'results/'
import os
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

no_PA = True
model_switch = True
data = np.load('results.npz')

u = data['u']
u_y = data['u_y']
u_g = data['u_gamma']
Es = data['Es']
Es_l = data['Es_l']
Es_r = data['Es_r']
t = data['time']
x = data['x']
l0 = data['l0']
du = np.diff(u, n=1, axis=0)
dt = t[1]-t[0]
vf = du/dt *3600 # GET IT IN um/h

act_t = 1200
t_off = 1680
tau_act = 150
tau_rel = 400
center = 1000

sigma_0 = 0.3e-3

t_act_idx = np.argwhere(t == act_t)[0][0]
newt = t
sigma_a_python = sigma_0 * (1 - np.exp(-newt/ tau_act)) * (1 - 1 / (1 + np.exp((-(newt - center) / tau_rel))))

print(sigma_a_python)
# Activation Profile Plot ##########################################################################################################################
latexify_sf_thesis()
fig = pplt.figure(refaspect=1.61)
ax = fig.add_subplot(1,1,1)

# ax.plot(t ,Es/Es[t_act_idx] ,color='k',lw=2,label=r'Es')


ax.plot(newt+1200,sigma_a_python ,color='k',lw=2,label=r'PA stress')
ax.plot(newt+1200,sigma_0 * (1 - np.exp(-newt/ tau_act)),color='gray',label=r'up')
ax.plot(newt+1200,sigma_0 * (1 - 1 / (1 + np.exp((-(newt - center) / tau_rel)))),color='C3',label=r'down')

# ax.axvline(ymin=0, ymax=1, x=t[t_act_idx], lw=1, color='bright sky blue')

ax.format(xtickminor=False,ytickminor=False)

ax.set_ylabel(r'activation Profile')
ax.set_xlabel(r'$t$ [s]')

# legend1 = ax.legend(handles=legend_elements,loc='best',ncols=1)
# legend2 =ax.legend(handles=additional_legend_elements,loc='upper left')
ax.legend(loc='upper right',ncols=1,frame=False)
# ax.add_artist(legend1)
# ax.set_title('Cell')
ax.set_xlim(0,3600)
# ax.set_ylim(0.5,1.5)
# ax.set_ylim(-0.5,5)
format_axes(ax)
fig.savefig(save_directory+'PA_stress.pdf',bbox_inches='tight')
fig.savefig(save_directory+'PA_stress.svg')


# Displacement Field Time Evolution ##########################################################################################################################
latexify_sf_thesis()
fig = pplt.figure(refaspect=1.61)
ax = fig.add_subplot(1,1,1)

for t_idx in range(len(t)):
    a = t[t_idx] / t[-1]
    if t_idx%2 ==0:
        if t[t_idx] <= act_t:
            ax.plot(x, u[t_idx], color='gray', lw=0.6)
        else:
            ax.plot(x, u[t_idx], color='C3', lw=0.4,alpha=a)

for t_idx in range(len(t)):
    scale = l0/np.max(l0)

    if t_idx%2 ==0:
        if t[t_idx] >= act_t:
            color = pplt.scale_luminance('bright sky blue', scale[t_idx])
            if no_PA is False:
                ax.axvline(ymin=0, ymax=1, x=l0[t_idx], lw=0.2, c=color)

if model_switch is True:
    ax.axvline(ymin=0, ymax=1, x=l0[t_act_idx], lw=1, color='k', label=r'switch')


# ax.plot(x ,u[50] ,color='C0',lw=2,label=r'$u_{\ms tot}$')
# ax.plot(x ,u_y[50] ,color='C1',lw=2,label=r'$u_{\ms y}$')
# ax.plot(x ,u_g[50] ,color='C2',lw=2,label=r'$u_{\ms \gamma}$')

ax.format(xtickminor=True,ytickminor=False)

ax.set_ylabel(r'$u$ [$\mu$m]')
ax.set_xlabel(r'$x$ [um]')

# legend1 = ax.legend(handles=legend_elements,loc='best',ncols=1)
# legend2 =ax.legend(handles=additional_legend_elements,loc='upper left')
ax.legend(loc='lower right',ncols=1,frame=False)
# ax.add_artist(legend1)
# ax.set_title('Cell')
# ax.set_xlim(0,5)
# ax.set_ylim(-0.1,0.3)
# ax.set_ylim(-0.5,5)
format_axes(ax)
fig.savefig(save_directory+'u(x,t).pdf',bbox_inches='tight')
fig.savefig(save_directory+'u(x,t).svg')

# U_gamma time evolution ##########################################################################################################################
latexify_sf_thesis()
fig = pplt.figure(refaspect=1.61)
ax = fig.add_subplot(1,1,1)

for t_idx in range(len(t)):
    a = t[t_idx] / t[-1]
    if t_idx%2 ==0:
        if t[t_idx] <= act_t:
            ax.plot(x, u_g[t_idx], color='gray', lw=0.6)
        else:
            ax.plot(x, u_g[t_idx], color='C3', lw=0.4,alpha=a)

for t_idx in range(len(t)):
    if t_idx%2 ==0:
        if t[t_idx] >= act_t:
            if no_PA is False:
                ax.axvline(ymin=0, ymax=1, x=l0[t_idx], lw=0.2, color='bright sky blue')


if model_switch is True:
    ax.axvline(ymin=0, ymax=1, x=l0[t_act_idx], lw=1, color='k', label=r'switch')

# ax.plot(x ,u[50] ,color='C0',lw=2,label=r'$u_{\ms tot}$')
# ax.plot(x ,u_y[50] ,color='C1',lw=2,label=r'$u_{\ms y}$')
# ax.plot(x ,u_g[50] ,color='C2',lw=2,label=r'$u_{\ms \gamma}$')

ax.format(xtickminor=True,ytickminor=False)

ax.set_ylabel(r'$u_{\gamma}$ [$\mu$m]')
ax.set_xlabel(r'$x$ [um]')

# legend1 = ax.legend(handles=legend_elements,loc='best',ncols=1)
# legend2 =ax.legend(handles=additional_legend_elements,loc='upper left')
ax.legend(loc='lower right',ncols=1,frame=False)
# ax.add_artist(legend1)
# ax.set_title('Cell')
# ax.set_xlim(0,5)
# ax.set_ylim(-0.1,0.3)
# ax.set_ylim(-0.5,5)
format_axes(ax)
fig.savefig(save_directory+'u_g(x,t).pdf',bbox_inches='tight')
fig.savefig(save_directory+'u_g(x,t).svg')

# U_Y time evolution ##########################################################################################################################
latexify_sf_thesis()
fig = pplt.figure(refaspect=1.61)
ax = fig.add_subplot(1,1,1)

for t_idx in range(len(t)):
    a = t[t_idx]/t[-1]
    if t_idx%2 ==0:
        if t[t_idx] <= act_t:
            ax.plot(x, u_y[t_idx], color='gray', lw=0.6)
        else:
            ax.plot(x, u_y[t_idx], color='C3', lw=0.2,alpha=a)

for t_idx in range(len(t)):
    if t_idx%2 ==0:
        if t[t_idx] >= act_t:
            if no_PA is False:
                ax.axvline(ymin=0, ymax=1, x=l0[t_idx], lw=0.2, color='bright sky blue')



if model_switch is True:
    ax.axvline(ymin=0, ymax=1, x=l0[t_act_idx], lw=1, color='k', label=r'switch')

# ax.plot(x ,u[50] ,color='C0',lw=2,label=r'$u_{\ms tot}$')
# ax.plot(x ,u_y[50] ,color='C1',lw=2,label=r'$u_{\ms y}$')
# ax.plot(x ,u_g[50] ,color='C2',lw=2,label=r'$u_{\ms \gamma}$')

ax.format(xtickminor=True,ytickminor=False)

ax.set_ylabel(r'$u_{Y}$ [$\mu$m]')
ax.set_xlabel(r'$x$ [um]')

# legend1 = ax.legend(handles=legend_elements,loc='best',ncols=1)
# legend2 =ax.legend(handles=additional_legend_elements,loc='upper left')
ax.legend(loc='lower right',ncols=1,frame=False)
# ax.add_artist(legend1)
# ax.set_title('Cell')
# ax.set_xlim(0,5)
# ax.set_ylim(-0.1,0.3)
# ax.set_ylim(-0.5,5)
format_axes(ax)
fig.savefig(save_directory+'u_y(x,t).pdf',bbox_inches='tight')
fig.savefig(save_directory+'u_y(x,t).svg')


# define some colors for the plots
colors_parent = ['#026473','#E3CC69','#77C8A6','#D96248'];

# Strain Energy Plot ##########################################################################################################################
latexify_sf_thesis()
fig = pplt.figure(refaspect=1.61)
ax = fig.add_subplot(1,1,1)
t_act_idx = np.argwhere(t == act_t)[0][0]
print(t_act_idx)
# ax.plot(t ,Es/Es[t_act_idx] ,color='k',lw=2,label=r'Es')
color_left = pplt.scale_luminance(colors_parent[2], 1)
color_right = pplt.scale_luminance(colors_parent[2], 0.5)
ax.plot(t ,Es_l/Es_l[t_act_idx] ,color=color_left,lw=2,label=r'Es left')
ax.plot(t ,Es_r/Es_r[t_act_idx] ,color=color_right,lw=2,label=r'Es right')
if no_PA is False:
    ax.axvline(ymin=0, ymax=1, x=t[t_act_idx], lw=1, color='bright sky blue',label='Model switch and PA')
else:
    ax.axvline(ymin=0, ymax=1, x=t[t_act_idx], lw=1, color='gray', label='Model switch')

ax.format(xtickminor=False,ytickminor=False)

ax.set_ylabel(r'relative $Es$')
ax.set_xlabel(r'$t$ [s]')

# legend1 = ax.legend(handles=legend_elements,loc='best',ncols=1)
# legend2 =ax.legend(handles=additional_legend_elements,loc='upper left')
ax.legend(loc='upper right',ncols=1,frame=False)
# ax.add_artist(legend1)
# ax.set_title('Cell')
# ax.set_xlim(0,5)
ax.set_ylim(0.5,1.5)
# ax.set_ylim(-0.5,5)
format_axes(ax)
fig.savefig(save_directory+'Es.pdf',bbox_inches='tight')
fig.savefig(save_directory+'Es.svg')
np.savez('Es_model_switch.npz', Es_l=Es_l, Es_r=Es_r)  # N/m,sec

# Flow Velocity for a certain time frame
latexify_sf_thesis()
fig = pplt.figure(refaspect=1.61)
ax = fig.add_subplot(1,1,1)
for t_idx in range(len(t)):
    a = t[t_idx]/t[-1]
    if t_idx%2 ==0:
        if t[t_idx] <= act_t:
            ax.plot(x, vf[t_idx], color='gray', lw=0.6)
        else:
            ax.plot(x, vf[t_idx], color='C0', lw=0.2,alpha=a)

for t_idx in range(len(t)):
    if t_idx%2 ==0:
        if t[t_idx] >= act_t:
            if no_PA is False:
                ax.axvline(ymin=0, ymax=1, x=l0[t_idx], lw=0.2, color='bright sky blue')
# ax.plot(x ,vf[10] ,color='C0',lw=2,label=r'vf')


if model_switch is True:
    ax.axvline(ymin=0, ymax=1, x=l0[t_act_idx], lw=1, color='k', label=r'switch')

ax.format(xtickminor=False,ytickminor=False)

ax.set_ylabel(r'$v$ [um/h]')
ax.set_xlabel(r'$x$ [um]')

# legend1 = ax.legend(handles=legend_elements,loc='best',ncols=1)
# legend2 =ax.legend(handles=additional_legend_elements,loc='upper left')
ax.legend(loc='lower right',ncols=1,frame=False)
# ax.add_artist(legend1)
# ax.set_title('Cell')
# ax.set_xlim(0,5)
# ax.set_ylim(-0.1,0.3)
# ax.set_ylim(-0.5,5)
format_axes(ax)
fig.savefig(save_directory+'vf.pdf',bbox_inches='tight')
fig.savefig(save_directory+'vf.svg')


# Kymograph for certain positions ##########################################################################################################################
latexify_sf_thesis()
fig = pplt.figure(refaspect=1.61)
ax = fig.add_subplot(1,1,1)
# ax.plot(t, x[499]+u[:,499] ,color='C1',lw=2,label=r'u(pos)')
# ax.plot(t ,x[500]+u[:,500] ,color='gray',lw=2,label=r'')
ax.plot(t ,u[:,501] ,color='gray',lw=2,label=r'')
ax.plot(t ,u[:,499] ,color='gray',lw=2,label=r'')
# ax.plot(t ,x[502]+u[:,502] ,color='gray',lw=2,label=r'')
# ax.plot(t ,x[498]+u[:,498] ,color='gray',lw=2,label=r'')


ax.format(xtickminor=False,ytickminor=False)

ax.set_ylabel(r'$space$ [um]')
ax.set_xlabel(r'$time$ [s]')

# legend1 = ax.legend(handles=legend_elements,loc='best',ncols=1)
# legend2 =ax.legend(handles=additional_legend_elements,loc='upper left')
ax.legend(loc='lower right',ncols=1,frame=False)
# ax.add_artist(legend1)
# ax.set_title('Cell')
# ax.set_xlim(0,5)
# ax.set_ylim(-0.1,0.3)
# ax.set_ylim(-0.5,5)
format_axes(ax)
fig.savefig(save_directory+'kym.pdf',bbox_inches='tight')
fig.savefig(save_directory+'kym.svg')


