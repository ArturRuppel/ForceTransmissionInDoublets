import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statannot
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


def make_box_and_swarmplots_with_test(x, y, df, ax, ymin, ymax, yticks, stat_annotation_offset, box_pairs, xticklabels, ylabel, title, colors,
                                linewidth_bp=0.7, width_bp=0.3, dotsize=1.8, linewidth_sw=0.3, alpha_sw=1, alpha_bp=0.8, ylabeloffset=1,
                                titleoffset=3.5, test='Mann-Whitney'):
    sns.set_palette(sns.color_palette(colors))  # sets colors
    # create box- and swarmplots
    sns.swarmplot(x=x, y=y, data=df, ax=ax, alpha=alpha_sw, linewidth=linewidth_sw, zorder=0, size=dotsize)
    bp = sns.boxplot(x=x, y=y, data=df, ax=ax, linewidth=linewidth_bp, notch=True, showfliers=False, width=width_bp, showmeans=True,
                     meanprops={"marker": "o",
                                "markerfacecolor": "white",
                                "markeredgecolor": "black",
                                "markersize": "3", "markeredgewidth": "0.5"})

    statannot.add_stat_annotation(bp, data=df, x=x, y=y, box_pairs=box_pairs,
                                  line_offset_to_box=stat_annotation_offset, test=test, comparisons_correction=None, text_format='star', loc='inside', verbose=3)

    # make boxplots transparent
    for patch in bp.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, alpha_bp))

    plt.setp(bp.artists, edgecolor='k')
    plt.setp(bp.lines, color='k')

    # set labels
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel(xlabel=None)
    ax.set_ylabel(ylabel=ylabel, labelpad=ylabeloffset)
    ax.set_title(label=title, pad=titleoffset)

    # set yaxis ticks
    ax.yaxis.set_ticks(yticks)

    # provide info on tick parameters
    ax.minorticks_on()
    ax.tick_params(direction='in', which='minor', length=3, bottom=False, top=False, left=True, right=True)
    ax.tick_params(direction='in', which='major', length=6, bottom=False, top=False, left=True, right=True)

    # set limits
    ax.set_ylim(ymin=ymin)
    ax.set_ylim(ymax=ymax)



def make_box_and_swarmplots(x, y, df, ax, ymin, ymax, yticks, xticklabels, ylabel, title, colors, linewidth_bp=0.7, width_bp=0.5,
                                 dotsize=1.3, linewidth_sw=0.3, alpha_sw=1, alpha_bp=0.8, ylabeloffset=1, titleoffset=3.5):
    sns.set_palette(sns.color_palette(colors))  # sets colors
    # create box- and swarmplots
    sns.swarmplot(x=x, y=y, data=df, ax=ax, alpha=alpha_sw, linewidth=linewidth_sw, zorder=0, size=dotsize)
    bp = sns.boxplot(x=x, y=y, data=df, ax=ax, linewidth=linewidth_bp, notch=True, showfliers=False, width=width_bp, showmeans=True,
                     meanprops={"marker": "o",
                                "markerfacecolor": "white",
                                "markeredgecolor": "black",
                                "markersize": "3", "markeredgewidth": "0.3"})

    # make boxplots transparent
    for patch in bp.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, alpha_bp))

    plt.setp(bp.artists, edgecolor='k')
    plt.setp(bp.lines, color='k')

    # set labels
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel(xlabel=None)
    ax.set_ylabel(ylabel=ylabel, labelpad=ylabeloffset)
    ax.set_title(label=title, pad=titleoffset)

    # set yaxis ticks
    ax.yaxis.set_ticks(yticks)

    # provide info on tick parameters
    ax.minorticks_on()
    ax.tick_params(direction='in', which='minor', length=3, bottom=False, top=False, left=True, right=True)
    ax.tick_params(direction='in', which='major', length=6, bottom=False, top=False, left=True, right=True)

    # set limits
    ax.set_ylim(ymin=ymin)
    ax.set_ylim(ymax=ymax)

    return bp

def plot_one_value_over_time(x, y, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors,
                             titleoffset=3.5, optolinewidth=0.1, xlabeloffset=1, ylabeloffset=1, xmax=False):
    y_mean = np.nanmean(y, axis=1)
    y_std = np.nanstd(y, axis=1)
    y_sem = y_std / np.sqrt(np.shape(y)[1])
    # create box- and swarmplots
    ax.errorbar(x, y_mean, yerr=y_sem, mfc='w', color=colors, marker='o', ms=2, linewidth=0.5, ls='none',
                markeredgewidth=0.5)

    # set labels
    ax.set_xlabel(xlabel=xlabel, labelpad=xlabeloffset)
    ax.set_ylabel(ylabel=ylabel, labelpad=ylabeloffset)
    ax.set_title(label=title, pad=titleoffset)

    # set ticks
    ax.xaxis.set_ticks(xticks)
    ax.yaxis.set_ticks(yticks)

    # provide info on tick parameters
    ax.minorticks_on()
    ax.tick_params(direction='in', which='minor', length=3, bottom=True, top=False, left=True, right=True)
    ax.tick_params(direction='in', which='major', length=6, bottom=True, top=False, left=True, right=True)

    # set limits
    ax.set_ylim(ymin=ymin, ymax=ymax)
    ax.set_xlim(xmin=min(x))
    if xmax == False:
        ax.set_xlim(xmax=max(x) + 1e-1)
    else:
        ax.set_xlim(xmax=xmax + 1e-1)
    try:
        # add anotations for opto pulses
        for i in np.arange(10):
            ax.axline((20 + i, ymin), (20 + i, ymax), linewidth=optolinewidth, color="cyan")
    except:
        return


def plot_two_values_over_time(x, y1, y2, xticks, yticks, ymin, ymax, xlabel, ylabel, title, ax, colors,
                              titleoffset=3.5, optolinewidth=0.1, xlabeloffset=1, ylabeloffset=1):
    y1_mean = np.nanmean(y1, axis=1)
    y1_std = np.nanstd(y1, axis=1)
    y1_sem = y1_std / np.sqrt(np.shape(y1)[1])

    y2_mean = np.nanmean(y2, axis=1)
    y2_std = np.nanstd(y2, axis=1)
    y2_sem = y2_std / np.sqrt(np.shape(y2)[1])

    ax.errorbar(x, y1_mean, yerr=y1_sem, mfc='w', color=colors[0], marker='o', ms=2, linewidth=0.5, ls='none',
                markeredgewidth=0.5)

    ax.errorbar(x, y2_mean, yerr=y2_sem, mfc='w', color=colors[1], marker='o', ms=2, linewidth=0.5, ls='none',
                markeredgewidth=0.5)
    # set labels
    ax.set_xlabel(xlabel=xlabel, labelpad=xlabeloffset)
    ax.set_ylabel(ylabel=ylabel, labelpad=ylabeloffset)
    ax.set_title(label=title, pad=titleoffset)

    # add anotations for opto pulses
    for i in np.arange(10):
        ax.axline((20 + i, ymin), (20 + i, ymax), linewidth=optolinewidth, color="cyan")

    # set ticks
    ax.xaxis.set_ticks(xticks)
    ax.yaxis.set_ticks(yticks)

    # provide info on tick parameters
    ax.minorticks_on()
    ax.tick_params(direction='in', which='minor', length=3, bottom=True, top=False, left=True, right=True)
    ax.tick_params(direction='in', which='major', length=6, bottom=True, top=False, left=True, right=True)

    # set limits
    ax.set_ylim(ymin=ymin, ymax=ymax)


def plot_actin_image_forces_ellipses_tracking_tangents(actin_image, x, y, Tx, Ty, T, a_top, b_top, a_bottom, b_bottom,
                                                       tx_topleft, ty_topleft, tx_topright, ty_topright, tx_bottomleft, ty_bottomleft,
                                                       tx_bottomright, ty_bottomright,
                                                       xc_top, yc_top, xc_bottom, yc_bottom, x_tracking_top, y_tracking_top,
                                                       x_tracking_bottom, y_tracking_bottom,
                                                       xc_topleft, yc_topleft, xc_topright, yc_topright, xc_bottomleft, yc_bottomleft,
                                                       xc_bottomright, yc_bottomright, ax, colors, extent):
    norm = mpl.colors.Normalize()
    norm.autoscale([0, 2])
    colormap = mpl.cm.turbo
    sm = mpl.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])

    # plot actin image
    ax.imshow(actin_image, cmap=plt.get_cmap("Greys"), interpolation="bilinear", extent=extent, aspect=(1))

    # plot forces
    ax.quiver(x, y, Tx, Ty, angles='xy', scale_units='xy', scale=0.2, color=colormap(norm(T)))

    # plot ellipses
    t = np.linspace(0, 2 * np.pi, 100)
    ax.plot(xc_top + a_top * np.cos(t), yc_top + b_top * np.sin(t), color=colors[0], linewidth=2)
    t = np.linspace(0, 2 * np.pi, 100)
    ax.plot(xc_bottom + a_bottom * np.cos(t), yc_bottom + b_bottom * np.sin(t), color=colors[0], linewidth=2)

    # plot tracking data
    ax.plot(x_tracking_top[::2], y_tracking_top[::2],
            color=colors[1], marker='o', markerfacecolor='none', markersize=4, markeredgewidth=1, linestyle='none')
    ax.plot(x_tracking_bottom[::2], y_tracking_bottom[::2],
            color=colors[1], marker='o', markerfacecolor='none', markersize=4, markeredgewidth=1, linestyle='none')

    # plot tangents
    ax.plot([xc_topleft, xc_topleft + 150 * tx_topleft], [yc_topleft, yc_topleft + 150 * ty_topleft],
            color='white', linewidth=2, linestyle=':')
    ax.plot([xc_topright, xc_topright + 150 * tx_topright], [yc_topright, yc_topright + 150 * ty_topright],
            color='white', linewidth=2, linestyle=':')
    ax.plot([xc_bottomleft, xc_bottomleft + 150 * tx_bottomleft], [yc_bottomleft, yc_bottomleft + 150 * ty_bottomleft],
            color='white', linewidth=2, linestyle=':')
    ax.plot([xc_bottomright, xc_bottomright + 150 * tx_bottomright], [yc_bottomright, yc_bottomright + 150 * ty_bottomright],
            color='white', linewidth=2, linestyle=':')

    ax.set_xlim([-0.1 * extent[1], 1.1 * extent[1]])
    ax.set_ylim([-0.1 * extent[3], 1.1 * extent[3]])

    ax.axis('off')

    return sm


def make_correlationplotsplots(x, y, hue, df, ax, xmin, xmax, ymin, ymax, xticks, yticks, xlabel, ylabel, colors,
                               dotsize=1.8, linewidth_sw=0.3, alpha_sw=1, ylabeloffset=1, xlabeloffset=0, titleoffset=3.5):
    # set colors
    sns.set_palette(sns.color_palette(colors))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, style=hue, ax=ax, alpha=alpha_sw, linewidth=linewidth_sw, size=dotsize)
    sns.regplot(data=df, x=x, y=y, scatter=False, ax=ax, color='black')

    # set labels
    ax.set_xlabel(xlabel=xlabel, labelpad=xlabeloffset)
    ax.set_ylabel(ylabel=ylabel, labelpad=ylabeloffset)

    # set limits
    ax.set_xlim(xmin=xmin, xmax=xmax)
    ax.set_ylim(ymin=ymin, ymax=ymax)

    # Define where you want ticks
    plt.sca(ax)
    plt.xticks(xticks)
    plt.yticks(yticks)

    # remove legend
    ax.get_legend().remove()

    # provide info on tick parameters
    plt.minorticks_on()
    plt.tick_params(direction='in', which='minor', length=3, bottom=True, top=True, left=True, right=True)
    plt.tick_params(direction='in', which='major', length=6, bottom=True, top=True, left=True, right=True)

    corr, p = pearsonr(df[x], df[y])

    corr = np.round(corr, decimals=3)
    # p = np.round(p,decimals=6)

    return corr, p


def create_baseline_filter(data, threshold):
    # initialize variables
    if np.ndim(data) < 3:
        data = data[..., np.newaxis]
    noVariables = np.shape(data)[2]
    t_end = np.shape(data)[0]
    baselinefilter_all = []
    # for each vector in data, find a linear regression and compare the slope to the threshold value. store result of this comparison in an array
    for i in range(noVariables):
        t = np.arange(t_end)
        model = LinearRegression().fit(t.reshape((-1, 1)), data[:, :, i])
        baselinefilter = np.absolute(model.coef_) < threshold
        baselinefilter_all.append(baselinefilter)

    # all vectors are combined to one through elementwise logical AND operation
    return np.all(baselinefilter_all, axis=0).reshape(-1)


def create_opto_increase_filter(data, threshold):
    opto_increase_filter = data > threshold

    if data.ndim > 1:
        # all vectors are combined to one through elementwise logical AND operation
        return np.all(opto_increase_filter, axis=1).reshape(-1)
    else:
        return opto_increase_filter


def apply_filter(data, filter):
    for key in data:
        shape = data[key].shape

        # find the new number of cells to find the new shape of data after filtering
        new_N = np.sum(filter)

        # to filter data of different dimensions, we first have to copy the filter vector into an array of the same shape as the data. We also create a variable with the new shape of the data
        if data[key].ndim == 1:
            filter_resized = filter
            newshape = [new_N]
            data[key] = data[key][filter_resized].reshape(newshape)
        elif data[key].ndim == 2:
            filter_resized = np.expand_dims(filter, axis=0).repeat(shape[0], 0)
            newshape = [shape[0], new_N]
            data[key] = data[key][filter_resized].reshape(newshape)
        elif data[key].ndim == 3:
            filter_resized = np.expand_dims(filter, axis=(0, 1)).repeat(shape[0], 0).repeat(shape[1], 1)
            newshape = [shape[0], shape[1], new_N]
            data[key] = data[key][filter_resized].reshape(newshape)
        elif data[key].ndim == 4:
            filter_resized = np.expand_dims(filter, axis=(0, 1, 2)).repeat(shape[0], 0).repeat(shape[1], 1).repeat(shape[2],
                                                                                                                                   2)
            newshape = [shape[0], shape[1], shape[2], new_N]
            data[key] = data[key][filter_resized].reshape(newshape)
        else:
            print('Nothing filtered, shape of array not supported')

    return data
