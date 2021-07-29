# %% import packages
import pickle
import numpy as np
import pandas as pd
import scipy.stats as st

#%% load data
folder = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/"

AR1to1d_fullstim_long = pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_long.dat", "rb"))
AR1to1s_fullstim_long = pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_long.dat", "rb"))
AR1to1d_fullstim_short = pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_short.dat", "rb"))
AR1to1s_fullstim_short = pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_short.dat", "rb"))
AR1to2d_halfstim = pickle.load(open(folder + "analysed_data/AR1to2d_halfstim.dat", "rb"))
AR1to1d_halfstim = pickle.load(open(folder + "analysed_data/AR1to1d_halfstim.dat", "rb"))
AR1to1s_halfstim = pickle.load(open(folder + "analysed_data/AR1to1s_halfstim.dat", "rb"))
AR2to1d_halfstim = pickle.load(open(folder + "analysed_data/AR2to1d_halfstim.dat", "rb"))

# load contour model analysis data
AR1to1d_fullstim_long_CM = pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_long_CM.dat", "rb"))
AR1to1d_fullstim_short_CM = pickle.load(open(folder + "analysed_data/AR1to1d_fullstim_short_CM.dat", "rb"))
AR1to1s_fullstim_long_CM = pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_long_CM.dat", "rb"))
AR1to1s_fullstim_short_CM = pickle.load(open(folder + "analysed_data/AR1to1s_fullstim_short_CM.dat", "rb"))
AR1to2d_halfstim_CM = pickle.load(open(folder + "analysed_data/AR1to2d_halfstim_CM.dat", "rb"))
AR1to1d_halfstim_CM = pickle.load(open(folder + "analysed_data/AR1to1d_halfstim_CM.dat", "rb"))
AR1to1s_halfstim_CM = pickle.load(open(folder + "analysed_data/AR1to1s_halfstim_CM.dat", "rb"))
AR2to1d_halfstim_CM = pickle.load(open(folder + "analysed_data/AR2to1d_halfstim_CM.dat", "rb"))

#%%
mean_stress_and_Es = {'1to1d_fullstim': {'Es': np.nanmean(AR1to1d_fullstim_long["TFM_data"]["Es"], axis=1),
                                         'sigma_xx': np.nanmean(AR1to1d_fullstim_long["MSM_data"]["sigma_xx_average"], axis=1),
                                         'sigma_xx_left': np.nanmean(AR1to1d_fullstim_long["MSM_data"]["sigma_xx_left_average"], axis=1),
                                         'sigma_xx_right': np.nanmean(AR1to1d_fullstim_long["MSM_data"]["sigma_xx_right_average"], axis=1),
                                         'sigma_yy': np.nanmean(AR1to1d_fullstim_long["MSM_data"]["sigma_yy_average"], axis=1),
                                         'sigma_yy_left': np.nanmean(AR1to1d_fullstim_long["MSM_data"]["sigma_yy_left_average"], axis=1),
                                         'sigma_yy_right': np.nanmean(AR1to1d_fullstim_long["MSM_data"]["sigma_yy_right_average"], axis=1)},
                      '1to1d_halfstim': {'Es': np.nanmean(AR1to1d_halfstim["TFM_data"]["Es"], axis=1),
                                         'sigma_xx': np.nanmean(AR1to1d_halfstim["MSM_data"]["sigma_xx_average"], axis=1),
                                         'sigma_xx_left': np.nanmean(AR1to1d_halfstim["MSM_data"]["sigma_xx_left_average"], axis=1),
                                         'sigma_xx_right': np.nanmean(AR1to1d_halfstim["MSM_data"]["sigma_xx_right_average"], axis=1),
                                         'sigma_yy': np.nanmean(AR1to1d_halfstim["MSM_data"]["sigma_yy_average"], axis=1),
                                         'sigma_yy_left': np.nanmean(AR1to1d_halfstim["MSM_data"]["sigma_yy_left_average"], axis=1),
                                         'sigma_yy_right': np.nanmean(AR1to1d_halfstim["MSM_data"]["sigma_yy_right_average"], axis=1)},
                      '1to1s_fullstim': {'Es': np.nanmean(AR1to1s_fullstim_long["TFM_data"]["Es"], axis=1),
                                         'sigma_xx': np.nanmean(AR1to1s_fullstim_long["MSM_data"]["sigma_xx_average"], axis=1),
                                         'sigma_xx_left': np.nanmean(AR1to1s_fullstim_long["MSM_data"]["sigma_xx_left_average"], axis=1),
                                         'sigma_xx_right': np.nanmean(AR1to1s_fullstim_long["MSM_data"]["sigma_xx_right_average"], axis=1),
                                         'sigma_yy': np.nanmean(AR1to1s_fullstim_long["MSM_data"]["sigma_yy_average"], axis=1),
                                         'sigma_yy_left': np.nanmean(AR1to1s_fullstim_long["MSM_data"]["sigma_yy_left_average"], axis=1),
                                         'sigma_yy_right': np.nanmean(AR1to1s_fullstim_long["MSM_data"]["sigma_yy_right_average"], axis=1)},
                      '1to1s_halfstim': {'Es': np.nanmean(AR1to1s_halfstim["TFM_data"]["Es"], axis=1),
                                         'sigma_xx': np.nanmean(AR1to1s_halfstim["MSM_data"]["sigma_xx_average"], axis=1),
                                         'sigma_xx_left': np.nanmean(AR1to1s_halfstim["MSM_data"]["sigma_xx_left_average"], axis=1),
                                         'sigma_xx_right': np.nanmean(AR1to1s_halfstim["MSM_data"]["sigma_xx_right_average"], axis=1),
                                         'sigma_yy': np.nanmean(AR1to1s_halfstim["MSM_data"]["sigma_yy_average"], axis=1),
                                         'sigma_yy_left': np.nanmean(AR1to1s_halfstim["MSM_data"]["sigma_yy_left_average"], axis=1),
                                         'sigma_yy_right': np.nanmean(AR1to1s_halfstim["MSM_data"]["sigma_yy_right_average"], axis=1)},
                      '1to2d_halfstim': {'Es': np.nanmean(AR1to2d_halfstim["TFM_data"]["Es"], axis=1),
                                         'sigma_xx': np.nanmean(AR1to2d_halfstim["MSM_data"]["sigma_xx_average"], axis=1),
                                         'sigma_xx_left': np.nanmean(AR1to2d_halfstim["MSM_data"]["sigma_xx_left_average"], axis=1),
                                         'sigma_xx_right': np.nanmean(AR1to2d_halfstim["MSM_data"]["sigma_xx_right_average"], axis=1),
                                         'sigma_yy': np.nanmean(AR1to2d_halfstim["MSM_data"]["sigma_yy_average"], axis=1),
                                         'sigma_yy_left': np.nanmean(AR1to2d_halfstim["MSM_data"]["sigma_yy_left_average"], axis=1),
                                         'sigma_yy_right': np.nanmean(AR1to2d_halfstim["MSM_data"]["sigma_yy_right_average"], axis=1)},
                      '2to1d_halfstim': {'Es': np.nanmean(AR2to1d_halfstim["TFM_data"]["Es"], axis=1),
                                         'sigma_xx': np.nanmean(AR2to1d_halfstim["MSM_data"]["sigma_xx_average"], axis=1),
                                         'sigma_xx_left': np.nanmean(AR2to1d_halfstim["MSM_data"]["sigma_xx_left_average"], axis=1),
                                         'sigma_xx_right': np.nanmean(AR2to1d_halfstim["MSM_data"]["sigma_xx_right_average"], axis=1),
                                         'sigma_yy': np.nanmean(AR2to1d_halfstim["MSM_data"]["sigma_yy_average"], axis=1),
                                         'sigma_yy_left': np.nanmean(AR2to1d_halfstim["MSM_data"]["sigma_yy_left_average"], axis=1),
                                         'sigma_yy_right': np.nanmean(AR2to1d_halfstim["MSM_data"]["sigma_yy_right_average"], axis=1)},
                      }

with open(folder + 'analysed_data/mean_stress_and_Es.dat', 'wb') as outfile:
    pickle.dump(mean_stress_and_Es, outfile, protocol=pickle.HIGHEST_PROTOCOL)

#%% create dataframe for halfstim data
# initialize empty dictionaries
concatenated_data_1to2d = {}
concatenated_data_1to1d = {}
concatenated_data_1to1s = {}
concatenated_data_2to1d = {}
concatenated_data = {}

# select keys which should end up in the dataframe
keys_to_load = {'a_baseline', 'b_baseline', 'line tension baseline [nN]', 'sigma_x_baseline', 'sigma_y_baseline',
                'RSI_xx_left', 'RSI_xx_right', 'RSI_yy_left', 'RSI_yy_right'}
# loop over all keys for contour model analysis data
for key1 in AR1to1d_halfstim_CM:
    for key2 in AR1to1d_halfstim_CM[key1]:    # keys are the same for all dictionaries so I'm just taking one example here
        if key2 in keys_to_load:  # only 1D data can be stored in the data frame
            # concatenate values from different experiments
            concatenated_data_1to2d[key2] = AR1to2d_halfstim_CM[key1][key2]
            concatenated_data_1to1d[key2] = AR1to1d_halfstim_CM[key1][key2]
            concatenated_data_1to1s[key2] = AR1to1s_halfstim_CM[key1][key2]
            concatenated_data_2to1d[key2] = AR2to1d_halfstim_CM[key1][key2]

            # concatenate doublet and singlet data to create pandas dataframe
            concatenated_data[key2] = np.concatenate((concatenated_data_1to2d[key2], concatenated_data_1to1d[key2],
                                                      concatenated_data_1to1s[key2], concatenated_data_2to1d[key2]))

# select keys to be imported
# loop over all keys
for key1 in AR1to1d_halfstim:
    for key2 in AR1to1d_halfstim[key1]:
        if key2 in keys_to_load:  # only 1D data can be stored in the data frame
            # concatenate values from different experiments
            concatenated_data_1to2d[key2] = AR1to2d_halfstim[key1][key2]
            concatenated_data_1to1d[key2] = AR1to1d_halfstim[key1][key2]
            concatenated_data_1to1s[key2] = AR1to1s_halfstim[key1][key2]
            concatenated_data_2to1d[key2] = AR2to1d_halfstim[key1][key2]

            concatenated_data[key2] = np.concatenate((concatenated_data_1to2d[key2], concatenated_data_1to1d[key2],
                                                      concatenated_data_1to1s[key2], concatenated_data_2to1d[key2]))
key2 = 'RSI_yy_right'
# get number of elements for both condition
n_1to2d = concatenated_data_1to2d[key2].shape[0]
n_1to1d = concatenated_data_1to1d[key2].shape[0]
n_1to1s = concatenated_data_1to1s[key2].shape[0]
n_2to1d = concatenated_data_2to1d[key2].shape[0]

# create a list of keys with the same dimensions as the data
keys1to2d = ['AR1to2d' for i in range(n_1to2d)]
keys1to1d = ['AR1to1d' for i in range(n_1to1d)]
keys1to1s = ['AR1to1s' for i in range(n_1to1s)]
keys2to1d = ['AR2to1d' for i in range(n_2to1d)]
keys = np.concatenate((keys1to2d, keys1to1d, keys1to1s, keys2to1d))

# add keys to dictionary with concatenated data
concatenated_data['keys'] = keys

# create dataframe
df = pd.DataFrame(concatenated_data)

# rename columns
df.rename(columns={'line tension baseline [nN]': 'line tension [nN]', 'sigma_x_baseline': 'sigma_x_CM [mN/m]', 'sigma_y_baseline': 'sigma_y_CM [mN/m]',
                   'a_baseline': 'a [um]', 'b_baseline': 'b [um]'}, inplace=True)


df.to_csv(folder + 'analysed_data/halfstim_data.csv', index=False)
df_halfstim = df
#%% create dataframe for fullstim data
# initialize empty dictionaries
concatenated_data_1to1d = {}
concatenated_data_1to1s = {}
concatenated_data = {}

# select keys which should end up in the dataframe
keys_to_load = {'a_baseline', 'b_baseline', 'line tension baseline [nN]', 'sigma_x_baseline', 'sigma_y_baseline', 'RSI_xx', 'RSI_yy'}
# loop over all keys for contour model analysis data
for key1 in AR1to1d_fullstim_long_CM:
    for key2 in AR1to1d_fullstim_long_CM[key1]:    # keys are the same for all dictionaries so I'm just taking one example here
        if key2 in keys_to_load:  # only 1D data can be stored in the data frame
            # concatenate values from different experiments
            concatenated_data_1to1d[key2] = AR1to1d_fullstim_long_CM[key1][key2]
            concatenated_data_1to1s[key2] = AR1to1s_fullstim_long_CM[key1][key2]

            # concatenate doublet and singlet data to create pandas dataframe
            concatenated_data[key2] = np.concatenate((concatenated_data_1to1d[key2], concatenated_data_1to1s[key2]))

# select keys to be imported
# loop over all keys
for key1 in AR1to1d_fullstim_long:
    for key2 in AR1to1d_fullstim_long[key1]:
        if key2 in keys_to_load:  # only 1D data can be stored in the data frame
            # concatenate values from different experiments
            concatenated_data_1to1d[key2] = AR1to1d_fullstim_long[key1][key2]
            concatenated_data_1to1s[key2] = AR1to1s_fullstim_long[key1][key2]

            # concatenate doublet and singlet data to create pandas dataframe
            concatenated_data[key2] = np.concatenate((concatenated_data_1to1d[key2], concatenated_data_1to1s[key2]))

key2 = 'RSI_yy'
# get number of elements for both condition
n_1to1d = concatenated_data_1to1d[key2].shape[0]
n_1to1s = concatenated_data_1to1s[key2].shape[0]

# create a list of keys with the same dimensions as the data
keys1to1d = ['AR1to1d' for i in range(n_1to1d)]
keys1to1s = ['AR1to1s' for i in range(n_1to1s)]
keys = np.concatenate((keys1to1d, keys1to1s))

# add keys to dictionary with concatenated data
concatenated_data['keys'] = keys

# create dataframe
df = pd.DataFrame(concatenated_data)

# rename columns
df.rename(columns={'line tension baseline [nN]': 'line tension [nN]', 'sigma_x_baseline': 'sigma_x_CM [mN/m]', 'sigma_y_baseline': 'sigma_y_CM [mN/m]',
                   'a_baseline': 'a [um]', 'b_baseline': 'b [um]'}, inplace=True)


df.to_csv(folder + 'analysed_data/fullstim_data.csv', index=False)
df_fullstim = df

# %% calculate mean and CI
def calculate_and_print_mean_and_CI(df, quantity):
    stats = df.groupby(['keys'])[quantity].agg(['mean', 'count', 'std'])
    confidence = 0.95
    CI = []
    quantities = []

    for i in stats.index:
        mean, count, std = stats.loc[i]
        CI.append(st.t.ppf((1 + confidence) / 2., count-1) * std / np.sqrt(count))      # the whole confidence interval is mean +- CI_error
        quantities.append(quantity)
    stats['CI'] = CI
    stats['quantity'] = quantities

    return stats


# caluclate mean and CI for fullstim data
list_of_quantities = {'a [um]', 'b [um]', 'line tension [nN]', 'sigma_x_CM [mN/m]', 'sigma_y_CM [mN/m]', 'RSI_xx', 'RSI_yy'}
stats = []
for quantity in list_of_quantities:
    stats.append(calculate_and_print_mean_and_CI(df_fullstim, quantity))
stats = pd.concat(stats)
stats.to_csv(folder + 'analysed_data/stats_fullstim.csv')


# caluclate mean and CI for halfstim data
list_of_quantities = {'a [um]', 'b [um]', 'line tension [nN]', 'sigma_x_CM [mN/m]', 'sigma_y_CM [mN/m]', 'RSI_xx_left', 'RSI_xx_right', 'RSI_yy_left', 'RSI_yy_right'}
stats = []
for quantity in list_of_quantities:
    stats.append(calculate_and_print_mean_and_CI(df_halfstim, quantity))
stats = pd.concat(stats)
stats.to_csv(folder + 'analysed_data/stats_halfstim.csv')


