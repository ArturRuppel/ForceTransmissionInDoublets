import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import seaborn as sns


def interpolate_simulation_maps(directory, filename, x_key, y_key, data_key1, data_key2, x_start, x_end, y_start, y_end, pixelsize, no_frames):
    '''x_start, x_end, y_start and y_and are the coordinates of the 4 courners of the heatmap in micron and the pixelsize in micron per pixel
    determines how many pixel the stressmap will have after interpolation'''

    def read_FEM_simulation_maps(directory, filename, x_key, y_key, data_key1, data_key2, c):
        df = pd.read_csv(directory + filename + "%s.csv" % (c))
        x = df[x_key].values
        y = df[y_key].values
        data1 = df[data_key1].values
        data2 = df[data_key2].values

        return x, y, data1, data2

    grid_x, grid_y = np.mgrid[x_start:x_end:(x_end - x_start) / pixelsize * 1j, y_start:y_end:(y_end - y_start) / pixelsize * 1j]

    data1_all = np.zeros([grid_x.shape[0], grid_x.shape[1], no_frames])
    data2_all = np.zeros([grid_x.shape[0], grid_x.shape[1], no_frames])

    for frame in range(no_frames):
        x, y, data1, data2 = read_FEM_simulation_maps(directory, filename, x_key, y_key, data_key1, data_key2, frame)
        data1_all[:, :, frame] = griddata((x, y), data1, (grid_x, grid_y), method='cubic')
        data2_all[:, :, frame] = griddata((x, y), data2, (grid_x, grid_y), method='cubic')

    return data1_all, data2_all



if __name__ == "__main__":
    directory = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/_FEM_simulations/"
    filename = "test_"
    x_key = ":0"
    y_key = ":1"
    data_key1 = "Stress:0"
    data_key2 = "Stress:4"
    x_start = -22.5     # left most x-value in micron
    x_end = 22.5        # right most x-value in micron
    y_start = -22.5     # left most y-value in micron
    y_end = 22.5        # right most y-value in micron
    pixelsize = 0.864   # desired final pixelsize in micron per pixel
    no_frames = 60      # nomber of frames that the simulation contains

    sigma_xx, sigma_yy = interpolate_simulation_maps(directory, filename, x_key, y_key, data_key1, data_key2, x_start, x_end, y_start, y_end, pixelsize, no_frames)

# #%%
# df = pd.DataFrame.from_dict(np.array([x,y,sigma_yy]).T)
# df.columns = ['X_value','Y_value','Z_value']
# df['Z_value'] = pd.to_numeric(df['Z_value'])
# pivotted= df.pivot('Y_value','X_value','Z_value')
# sns.heatmap(pivotted,cmap='turbo')
