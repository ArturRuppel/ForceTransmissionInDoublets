import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import pickle
from skimage.morphology import closing, dilation, disk
from scipy.sparse import coo_matrix



def interpolate_simulation_maps(directory, filename, x_key, y_key, data_key1, data_key2, grid_x, grid_y, pixelsize):
    '''x_start, x_end, y_start and y_and are the coordinates of the 4 courners of the heatmap in micron and the pixelsize in micron per pixel
    determines how many pixel the stressmap will have after interpolation'''

    def read_FEM_simulation_maps(directory, filename, x_key, y_key, data_key1, data_key2):
        # df = pd.read_csv(directory + filename + "%s.csv" % (c))
        df = pd.read_csv(directory + filename)
        x = df[x_key].values
        y = df[y_key].values
        data1 = df[data_key1].values
        data2 = df[data_key2].values

        return y, x, data1, data2

    data1_all = np.zeros([grid_x.shape[0], grid_x.shape[1]])
    data2_all = np.zeros([grid_x.shape[0], grid_x.shape[1]])

    x, y, data1, data2 = read_FEM_simulation_maps(directory, filename, x_key, y_key, data_key1, data_key2)
    data1_all[:, :] = griddata((x, y), data1, (grid_x, grid_y), method='cubic')
    data2_all[:, :] = griddata((x, y), data2, (grid_x, grid_y), method='cubic')

    return data1_all, data2_all

def get_mask_from_points(directory, filename, x_key, y_key, grid_x):
    '''x_start, x_end, y_start and y_and are the coordinates of the 4 courners of the heatmap in micron and the pixelsize in micron per pixel
    determines how many pixel the stressmap will have after interpolation'''

    def read_FEM_simulation_maps(directory, filename, x_key, y_key):

        df = pd.read_csv(directory + filename)
        x = df[x_key].values
        y = df[y_key].values

        return y, x

    x, y = read_FEM_simulation_maps(directory, filename, x_key, y_key)
    # x_rounded = np.round(x * 10 + int(grid_x.shape[0]/2))
    # y_rounded = np.round(y * 10 + int(grid_x.shape[1]/2))
    x_rounded = np.round(x + int(grid_x.shape[0]/2))
    y_rounded = np.round(y + int(grid_x.shape[1]/2))
    data = np.ones(x.shape)
    mask = coo_matrix((data, (x_rounded, y_rounded)), shape=(grid_x.shape)).toarray()
    mask = mask > 0
    footprint = disk(3)
    closed = closing(mask, footprint)
    dilated = dilation(closed, footprint)
    # plt.imshow(dilated)
    # plt.show()
    return disk

def analyze_FEM_dataf(FEM_data):
    for key in FEM_data:
        FEM_data[key]["sigma_avg_norm"] = (FEM_data[key]["sigma_xx"] + FEM_data[key]["sigma_yy"]) / 2
        FEM_data[key]["sigma_avg_norm_x_profile"] = np.nanmean(FEM_data[key]["sigma_avg_norm"], axis=0)
        FEM_data[key]["sigma_normal_x_profile_increase"] = \
            FEM_data[key]["sigma_avg_norm_x_profile"][:, 32] - FEM_data[key]["sigma_avg_norm_x_profile"][:, 20]

    return FEM_data

if __name__ == "__main__":
    directory = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/_FEM_simulations/doublets_warped/"
    filename = "stress_feedbacks_warped_"

    x_key = "Points:0"
    y_key = "Points:1"
    FEM_data = {}   # initialize dictionary
    x_start = -22.5  # left most x-value in micron
    x_end = 22.5  # right most x-value in micron
    y_start = -22.5  # left most y-value in micron
    y_end = 22.5  # right most y-value in micron
    pixelsize = 0.864  # desired final pixelsize in micron per pixel
    no_frames = 60  # nomber of frames
    grid_x, grid_y = np.mgrid[x_start:x_end:(x_end - x_start) / pixelsize * 1j, y_start:y_end:(y_end - y_start) / pixelsize * 1j]

    feedbacks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for feedback in feedbacks:
        print("Reading data from feedback: " + str(feedback))
        temp_dict = {}
        sigma_xx = np.zeros([grid_x.shape[0], grid_x.shape[1], no_frames])
        sigma_yy = np.zeros([grid_x.shape[0], grid_x.shape[1], no_frames])
        data_key1 = "Stress_"+str(feedback)+":0"
        data_key2 = "Stress_"+str(feedback)+":4"

        for frame in range(no_frames):
            filename_full = filename + str(frame) + ".csv"
            mask = get_mask_from_points(directory, filename_full, x_key, y_key, grid_x)
            sigma_xx_temp, sigma_yy_temp, = interpolate_simulation_maps(directory, filename_full, x_key, y_key,
                                                                                       data_key1, data_key2, grid_x, grid_y, pixelsize)
            # sigma_xx_temp[~mask] = 'nan'
            # sigma_yy_temp[~mask] = 'nan'
            sigma_xx[:, :, frame] = sigma_xx_temp
            sigma_yy[:, :, frame] = sigma_yy_temp

        temp_dict["sigma_xx"] = sigma_xx
        temp_dict["sigma_yy"] = sigma_yy
        FEM_data["feedback" + str(feedback)] = temp_dict

    FEM_data = analyze_FEM_dataf(FEM_data)

    with open("C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/_FEM_simulations/FEM_doublets.dat", 'wb') as outfile:
        pickle.dump(FEM_data, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    # sigma_avg_norm_diff = (sigma_xx_as + sigma_yy_as) / 2 - (sigma_xx_bs + sigma_yy_bs) / 2

# #%%
# df = pd.DataFrame.from_dict(np.array([x,y,sigma_yy]).T)
# df.columns = ['X_value','Y_value','Z_value']
# df['Z_value'] = pd.to_numeric(df['Z_value'])
# pivotted= df.pivot('Y_value','X_value','Z_value')
# sns.heatmap(pivotted,cmap='turbo')