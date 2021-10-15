import pickle
import numpy as np
import pandas as pd
from scipy.interpolate import griddata, Rbf
from scipy.spatial import Delaunay
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def toliste(a):
    liste = []
    for i in range(len(a)):
        liste.append(list(a[i]))
    return liste


def tolistoftuples(a):
    liste = []
    for i in range(len(a)):
        liste.append(tuple(a[i]))
    return liste


def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges


def find_edges_with(i, edge_set):
    i_first = [j for (x, j) in edge_set if x == i]
    i_second = [j for (j, x) in edge_set if x == i]
    return i_first, i_second


def stitch_boundaries(edges):
    edge_set = edges.copy()
    boundary_lst = []
    while len(edge_set) > 0:
        boundary = []
        edge0 = edge_set.pop()
        boundary.append(edge0)
        last_edge = edge0
        while len(edge_set) > 0:
            i, j = last_edge
            j_first, j_second = find_edges_with(j, edge_set)
            if j_first:
                edge_set.remove((j, j_first[0]))
                edge_with_j = (j, j_first[0])
                boundary.append(edge_with_j)
                last_edge = edge_with_j
            elif j_second:
                edge_set.remove((j_second[0], j))
                edge_with_j = (j, j_second[0])  # flip edge rep
                boundary.append(edge_with_j)
                last_edge = edge_with_j

            if edge0[0] == last_edge[1]:
                break

        boundary_lst.append(boundary)
    return boundary_lst


def return_points(directory, filename, x_key, y_key):
    # df = pd.read_csv(directory + filename + "%s.csv" % (c))
    df = pd.read_csv(directory + filename)
    x = df[x_key].values
    y = df[y_key].values

    return y, x


def interpolate_simulation_maps(directory, filename, x_key, y_key, data_key1, data_key2, grid_x, grid_y):
    '''x_start, x_end, y_start and y_and are the coordinates of the 4 courners of the heatmap in micron and the pixelsize in micron per pixel
    determines how many pixel the stressmap will have after interpolation'''

    def read_FEM_simulation_maps(directory, filename, x_key, y_key, data_key1, data_key2):
        df = pd.read_csv(directory + filename)
        x = df[x_key].values
        y = df[y_key].values
        data1 = df[data_key1].values
        data2 = df[data_key2].values

        return y, x, data1, data2

    data1_all = np.zeros([grid_x.shape[0], grid_x.shape[1]])
    data2_all = np.zeros([grid_x.shape[0], grid_x.shape[1]])

    x, y, data1, data2 = read_FEM_simulation_maps(directory, filename, x_key, y_key, data_key1, data_key2)
    data1_all[:, :] = griddata((x, y), data1, (grid_x, grid_y), method='linear')
    data2_all[:, :] = griddata((x, y), data2, (grid_x, grid_y), method='linear')

    return data1_all, data2_all


def interpolate_simulation_maps_rbf(directory, filename, x, y, data_key1, data_key2, grid_x, grid_y):
    '''x_start, x_end, y_start and y_and are the coordinates of the 4 courners of the heatmap in micron and the pixelsize in micron per pixel
    determines how many pixel the stressmap will have after interpolation'''

    def read_FEM_simulation_maps(directory, filename, data_key1, data_key2):
        # df = pd.read_csv(directory + filename + "%s.csv" % (c))
        df = pd.read_csv(directory + filename)
        data1 = df[data_key1].values
        data2 = df[data_key2].values

        return data1, data2

    data1, data2 = read_FEM_simulation_maps(directory, filename, data_key1, data_key2)
    rbf1 = Rbf(x, y, data1, function='gaussian')
    rbf2 = Rbf(x, y, data2, function='gaussian')
    data1_all = rbf1(grid_x, grid_y)
    data2_all = rbf2(grid_x, grid_y)

    return data1_all, data2_all


def analyze_FEM_data(FEM_data, pixelsize):
    for key in FEM_data:
        activation_front = int(FEM_data[key]["sigma_avg_normal"].shape[0] / 2) - int(10 * pixelsize)        # activation front ist 10 micron left of center
        mirror_front = int(FEM_data[key]["sigma_avg_normal"].shape[0] / 2) + int(10 * pixelsize)
        FEM_data[key]["sigma_avg_normal"] = (FEM_data[key]["sigma_xx"] + FEM_data[key]["sigma_yy"]) / 2
        FEM_data[key]["sigma_avg_normal_average"] = np.nanmean(FEM_data[key]["sigma_avg_normal"], axis=(0, 1))
        FEM_data[key]["sigma_avg_normal_average_left"] = np.nanmean(FEM_data[key]["sigma_avg_normal"][:, 0:activation_front], axis=(0, 1))
        FEM_data[key]["sigma_avg_normal_average_right"] = np.nanmean(FEM_data[key]["sigma_avg_normal"][:, activation_front:-1], axis=(0, 1))

        FEM_data[key]["relsigma_avg_normal_average"] = FEM_data[key]["sigma_avg_normal_average"] / FEM_data[key]["sigma_avg_normal_average"][10]
        FEM_data[key]["relsigma_avg_normal_average_left"] = FEM_data[key]["sigma_avg_normal_average_left"] / FEM_data[key]["sigma_avg_normal_average"][10]
        FEM_data[key]["relsigma_avg_normal_average_right"] = FEM_data[key]["sigma_avg_normal_average_right"] / FEM_data[key]["sigma_avg_normal_average"][10]

        # FEM_data[key]["coupling"] = FEM_data[key]["relsigma_avg_normal_average_left"][32] * FEM_data[key]["relsigma_avg_normal_average_right"][32]

        FEM_data[key]["sigma_xx_x_profile"] = np.nanmean(FEM_data[key]["sigma_xx"], axis=0)
        FEM_data[key]["sigma_yy_x_profile"] = np.nanmean(FEM_data[key]["sigma_yy"], axis=0)

        FEM_data[key]["sigma_avg_normal_x_profile"] = np.nanmean(FEM_data[key]["sigma_avg_normal"], axis=0)
        FEM_data[key]["sigma_xx_x_profile_increase"] = FEM_data[key]["sigma_xx_x_profile"][:, 32] - FEM_data[key]["sigma_xx_x_profile"][:,
                                                                                                    20]
        FEM_data[key]["sigma_yy_x_profile_increase"] = FEM_data[key]["sigma_yy_x_profile"][:, 32] - FEM_data[key]["sigma_yy_x_profile"][:,
                                                                                                    20]
        FEM_data[key]["sigma_normal_x_profile_increase"] = FEM_data[key]["sigma_avg_normal_x_profile"][:, 32] - FEM_data[key]["sigma_avg_normal_x_profile"][:, 20]

        FEM_data[key]["xx_stress_increase_ratio"] = np.nansum(FEM_data[key]["sigma_xx_x_profile_increase"][mirror_front:-1]) / \
                                                    (np.abs(np.nansum(FEM_data[key]["sigma_xx_x_profile_increase"][0:activation_front])) +
                                                     np.abs(np.nansum(FEM_data[key]["sigma_xx_x_profile_increase"][mirror_front:-1])))

        FEM_data[key]["yy_stress_increase_ratio"] = np.nansum(FEM_data[key]["sigma_yy_x_profile_increase"][mirror_front:-1]) / \
                                                    (np.abs(np.nansum(FEM_data[key]["sigma_yy_x_profile_increase"][0:activation_front])) +
                                                     np.abs(np.nansum(FEM_data[key]["sigma_yy_x_profile_increase"][mirror_front:-1])))

        # # find maximum and/or minimun in stress increaye curves
        # y = FEM_data[key]["sigma_normal_x_profile_increase"]
        # max_left, _ = find_peaks(y[0:10])
        # # right is sometimes minimum and sometimes maximum, so I look for both
        # max_right, _ = find_peaks(y[40:50])
        # min_right, _ = find_peaks(-y[40:50])
        # min_right = min_right + 40
        # max_right = max_right + 40
        # peaks_pos = np.concatenate((max_left, min_right, max_right))
        # peaks = y[peaks_pos]

    return FEM_data


if __name__ == "__main__":
    directory = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/_FEM_simulations/doublets/"
    filename = "stress_feedbacks_"
    x_key = ":0"
    y_key = ":1"
    FEM_data = {}  # initialize dictionary
    x_start = -22.5  # left most x-value in micron
    x_end = 22.5  # right most x-value in micron
    y_start = -22.5  # left most y-value in micron
    y_end = 22.5  # right most y-value in micron
    pixelsize = 0.864  # desired final pixelsize in micron per pixel
    no_frames = 60
    grid_x, grid_y = np.mgrid[x_start:x_end:(x_end - x_start) / pixelsize * 1j, y_start:y_end:(y_end - y_start) / pixelsize * 1j]
    shortcut = True

    if shortcut == False:
        # feedbacks = [1.0]
        # feedbacks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        feedbacks = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        for feedback in feedbacks:
            print("Reading data from feedback: " + str(feedback))
            temp_dict = {}
            sigma_xx = np.zeros([grid_x.shape[0], grid_x.shape[1], no_frames])
            sigma_yy = np.zeros([grid_x.shape[0], grid_x.shape[1], no_frames])
            u_substrate = np.zeros([grid_x.shape[0], grid_x.shape[1], no_frames])
            v_substrate = np.zeros([grid_x.shape[0], grid_x.shape[1], no_frames])
            kN = np.zeros([grid_x.shape[0], grid_x.shape[1], no_frames])
            Es = np.zeros(no_frames)
            data_key1 = "Stress_" + str(feedback) + ":0"
            data_key2 = "Stress_" + str(feedback) + ":4"
            data_key3 = "Displacement_" + str(feedback) + ":0"
            data_key4 = "Displacement_" + str(feedback) + ":1"
            data_key5 = "Displacement Substrate_" + str(feedback) + ":0"
            data_key6 = "Displacement Substrate_" + str(feedback) + ":1"
            data_key7 = "kN_" + str(feedback)

            x_ref, y_ref = return_points(directory, filename + "2.csv", x_key, y_key)
            ux_ref, uy_ref = return_points(directory, filename + "2.csv", data_key3, data_key4)
            x = x_ref + ux_ref
            y = y_ref + uy_ref

            for frame in np.arange(no_frames):
                print('Interpolating map from frame: ' + str(frame))
                filename_full = filename + str(frame + 2) + ".csv"

                sigma_xx[:, :, frame], sigma_yy[:, :, frame] = interpolate_simulation_maps_rbf(directory, filename_full, x, y, data_key1,
                                                                                               data_key2, grid_x, grid_y)

                u_substrate[:, :, frame], v_substrate[:, :, frame] = interpolate_simulation_maps_rbf(directory, filename_full, x, y,
                                                                                                     data_key5, data_key6,
                                                                                                     grid_x, grid_y)

                kN[:, :, frame], _ = interpolate_simulation_maps_rbf(directory, filename_full, x, y, data_key7, data_key7,
                                                                     grid_x, grid_y)

            u_substrate *= 1e-6  # convert to m
            v_substrate *= 1e-6  # convert to m
            t_x = u_substrate * kN * 1e12  # convert to Pa
            t_y = v_substrate * kN * 1e12  # convert to Pa
            t = np.sqrt(t_x ** 2 + t_y ** 2)

            temp_dict["u_substrate"] = u_substrate
            temp_dict["v_substrate"] = v_substrate
            temp_dict["t_x"] = t_x
            temp_dict["t_y"] = t_y
            temp_dict["sigma_xx"] = sigma_xx
            temp_dict["sigma_yy"] = sigma_yy
            FEM_data["feedback" + str(feedback)] = temp_dict
    elif shortcut == True:
        FEM_data = pickle.load(
            open("C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/_FEM_simulations/FEM_doublets.dat", "rb"))

    FEM_data = analyze_FEM_data(FEM_data, pixelsize)

    with open("C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/_FEM_simulations/FEM_doublets.dat", 'wb') as outfile:
        pickle.dump(FEM_data, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    # sigma_avg_norm_diff = (sigma_xx_as + sigma_yy_as) / 2 - (sigma_xx_bs + sigma_yy_bs) / 2

# #%%
# df = pd.DataFrame.from_dict(np.array([x,y,sigma_yy]).T)
# df.columns = ['X_value','Y_value','Z_value']
# df['Z_value'] = pd.to_numeric(df['Z_value'])
# pivotted= df.pivot('Y_value','X_value','Z_value')
# sns.heatmap(pivotted,cmap='turbo')
