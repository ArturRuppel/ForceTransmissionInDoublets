import csv
import numpy as np
import pandas as pd
from copy import deepcopy
import os
import sys
import scipy.io as sio
import circle_fit as cf
from lmfit import Minimizer, Parameter, Model, report_fit
from scipy import optimize
import time

''' Main Analysis Script for TFM-Opto Experiment for Cells on H-shaped Micropattern (Artur Ruppel, Dennis Wörthmüller)

    Copyright: Dennis Wörthmüller
    Date: April 29, 2021  

    External python modules which are necessary: numpy, pandas scipy, lmfit

'''

# The the only things that should and have to be changed in this script are in the "-----" marked domain
# -------------------------------------------------------------------------------------------------------------------------------

# Number of cells for each condition (just a little look up table)
# AR1to1:29
# AR1to2 43
# AR2to1 18  
# AR1to1_singlets 41

# adjust this number
noCells = 2
noFrames = 60

# location where all the data input data sits and where the output data is written to 
# adjust this path to specifiy read directory (read and save direactory are the same)
directory = "C:/Users/Balland/Desktop/sigma_xx_averages/AR2to1 doublets half stim/"

# here you can specify which task to perform, default is that script performs all tasks
# which operations should be performed in the script
# Note !!!!!!!!!!!: If the script is run for the first time all options should be set to True
cell_no_start_loop = 1 #Default value is 1
do_circle_fit = True
do_calc_tangents = True
do_TEM_analysis_with_circle_fit = True
do_ellipse_approx = True
do_ellipse_fit = True
# -------------------------------------------------------------------------------------------------------------------------------

#####################################################################################
# Additional code snippets and functions which are generalized and will be used in main code:
# conversion factor from pixel to micro meter um
uM_per_pix = 0.108

def save_dict_to_csv(data_dict, path):
    '''Save a Dictionary to .csv file 

       Keys in dictionary give header (names of columns) in csv.

    '''
    # print(data_dict['r left arc'])
    length_items = []
    for key in data_dict:
        length_items.append(len(data_dict[key]))

    max_len = max(length_items)

    # make all dicts same length
    for key in data_dict:
        l = len(data_dict[key])
        if not max_len == l:
            data_dict[key] = np.concatenate(
                (data_dict[key], np.array(['']*(max_len-l))))
    # save data as data frame to csv
    df = pd.DataFrame(data_dict)
    df.to_csv(path, sep=',', encoding='utf-8')

# read the fibertrack x and y components of start and end point of top and bottom fiber (function still misses a description)
def read_tangent_points_mat(directory,c):

    read_path = directory + "fibertracking (%s).mat" % (c+1)
    mat = sio.loadmat(read_path)
    print(mat.keys())

    # first index gives point and second index gives time point
    # first point x coordinate over all time frames
    X_top_left = np.array(mat['Xtop'])[0, :]
    # first point y coordinate over all time frames
    Y_top_left = np.array(mat['Ytop'])[0, :]
    # last point x coordinate over all time frames
    X_top_right = np.array(mat['Xtop'])[-1, :]
    # last point y coordinate over all time frames
    Y_top_right = np.array(mat['Ytop'])[-1, :]

    # first point x coordinate over all time frames
    X_bottom_left = np.array(mat['Xbottom'])[0, :]
    # first point y coordinate over all time frames
    Y_bottom_left = np.array(mat['Ybottom'])[0, :]
    # last point x coordinate over all time frames
    X_bottom_right = np.array(mat['Xbottom'])[-1, :]
    # last point y coordinate over all time frames
    Y_bottom_right = np.array(mat['Ybottom'])[-1, :]

    return X_top_left, Y_top_left, X_top_right, Y_top_right, X_bottom_left, Y_bottom_left, X_bottom_right, Y_bottom_right

# read the radii and center data from radii_hyperfit (n).csv and return them in px (pixel)
def read_raddi_and_center_fit_circle_px(directory,c):

    df = pd.read_csv(directory+'radii_hyperfit (%s).csv' % (c+1))
    # center coordinates in px
    XC_upper = df['x-pos center [px] (upper)'].values
    YC_upper = df['y-pos center [px] (upper)'].values
    XC_lower = df['x-pos center [px] (lower)'].values
    YC_lower = df['y-pos center [px] (lower)'].values

    R_upper = df['Radius [px] (upper)'].values
    R_lower = df['Radius [px] (lower)'].values

    return XC_upper, YC_upper, XC_lower, YC_lower, R_upper, R_lower

# raddi and tangents data and return them in um, the tangents (n).csv also contains radii data which makes it easier (only read one file)
# function still misses a description
def read_radii_and_tangents_um(directory,c):
    '''Description '''



    df = pd.read_csv(directory+'tangents (%s).csv' % (c+1))
    # center coordinates in px

    R_upper = df['Radius [px] (upper)'].values*uM_per_pix
    R_lower = df['Radius [px] (lower)'].values*uM_per_pix

    tx_top_left = df['tx top left'].values
    ty_top_left = df['ty top left'].values

    tx_top_right = df['tx top right'].values
    ty_top_right = df['ty top right'].values

    tx_bottom_right = df['tx bottom right'].values
    ty_bottom_right = df['ty bottom right'].values

    tx_bottom_left = df['tx bottom left'].values
    ty_bottom_left = df['ty bottom left'].values
    # return radius in um
    return R_upper, R_lower, tx_top_left, ty_top_left, tx_top_right, ty_top_right, tx_bottom_right, ty_bottom_right, tx_bottom_left, ty_bottom_left

# read the force data in

# read forces in nN and subtract force contribution from sigma_x at adherent fiber
def read_force_data_mat(directory,c):
    # bottom forces have a negative sign because y-axis pointing down 
    read_path = directory+"corner_averages.mat"
    read_path_sigma_contribution = directory + "sigma_contribution.mat"

    mat = sio.loadmat(read_path)
    # mat_sigma_contr = sio.loadmat(read_path_sigma_contribution)

    # print(mat.keys())
    # print(mat_sigma_contr.keys())
    # F_x_sigma_left = np.array(mat_sigma_contr['Fx_centerleft'])[:, c]
    # F_x_sigma_right = np.array(mat_sigma_contr['Fx_centerright'])[:, c]
    sigma_xx = np.load(directory + '/sigma_xx.npy')
    F_x_sigma_left = sigma_xx[:, c]*35*1e-6#*1.41 # length of a fiber for 1to1
    F_x_sigma_right = sigma_xx[:, c]*35*1e-6#*1.41
    # topleft
    F_x_topleft = np.array(mat['Tx_topleft'])[:, c]# - F_x_sigma_left/2  # all forces in time for cell c
    F_y_topleft = np.array(mat['Ty_topleft'])[:, c]
    # F_total_topleft = np.sqrt(F_x_topleft**2 + F_y_topleft**2)
    # topright
    # all forces in time for cell c
    F_x_topright = np.array(mat['Tx_topright'])[:, c]# - F_x_sigma_right/2
    F_y_topright = np.array(mat['Ty_topright'])[:, c]
    # F_total_topright = np.sqrt(F_x_topright**2 + F_y_topright**2)
    # bottomright
    F_x_bottomright = np.array(mat['Tx_bottomright'])[:, c]# - F_x_sigma_right/2  # all forces in time for cell c
    F_y_bottomright = np.array(mat['Ty_bottomright'])[:, c]
    # F_total_bottomright = np.sqrt(F_x_bottomright**2 + F_y_bottomright**2)
    # bottomleft
    F_x_bottomleft = np.array(mat['Tx_bottomleft'])[:, c]# - F_x_sigma_left/2  # all forces in time for cell c
    F_y_bottomleft = np.array(mat['Ty_bottomleft'])[:, c]
    # F_total_bottomleft = np.sqrt(F_x_bottomleft**2 + F_y_bottomleft**2)

    # F_total = F_total_topleft + F_total_topright + F_total_bottomright + F_total_bottomleft

    # force in nN
    print(F_y_bottomleft,F_y_bottomright,F_y_topleft,F_y_topright)
    return F_x_topleft*1e9, F_y_topleft*1e9, F_x_topright*1e9, F_y_topright*1e9, F_x_bottomright*1e9, F_y_bottomright*1e9, F_x_bottomleft*1e9, F_y_bottomleft*1e9


# read radii tangents and center and return them in um
def read_radii_and_tangents_and_center_um(directory,c):

    df = pd.read_csv(directory+'tangents (%s).csv' % (c+1))
    # center coordinates in px

    R_upper = df['Radius [px] (upper)'].values*uM_per_pix
    R_lower = df['Radius [px] (lower)'].values*uM_per_pix

    xc_upper = df['x-pos center [px] (upper)'].values*uM_per_pix
    yc_upper = df['y-pos center [px] (upper)'].values*uM_per_pix

    xc_lower = df['x-pos center [px] (lower)'].values*uM_per_pix
    yc_lower = df['y-pos center [px] (lower)'].values*uM_per_pix

    tx_top_left = df['tx top left'].values
    ty_top_left = df['ty top left'].values

    tx_top_right = df['tx top right'].values
    ty_top_right = df['ty top right'].values

    tx_bottom_right = df['tx bottom right'].values
    ty_bottom_right = df['ty bottom right'].values

    tx_bottom_left = df['tx bottom left'].values
    ty_bottom_left = df['ty bottom left'].values
    # return radius in um
    return R_upper, R_lower, xc_upper, yc_upper, xc_lower, yc_lower, tx_top_left, ty_top_left, tx_top_right, ty_top_right, tx_bottom_right, ty_bottom_right, tx_bottom_left, ty_bottom_left

# read sigma contribution and return it in nN/um
def read_sigma_contribution_mat(directory,c):

    read_path_sigma_contribution = directory + "sigma_contribution.mat"
    mat_sigma_contr = sio.loadmat(read_path_sigma_contribution)
    # print(mat.keys())
    print(mat_sigma_contr.keys())
    # in SI N/m
    sigma_x_left = np.array(mat_sigma_contr['sigma_x_left'])[:, c]
    sigma_x_right = np.array(mat_sigma_contr['sigma_x_right'])[:, c]

    # in nN/um
    return sigma_x_left*1e3, sigma_x_right*1e3

# read line tensions from TEM_circles oputput is in nN
def read_line_tension(directory,c):

    df = pd.read_csv(directory+'TEM_circles (%s).csv' % (c+1))
    # center coordinates in px

    landa_top_left = df['line tension top left'].values
    landa_top_right = df['line tension top right'].values

    landa_bottom_left = df['line tension bottom left'].values
    landa_bottom_right = df['line tension bottom right'].values

    return landa_top_left, landa_top_right, landa_bottom_left, landa_bottom_right



# functions necessary for ellipse std estimation
def read_ellipse_center_um(directory,c):

    df = pd.read_csv(directory+'ellipse_approx (%s).csv' % (c+1))
    # center coordinates in px

    a_top = df['a top [um]'].values
    b_top = df['b top [um]'].values

    a_bottom = df['a bottom [um]'].values
    b_bottom = df['b bottom [um]'].values

    xc_top = df['xc top [um]'].values
    yc_top = df['yc top [um]'].values

    xc_bottom = df['xc bottom [um]'].values
    yc_bottom = df['yc bottom [um]'].values

    # return radius in um
    return a_top, b_top, xc_top, yc_top, a_bottom, b_bottom, xc_bottom, yc_bottom

def read_radii_center_um(directory,c):

    df = pd.read_csv(directory+'tangents (%s).csv' % (c+1))
    # center coordinates in px

    R_upper = df['Radius [px] (upper)'].values*uM_per_pix
    R_lower = df['Radius [px] (lower)'].values*uM_per_pix

    xc_upper = df['x-pos center [px] (upper)'].values*uM_per_pix
    yc_upper = df['y-pos center [px] (upper)'].values*uM_per_pix

    xc_lower = df['x-pos center [px] (lower)'].values*uM_per_pix
    yc_lower = df['y-pos center [px] (lower)'].values*uM_per_pix

    # return radius in um
    return R_upper, R_lower, xc_upper, yc_upper, xc_lower, yc_lower


# functions necessary for ellipse fit
def read_ellipse_center_fit_um(directory, c):

    df = pd.read_csv(directory+'ellipse_data_fit (%s).csv' % (c+1))
    # center coordinates in px

    a_top = df['a top [um]'].values
    b_top = df['b top [um]'].values

    a_bottom = df['a bottom [um]'].values
    b_bottom = df['b bottom [um]'].values

    xc_top = df['xc top [um]'].values
    yc_top = df['yc top [um]'].values

    xc_bottom = df['xc bottom [um]'].values
    yc_bottom = df['yc bottom [um]'].values

    # return radius in um
    return a_top, b_top, xc_top, yc_top, a_bottom, b_bottom, xc_bottom, yc_bottom


def read_ellipse_center_um_sigma(directory, c):

    df = pd.read_csv(directory+'ellipse_approx (%s).csv' % (c+1))
    # center coordinates in px

    a_top = df['a top [um]'].values
    b_top = df['b top [um]'].values

    a_bottom = df['a bottom [um]'].values
    b_bottom = df['b bottom [um]'].values

    xc_top = df['xc top [um]'].values
    yc_top = df['yc top [um]'].values

    xc_bottom = df['xc bottom [um]'].values
    yc_bottom = df['yc bottom [um]'].values

    sigma_y_top = df['sigma y top [nN/um]'].values
    sigma_y_bottom = df['sigma y bottom [nN/um]'].values

    sigma_x = df['sigma x [nN/um]'].values

    # return radius in um
    return a_top, b_top, xc_top, yc_top, a_bottom, b_bottom, xc_bottom, yc_bottom, sigma_y_top, sigma_y_bottom, sigma_x





##################################################################################################
################         !!!       MAIN FUNCTION STARTS HERE         !!!          ################
##################################################################################################



#####################################################################################
# Step 1: fit circles to ellipses
# Input file fibertrack (n).mat, Output file radii_hyperfit (n).csv

def main_circle_fit(directory):
    ''' Read fibertracking and fit circles to it using the hyoer fit algorithm

        Input: directory where fibertrack (n).mat sits 
        Output: circle_fit (n).csv -> contains: 
    
    '''

    def fit_data_in_file(file_path, arc='left'):

        R_list = []
        error_list = []
        xc_list = []
        yc_list = []

        mat = sio.loadmat(file_path)
        print(mat.keys())
        # first index gives position and second index gives frame/time
        x_data = np.array(mat['X%s' % (arc)])
        y_data = np.array(mat['Y%s' % (arc)])

        for frame in np.arange(0, noFrames):

            # first index gives position and second index gives frame/time
            x = x_data[:, frame]
            y = y_data[:, frame]

            coords = [[x[i], y[i]] for i in range(len(x))]
            xc, yc, R, s = cf.hyper_fit(coords)

            R_list.append(R)
            error_list.append(s)
            xc_list.append(xc)
            yc_list.append(yc)

        return xc_list, yc_list, R_list, error_list

    for c in range(cell_no_start_loop-1,noCells):
        # c starts from 0 but file numbers (actual cell number) starts from 1
        read_path = directory + "fibertracking (%s).mat" % (c+1)

        # for each cell return fitted data (as list with entries meaning the frame) for all arcs of all frames 
        xc_r, yc_r, R_r, error_r = fit_data_in_file(read_path, 'right')
        xc_l, yc_l, R_l, error_l = fit_data_in_file(read_path, 'left')
        xc_up, yc_up, R_up, error_up = fit_data_in_file(read_path, 'top')
        xc_lo, yc_lo, R_lo, error_lo = fit_data_in_file(read_path, 'bottom')

        # create data dictionary
        data_dict = {'Radius [px] (upper)': R_up,
                     'Radius Std [px] (upper)': error_up,
                     'x-pos center [px] (upper)': xc_up,
                     'y-pos center [px] (upper)': yc_up,
                     'Radius [px] (right)': R_r,
                     'Radius Std [px] (right)': error_r,
                     'x-pos center [px] (right)': xc_r,
                     'y-pos center [px] (right)': yc_r,
                     'Radius [px] (lower)': R_lo,
                     'Radius Std [px] (lower)': error_lo,
                     'x-pos center [px] (lower)': xc_lo,
                     'y-pos center [px] (lower)': yc_lo,
                     'Radius [px] (left)': R_l,
                     'Radius Std [px] (left)': error_l,
                     'x-pos center [px] (left)': xc_l,
                     'y-pos center [px] (left)': yc_l}
        # save fit data to csv for each cell
        save_dict_to_csv(data_dict, directory + '/radii_hyperfit (%s).csv'%(c+1))


    print("Circle Fit terminated successfully!")
    return None

# run circle_fit function
if do_circle_fit is True:
    main_circle_fit(directory)
else:
    print("main_circle_fit is not called!")

####################################################################################
# Step 2: calculate tangents of circles at fibertrack start and end point
# Input files needed: fibertracking (n).mat and radii_hyperfit (n).csv, Output tangents_n.csv

def main_calc_tangents(directory):
    ''' Read radii_hyperfit (n).csv and fibertrack (n).mat to calculate tangents at start and end point of fiber track

        Input: directory where input-files sit 
        Output: tangents (n).csv -> contains: 
    
    '''
    # tl = top left, other "pos" are tr, bl, br
    def calc_tangent(X, Y, XC, YC, R, pos='tl'):
        '''caluclate the tangents of point (X,Y) but also return closest point on circle (x,y)'''
        theta = np.arctan((Y-YC)/(X-XC))

        if pos == 'bl':
            theta = theta + np.pi
        elif pos == 'tl':
            theta = np.pi+theta

        x = XC + R*np.cos(theta)
        y = YC + R*np.sin(theta)
        tx = -np.sin(theta)
        ty = np.cos(theta)

        if pos == 'tl':
            theta = theta + np.pi
            tx = -tx
            ty = -ty

        elif pos == 'br':
            theta = np.pi+theta
            tx = -tx
            ty = -ty

        # ty = (x-XC)/R
        # tx = np.sqrt(1-ty**2)
        # if pos == 'tl':
        #     print("X touch ",X,"XC ",XC,"R ",R)
        #     print(ty)

        # if pos == 'tr':
        #     tx = -tx
        #     ty = -ty
        # elif pos == 'br':
        #     tx = -tx
        # elif pos == 'bl':
        #     ty = -ty

        return tx, ty, x, y

    for c in range(cell_no_start_loop-1,noCells):
        # each quantity here is a vector/array containing datapoint for all time frames
        (X_top_left, Y_top_left, X_top_right, Y_top_right,
         X_bottom_left, Y_bottom_left, X_bottom_right, Y_bottom_right) = read_tangent_points_mat(directory,c)

        XC_upper, YC_upper, XC_lower, YC_lower, R_upper, R_lower = read_raddi_and_center_fit_circle_px(directory, c)

        # tangents top left
        tx_top_left, ty_top_left, x_top_left, y_top_left = calc_tangent(X_top_left, Y_top_left, XC_upper, YC_upper, R_upper, pos='tl')
        # tangents top right
        tx_top_right, ty_top_right, x_top_right, y_top_right = calc_tangent(X_top_right, Y_top_right, XC_upper, YC_upper, R_upper, pos='tr')
        # tangents bottom left
        tx_bottom_left, ty_bottom_left, x_bottom_left, y_bottom_left = calc_tangent(X_bottom_left, Y_bottom_left, XC_lower, YC_lower, R_lower, pos='bl')
        # tangents bottom right
        tx_bottom_right, ty_bottom_right, x_bottom_right, y_bottom_right = calc_tangent(X_bottom_right, Y_bottom_right, XC_lower, YC_lower, R_lower, pos='br')

        # debugg function (should be commented out always)
        # def print_numbers():
            # print("Cell No ", c+1)
            # print("Top left")
            # print("Radius ", R_upper[0])
            # print("Touching Point ",X_top_left[0],Y_top_left[0])
            # print("Tangent Point ",x_top_left[0],y_top_left[0])
            # print("Center Point ",XC_upper[0],YC_upper[0])
            # print("Tangents ",tx_top_left[0],ty_top_left[0])
            # print("Norm ", tx_top_left[0]**2+ty_top_left[0]**2)
            # print("Theta ", np.arctan((Y_top_left[0]-YC_upper[0])/(X_top_left[0]-XC_upper[0]))/np.pi)
            # print("Origin Centered ", X_top_left[0]-XC_upper[0], Y_top_left[0]-YC_upper[0])
            # print("################################")
            # print("Top right")
            # print("Radius ", R_upper[0])
            # print("Touching Point ",X_top_right[0],Y_top_right[0])
            # print("Tangent Point ",x_top_right[0],y_top_right[0])
            # print("Center Point ",XC_upper[0],YC_upper[0])
            # print("Tangents ",tx_top_right[0],ty_top_right[0])
            # print("Norm ", tx_top_right[0]**2+ty_top_right[0]**2)
            # print("Theta ", np.arctan((Y_top_right[0]-YC_upper[0])/(X_top_right[0]-XC_upper[0]))/np.pi)
            # print("Origin Centered ", X_top_right[0]-XC_upper[0], Y_top_right[0]-YC_upper[0])
            # print("################################")
            # print("Bottom left")
            # print("Radius ", R_lower[0])
            # print("Touching Point ",X_bottom_left[0],Y_bottom_left[0])
            # print("Tangent Point ",x_bottom_left[0],y_bottom_left[0])
            # print("Center Point ",XC_lower[0],YC_lower[0])
            # print("Tangents ",tx_bottom_left[0],ty_bottom_left[0])
            # print("Norm ", tx_bottom_left[0]**2+ty_bottom_left[0]**2)
            # print("Theta ", np.arctan((Y_bottom_left[0]-YC_lower[0])/(X_bottom_left[0]-XC_lower[0]))/np.pi)
            # print("Origin Centered ", X_bottom_left[0]-XC_lower[0], Y_bottom_left[0]-YC_lower[0])
            # print("################################")
            # print("Bottom right")
            # print("Radius ", R_upper[0])
            # print("Touching Point ",X_bottom_right[0],Y_bottom_right[0])
            # print("Tangent Point ",x_bottom_right[0],y_bottom_right[0])
            # print("Center Point ",XC_lower[0],YC_lower[0])
            # print("Tangents ",tx_bottom_right[0],ty_bottom_right[0])
            # print("Norm ", tx_bottom_right[0]**2+ty_bottom_right[0]**2)
            # print("Theta ", np.arctan((Y_bottom_right[0]-YC_lower[0])/(X_bottom_right[0]-XC_lower[0]))/np.pi)
            # print("Origin Centered ", X_bottom_right[0]-XC_lower[0], Y_bottom_right[0]-YC_lower[0])
            # print("################################")

        # define data dictionary for each cell
        data_dict = {'tx top left': tx_top_left,
                     'ty top left': ty_top_left,
                     'tx top right': tx_top_right,
                     'ty top right': ty_top_right,
                     'tx bottom left': tx_bottom_left,
                     'ty bottom left': ty_bottom_left,
                     'tx bottom right': tx_bottom_right,
                     'ty bottom right': ty_bottom_right,
                     'xTouch top left': X_top_left,
                     'yTouch top left': Y_top_left,
                     'xTouch top right': X_top_right,
                     'yTouch top right': Y_top_right,
                     'xTouch bottom left': X_bottom_left,
                     'yTouch bottom left': Y_bottom_left,
                     'xTouch bottom right': X_bottom_right,
                     'yTouch bottom right': Y_bottom_right,
                     'x-pos center [px] (upper)': XC_upper,
                     'y-pos center [px] (upper)': YC_upper,
                     'x-pos center [px] (lower)': XC_lower,
                     'y-pos center [px] (lower)': YC_lower,
                     'Radius [px] (upper)': R_upper,
                     'Radius [px] (lower)': R_lower}
        # write date dictionary to tangents (c+1).csv (each cell)
        save_dict_to_csv(data_dict, directory+'tangents (%s).csv' % (c+1))

    print("Calculating tangents terminated successfully!")
    return None

# run main_calc_tangents function
if do_calc_tangents is True:
    main_calc_tangents(directory)
else:
    print("main_calc_tangents is not called!") 

###################################################################################
# Step 3: perform a tfm analysis based on the circle fits 
# Input files needed: tangents (n).csv, corner_averages.mat,sigma_contribution.mat, Output TFM_circles (n).csv

def main_TEM_circles(directory):
    '''Desciption'''
    # calculate forces coming from free (landa) and adherent fiber (f)
    def calc_landa_and_f(tx, ty, Fx, Fy):
        landa = Fx/tx
        f = Fy - Fx*ty/tx
        return landa, f

    for c in range(cell_no_start_loop-1,noCells):
        # each quantity here is a vector/array containing datapoint for all time frames
        (R_upper, R_lower, tx_top_left, ty_top_left, tx_top_right, ty_top_right,
         tx_bottom_right, ty_bottom_right, tx_bottom_left, ty_bottom_left) = read_radii_and_tangents_um(directory,c)  # radii in um

        Fx_top_left, Fy_top_left, Fx_top_right, Fy_top_right, Fx_bottom_right, Fy_bottom_right, Fx_bottom_left, Fy_bottom_left = read_force_data_mat(directory,c)  # in nN

        landa_top_left, f_top_left = calc_landa_and_f(tx_top_left, ty_top_left, Fx_top_left, Fy_top_left)  # in nN
        landa_top_right, f_top_right = calc_landa_and_f(tx_top_right, ty_top_right, Fx_top_right, Fy_top_right)
        landa_bottom_left, f_bottom_left = calc_landa_and_f(tx_bottom_left, ty_bottom_left, Fx_bottom_left, Fy_bottom_left)
        landa_bottom_right, f_bottom_right = calc_landa_and_f(tx_bottom_right, ty_bottom_right, Fx_bottom_right, Fy_bottom_right)

        sigma_top_left = landa_top_left/R_upper  # in nN/um
        sigma_top_right = landa_top_right/R_upper
        sigma_bottom_left = landa_bottom_left/R_lower
        sigma_bottom_right = landa_bottom_right/R_lower

        # debug prints
        # print("Cell ",c+1)
        # print("# Bottom Left")
        # print(Fx_bottom_left[0],Fy_bottom_left[0])
        # print(f_bottom_left[0])
        # print(tx_bottom_left[0],ty_bottom_left[0])
        # print("# Top Left")
        # print(Fx_top_left[0],Fy_top_left[0])
        # print(f_top_left[0])
        # print(tx_top_left[0],ty_top_left[0])
        # print("# Bottom Right")
        # print(Fx_bottom_right[0],Fy_bottom_right[0])
        # print(f_bottom_right[0])
        # print(tx_bottom_right[0],ty_bottom_right[0])
        # print("# Top Right")
        # print(Fx_top_right[0],Fy_top_right[0])
        # print(f_top_right[0])
        # print(tx_top_right[0],ty_top_right[0])
        # print("######################")

        data_dict = {'line tension top left': landa_top_left,
                     'f top left': f_top_left,
                     'line tension top right': landa_top_right,
                     'f top right': f_top_right,
                     'line tension bottom left': landa_bottom_left,
                     'f bottom left': f_bottom_left,
                     'line tension bottom right': landa_bottom_right,
                     'f bottom right': f_bottom_right,
                     'surface tension top left': sigma_top_left,
                     'surface tension top right': sigma_top_right,
                     'surface tension bottom left': sigma_bottom_left,
                     'surface tension bottom right': sigma_bottom_right,
                     'Radius [um] (upper)': R_upper,
                     'Radius [um] (lower)': R_lower}

        save_dict_to_csv(data_dict, directory +'/TEM_circles (%s).csv' % (c+1))
    print("TEM circle analysis terminated successfully!")
    return None

# run main_TEM_circles
if do_TEM_analysis_with_circle_fit is True:
    main_TEM_circles(directory)
else:
    print("main_TEM_circles is not called!")

####################################################################################
# Step 4: given TEM_circle data and sigma_x contribution estimate ellipse based on curvature of circle
# Input files needed, tangents (n).csv, sigma_contribution.mat,TEM_circles (n).csv, Output ellipse_approx (n).csv

def main_ellipse_approx(directory):
    '''Description'''

    theta_max = 15  # in degree
    theta_max_rad = theta_max/180*np.pi
    
    def return_results(sigma_y, landa, phi, sigma_x, R, xc, yc, arc='top'):


        b_square = (landa/sigma_x)**2 * (1+(sigma_x/sigma_y) * np.tan(phi)**2)/(1+np.tan(phi)**2)
        a_square = (landa)**2/(sigma_x*sigma_y) * (1+(sigma_x/sigma_y)*np.tan(phi)**2)/(1+np.tan(phi)**2)
        b = np.sqrt(b_square)
        a = np.sqrt(a_square)
        sigma_isotrop = 0.5*(sigma_x+sigma_y)
        sigma_anisotrop_x = 0.5*(sigma_x-sigma_y)
        sigma_anisotrop_y = 0.5*(sigma_y-sigma_x)

        delta = np.absolute(b-R)
        xc_ellipse = xc
        if arc == 'top':
            if b > R:
                yc_ellipse = yc-delta
            elif b <= R:
                yc_ellipse = yc+delta
        elif arc == 'bottom':
            if b > R:
                yc_ellipse = yc+delta
            elif b <= R:
                yc_ellipse = yc-delta

        return a, b, sigma_isotrop, sigma_anisotrop_x, sigma_anisotrop_y, xc_ellipse, yc_ellipse

    
    def estimate_ellipse_approx_std(c):

        fibertrack_path = directory + "fibertracking (%s).mat" % (c+1)
        mat = sio.loadmat(fibertrack_path)
        print(mat.keys())
        # first index gives position and second index gives frame/time
        x_data_bottom = np.array(mat['Xbottom'])
        y_data_bottom = np.array(mat['Ybottom'])

        # first index gives position and second index gives frame/time
        x_data_top = np.array(mat['Xtop'])
        y_data_top = np.array(mat['Ytop'])

        R_upper, R_lower, xc_upper, yc_upper, xc_lower, yc_lower = read_radii_center_um(directory,c)
        a_top, b_top, xc_top, yc_top, a_bottom, b_bottom, xc_bottom, yc_bottom = read_ellipse_center_um(directory,c)

        a_u = a_top
        b_u = b_top

        a_l = a_bottom
        b_l = b_bottom


        std_u_list = []
        std_l_list = []
        for frame in range(noFrames):
            # first read in tracks for each frame
            # first index gives position and second index gives frame/time
            x_top_track = x_data_top[:, frame]*uM_per_pix
            y_top_track = y_data_top[:, frame]*uM_per_pix

            # first index gives position and second index gives frame/time
            x_bottom_track = x_data_bottom[:, frame]*uM_per_pix
            y_bottom_track = y_data_bottom[:, frame]*uM_per_pix

            # loop over all data pointa in one track
            dist_container_u = []
            dist_container_l = []

            dist_container_circle_u = []
            dist_container_circle_l = []
            # print("No of tracking points ", len(x_top_track), frame)
            # print("No of tracking points ", len(x_bottom_track), frame)
            for j in range(len(x_top_track)):
                dist_u = estimate_distance(x_top_track[j], y_top_track[j], a_u[frame], b_u[frame], xc_top[frame], yc_top[frame], angle=0)
                dist_circle_u = ((xc_upper[frame]-x_top_track[j])**2 +(yc_upper[frame]-y_top_track[j])**2)**0.5 - R_upper[frame]
                # store squared distance
                dist_container_u.append(dist_u**2)
                dist_container_circle_u.append(dist_circle_u**2)

            for j in range(len(x_bottom_track)):
                dist_l = estimate_distance(x_bottom_track[j], y_bottom_track[j], a_l[frame], b_l[frame], xc_bottom[frame], yc_bottom[frame], angle=0)
                dist_circle_l = ((xc_lower[frame]-x_bottom_track[j])**2 + (yc_lower[frame]-y_bottom_track[j])**2)**0.5 - R_lower[frame]

                # store squared distance
                dist_container_l.append(dist_l**2)
                dist_container_circle_l.append(dist_circle_l**2)

                
                std_u = np.sqrt(np.sum(dist_container_u)/len(dist_container_u))
                std_l = np.sqrt(np.sum(dist_container_l)/len(dist_container_l))

            # arrays of length 60 which have std of top/bottom ellipse for each frame
            std_u_list.append(std_u)
            std_l_list.append(std_l)

        return std_u_list,std_l_list

    def ellipe_tan_dot(rx, ry, px, py, theta):
        '''Dot product of the equation of the line formed by the point
           with another point on the ellipse's boundary and the tangent of the ellipse
           at that point on the boundary.
        '''
        return ((rx ** 2 - ry ** 2) * np.cos(theta) * np.sin(theta) -px * rx * np.sin(theta) + py * ry * np.cos(theta))


    def ellipe_tan_dot_derivative(rx, ry, px, py, theta):
        '''derivative of ellipe_tan_dot.
        '''
        return ((rx ** 2 - ry ** 2) * (np.cos(theta) ** 2 - np.sin(theta) ** 2) -px * rx * np.cos(theta) - py * ry * np.sin(theta))


    def estimate_distance(x, y, rx, ry, x0=0, y0=0, angle=0, error=1e-5):
        '''Given a point (x, y), and an ellipse with major - minor axis (rx, ry),
           its center at (x0, y0), and with a counter clockwise rotation of
           `angle` degrees, will return the distance between the ellipse and the
           closest point on the ellipses boundary.
        '''
        x -= x0
        y -= y0
        if angle:
            # rotate the points onto an ellipse whose rx, and ry lay on the x, y
            # axis
            angle = -pi / 180. * angle
            x, y = x * np.cos(angle) - y * np.sin(angle), x * np.sin(angle) + y * np.cos(angle)

        theta = np.arctan2(rx * y, ry * x)
        while np.absolute(ellipe_tan_dot(rx, ry, x, y, theta)) > error:
            theta -= ellipe_tan_dot(rx, ry, x, y, theta) / ellipe_tan_dot_derivative(rx, ry, x, y, theta)

        px, py = rx * np.cos(theta), ry * np.sin(theta)
        return ((x - px) ** 2 + (y - py) ** 2) ** .5
    
    sigma_xx = np.load(directory+'sigma_xx.npy')
    
    for c in range(cell_no_start_loop-1,noCells):

        print("CELL ", c+1)
        # define save container
        a_list_t = []
        b_list_t = []
        sigma_y_list_t = []
        xc_list_t = []
        yc_list_t = []

        a_list_b = []
        b_list_b = []
        sigma_y_list_b = []
        xc_list_b = []
        yc_list_b = []

        sigma_x_list = []
        

        (R_upper, R_lower,
         xc_upper, yc_upper,
         xc_lower, yc_lower,
         tx_top_left, ty_top_left,
         tx_top_right, ty_top_right,
         tx_bottom_right, ty_bottom_right,
         tx_bottom_left, ty_bottom_left) = read_radii_and_tangents_and_center_um(directory,c)

        # sigma_x_left, sigma_x_right = read_sigma_contribution_mat(directory,c)
        sigma_x_left = sigma_xx[:,c]
        sigma_x_right = sigma_xx[:,c]

        landa_top_left, landa_top_right, landa_bottom_left, landa_bottom_right = read_line_tension(directory,c)

        for frame in np.arange(0, noFrames):
            print("FRAME ", frame)
            R_u = R_upper[frame]
            R_l = R_lower[frame]

            curvature_u = (1/R_u).item()
            curvature_l = (1/R_l).item()

            xc_u = xc_upper[frame]
            yc_u = yc_upper[frame]
            xc_l = xc_lower[frame]
            yc_l = yc_lower[frame]

            ty_tl = np.absolute(ty_top_left[frame])
            ty_tr = np.absolute(ty_top_right[frame])
            ty_bl = np.absolute(ty_bottom_left[frame])
            ty_br = np.absolute(ty_bottom_right[frame])

            ty_top = (ty_tl + ty_tr)/2
            ty_bottom = (ty_bl + ty_br)/2

            phi_tl = np.arcsin(ty_tl)
            phi_tr = np.arcsin(ty_tr)
            phi_bl = np.arcsin(ty_bl)
            phi_br = np.arcsin(ty_br)

            phi_top = (phi_tl + phi_tr)/2
            phi_bottom = (phi_bl + phi_br)/2

            landa_tl = np.absolute(landa_top_left[frame])
            landa_tr = np.absolute(landa_top_right[frame])
            landa_bl = np.absolute(landa_bottom_left[frame])
            landa_br = np.absolute(landa_bottom_right[frame])

            landa_top = (landa_tl + landa_tr)/2
            landa_bottom = (landa_bl + landa_br)/2

            sigma_x_l = sigma_x_left[frame]
            sigma_x_r = sigma_x_right[frame]

            sigma_x_av = (sigma_x_l + sigma_x_r)/2

            def k_top(x, sigma_y):
                phi = phi_top
                landa = landa_top  # nN
                sigma_x = sigma_x_av  # nN/um
                print("phi ", phi)
                print("landa ", landa)
                print("sigma_x ", sigma_x)
                b_square = (landa/sigma_x)**2 * (1+(sigma_x/sigma_y) * np.tan(phi)**2)/(1+np.tan(phi)**2)
                a_square = (landa)**2/(sigma_x*sigma_y) * (1+(sigma_x/sigma_y)*np.tan(phi)**2)/(1+np.tan(phi)**2)
                a = np.sqrt(a_square)
                b = np.sqrt(b_square)
                print("a ", a)
                print("b ", b)
                # returns the curvature over x range
                return a*b/(a**2*np.sin(x)**2+b**2*np.cos(x)**2)**(3/2)

            def k_bottom(x, sigma_y):
                phi = phi_bottom
                landa = landa_bottom  # nN
                sigma_x = sigma_x_av  # nN/um
                b_square = (landa/sigma_x)**2 * (1+(sigma_x/sigma_y) * np.tan(phi)**2)/(1+np.tan(phi)**2)
                a_square = (landa)**2/(sigma_x*sigma_y) * (1+(sigma_x/sigma_y)*np.tan(phi)**2)/(1+np.tan(phi)**2)
                a = np.sqrt(a_square)
                b = np.sqrt(b_square)
                # print("a ", a)
                # print("b ",b)
                # returns the curvature over x range
                return a*b/(a**2*np.sin(x)**2+b**2*np.cos(x)**2)**(3/2)

            x = np.linspace(np.pi/2-(theta_max_rad/np.pi)*np.pi,np.pi/2+(theta_max_rad/np.pi)*np.pi, 1000)

            model_top = Model(k_top, independent_vars=['x'], nan_policy='omit')

            model_bottom = Model(k_bottom, independent_vars=['x'], nan_policy='omit')

            params_top = model_top.make_params(sigma_y=landa_top/R_u)
            params_bottom = model_bottom.make_params(sigma_y=landa_bottom/R_l)

            print(model_top.param_names, model_top.independent_vars)

            # params.add(name="factor", min=0.0, max=1)
            # model.print_param_hints(colwidth=8)

            # perform the fit
            result_top = model_top.fit(curvature_u, params_top, x=x, method='leastsq')
            result_bottom = model_bottom.fit(curvature_l, params_bottom, x=x, method='leastsq')

            # print(result.fit_report())
            sigma_y_top = result_top.params['sigma_y'].value
            a_top, b_top, sigma_isotrop_top, sigma_anisotrop_x_top, sigma_anisotrop_y_top, xc_ellipse_top, yc_ellipse_top = return_results(sigma_y_top, landa_top, phi_top, sigma_x_av, R_u, xc_u, yc_u, arc='top')
            # print('sigma_x ', sigma_x_av)
            # print('sigma_y_top ', sigma_y_top)
            # print('sigma_anisotrop_x_top ', sigma_anisotrop_x_top)
            # print('sigma_anisotrop_y_top ', sigma_anisotrop_y_top)
            # print('a_top ', a_top)
            # print('b_top ', b_top)
            # print("xc_ellipse_top ", xc_ellipse_top)
            # print("yc_ellipse_top ", yc_ellipse_top)

            sigma_y_bottom = result_bottom.params['sigma_y'].value
            a_bottom, b_bottom, sigma_isotrop_bottom, sigma_anisotrop_x_bottom, sigma_anisotrop_y_bottom, xc_ellipse_bottom, yc_ellipse_bottom = return_results(sigma_y_bottom, landa_bottom, phi_bottom, sigma_x_av, R_l, xc_l, yc_l, arc='bottom')
            # print('sigma_x ', sigma_x_av)
            # print('sigma_y_bottom ', sigma_y_bottom)
            # print('sigma_anisotrop_x_bottom ', sigma_anisotrop_x_bottom)
            # print('sigma_anisotrop_y_bottom ', sigma_anisotrop_y_bottom)
            # print('a_bottom ', a_bottom)
            # print('b_bottom ', b_bottom)
            # print("xc_ellipse_bottom ", xc_ellipse_bottom)
            # print("yc_ellipse_bottom ", yc_ellipse_bottom)

            a_list_t.append(a_top)
            b_list_t.append(b_top)
            sigma_y_list_t.append(sigma_y_top)
            xc_list_t.append(xc_ellipse_top)
            yc_list_t.append(yc_ellipse_top)

            a_list_b.append(a_bottom)
            b_list_b.append(b_bottom)
            sigma_y_list_b.append(sigma_y_bottom)
            xc_list_b.append(xc_ellipse_bottom)
            yc_list_b.append(yc_ellipse_bottom)

            sigma_x_list.append(sigma_x_av)
            sigma_y_mean = (sigma_y_top + sigma_y_bottom)/2
       

        data_dict = {'sigma x [nN/um]': sigma_x_list,
                     'sigma y top [nN/um]': sigma_y_list_t,
                     'a top [um]': a_list_t,
                     'b top [um]': b_list_t,
                     'xc top [um]': xc_list_t,
                     'yc top [um]': yc_list_t,
                     'a bottom [um]': a_list_b,
                     'b bottom [um]': b_list_b,
                     'sigma y bottom [nN/um]': sigma_y_list_b,
                     'xc bottom [um]': xc_list_b,
                     'yc bottom [um]': yc_list_b}

        save_dict_to_csv(data_dict, directory+'/ellipse_approx (%s).csv'%(c+1))

        # estimate std for each cell
        std_t, std_b = estimate_ellipse_approx_std(c)
        # save again to add std estimation
        data_dict = {'sigma x [nN/um]': sigma_x_list,
                     'sigma y top [nN/um]': sigma_y_list_t,
                     'a top [um]': a_list_t,
                     'b top [um]': b_list_t,
                     'xc top [um]': xc_list_t,
                     'yc top [um]': yc_list_t,
                     'std top [um]': std_t,
                     'a bottom [um]': a_list_b,
                     'b bottom [um]': b_list_b,
                     'sigma y bottom [nN/um]': sigma_y_list_b,
                     'xc bottom [um]': xc_list_b,
                     'yc bottom [um]': yc_list_b,
                     'std bottom [um]': std_b}

        save_dict_to_csv(data_dict, directory +
                         '/ellipse_approx (%s).csv' % (c+1))

    print("Ellipse approximation terminated successfully!")
    return None


# run main_ellipse_approx
if do_ellipse_approx is True:
    main_ellipse_approx(directory)
else:
    print("main_ellipse_approx is not called!")


####################################################################################
# Step 5: given Ellipse approximation data as initial minimization values, fit ellipse to arc and minimize std 
# Input files needed, tangents (n).csv, sigma_contribution.mat,TEM_circles (n).csv,ellipse_approx (n).csv, Output ellipse_data_fit (n).csv

def main_ellipse_fit(directory):

    def estimate_ellipse_fit_std(c):

        fibertrack_path = directory + "fibertracking (%s).mat" % (c+1)
        mat = sio.loadmat(fibertrack_path)
        print(mat.keys())
        # first index gives position and second index gives frame/time
        x_data_bottom = np.array(mat['Xbottom'])
        y_data_bottom = np.array(mat['Ybottom'])

        # first index gives position and second index gives frame/time
        x_data_top = np.array(mat['Xtop'])
        y_data_top = np.array(mat['Ytop'])

        R_upper, R_lower, xc_upper, yc_upper, xc_lower, yc_lower = read_radii_center_um(
            directory, c)
        a_top, b_top, xc_top, yc_top, a_bottom, b_bottom, xc_bottom, yc_bottom = read_ellipse_center_fit_um(
            directory, c)

        a_u = a_top
        b_u = b_top

        a_l = a_bottom
        b_l = b_bottom

        std_u_list = []
        std_l_list = []
        for frame in range(noFrames):
            # first read in tracks for each frame
            # first index gives position and second index gives frame/time
            x_top_track = x_data_top[:, frame]*uM_per_pix
            y_top_track = y_data_top[:, frame]*uM_per_pix

            # first index gives position and second index gives frame/time
            x_bottom_track = x_data_bottom[:, frame]*uM_per_pix
            y_bottom_track = y_data_bottom[:, frame]*uM_per_pix

            # loop over all data pointa in one track
            dist_container_u = []
            dist_container_l = []

            dist_container_circle_u = []
            dist_container_circle_l = []
            # print("No of tracking points ", len(x_top_track), frame)
            # print("No of tracking points ", len(x_bottom_track), frame)
            for j in range(len(x_top_track)):
                dist_u = estimate_distance(
                    x_top_track[j], y_top_track[j], a_u[frame], b_u[frame], xc_top[frame], yc_top[frame], angle=0)
                dist_circle_u = ((xc_upper[frame]-x_top_track[j])**2 + (
                    yc_upper[frame]-y_top_track[j])**2)**0.5 - R_upper[frame]
                # store squared distance
                dist_container_u.append(dist_u**2)
                dist_container_circle_u.append(dist_circle_u**2)

            for j in range(len(x_bottom_track)):
                dist_l = estimate_distance(
                    x_bottom_track[j], y_bottom_track[j], a_l[frame], b_l[frame], xc_bottom[frame], yc_bottom[frame], angle=0)
                dist_circle_l = ((xc_lower[frame]-x_bottom_track[j])**2 + (
                    yc_lower[frame]-y_bottom_track[j])**2)**0.5 - R_lower[frame]

                # store squared distance
                dist_container_l.append(dist_l**2)
                dist_container_circle_l.append(dist_circle_l**2)

                std_u = np.sqrt(np.sum(dist_container_u)/len(dist_container_u))
                std_l = np.sqrt(np.sum(dist_container_l)/len(dist_container_l))

            # arrays of length 60 which have std of top/bottom ellipse for each frame
            std_u_list.append(std_u)
            std_l_list.append(std_l)

        return std_u_list, std_l_list

    def ellipe_tan_dot(rx, ry, px, py, theta):
        '''Dot product of the equation of the line formed by the point
           with another point on the ellipse's boundary and the tangent of the ellipse
           at that point on the boundary.
        '''
        return ((rx ** 2 - ry ** 2) * np.cos(theta) * np.sin(theta) - px * rx * np.sin(theta) + py * ry * np.cos(theta))

    def ellipe_tan_dot_derivative(rx, ry, px, py, theta):
        '''derivative of ellipe_tan_dot.
        '''
        return ((rx ** 2 - ry ** 2) * (np.cos(theta) ** 2 - np.sin(theta) ** 2) - px * rx * np.cos(theta) - py * ry * np.sin(theta))

    def estimate_distance(x, y, rx, ry, x0=0, y0=0, angle=0, error=1e-5):
        '''Given a point (x, y), and an ellipse with major - minor axis (rx, ry),
           its center at (x0, y0), and with a counter clockwise rotation of
           `angle` degrees, will return the distance between the ellipse and the
           closest point on the ellipses boundary.
        '''
        x -= x0
        y -= y0
        if angle:
            # rotate the points onto an ellipse whose rx, and ry lay on the x, y
            # axis
            angle = -pi / 180. * angle
            x, y = x * np.cos(angle) - y * np.sin(angle), x * \
                np.sin(angle) + y * np.cos(angle)

        theta = np.arctan2(rx * y, ry * x)
        while np.absolute(ellipe_tan_dot(rx, ry, x, y, theta)) > error:
            theta -= ellipe_tan_dot(rx, ry, x, y, theta) / \
                ellipe_tan_dot_derivative(rx, ry, x, y, theta)

        px, py = rx * np.cos(theta), ry * np.sin(theta)
        return ((x - px) ** 2 + (y - py) ** 2) ** .5

    # this function has a time out included to not get stucked in the while loop
    def estimate_distance_with_timeout(x, y, rx, ry, x0=0, y0=0, angle=0, error=1e-5):
        '''Given a point (x, y), and an ellipse with major - minor axis (rx, ry),
        its center at (x0, y0), and with a counter clockwise rotation of
        `angle` degrees, will return the distance between the ellipse and the
        closest point on the ellipses boundary.
        '''
        timeout = time.time() + 20*1
        x -= x0
        y -= y0
        if angle:
            # rotate the points onto an ellipse whose rx, and ry lay on the x, y
            # axis
            angle = -pi / 180. * angle
            x, y = x * np.cos(angle) - y * np.sin(angle), x * \
                np.sin(angle) + y * np.cos(angle)

        theta = np.arctan2(rx * y, ry * x)
        while np.absolute(ellipe_tan_dot(rx, ry, x, y, theta)) > error:
            theta -= ellipe_tan_dot(
                rx, ry, x, y, theta) / \
                ellipe_tan_dot_derivative(rx, ry, x, y, theta)
            if time.time() > timeout:
                break

        px, py = rx * np.cos(theta), ry * np.sin(theta)
        t_out_reached = False
        if time.time() < timeout:
            dist = ((x - px) ** 2 + (y - py) ** 2) ** .5
        else:
            dist = 0
            print("TIME OUT REACHED")

        return dist, t_out_reached

    def return_results(sigma_y, landa, phi, sigma_x, R, xc, yc, arc='top'):

        b_square = (landa/sigma_x)**2 * (1+(sigma_x/sigma_y) * np.tan(phi)**2)/(1+np.tan(phi)**2)
        a_square = (landa)**2/(sigma_x*sigma_y) * (1+(sigma_x/sigma_y)*np.tan(phi)**2)/(1+np.tan(phi)**2)
        b = np.sqrt(b_square)
        a = np.sqrt(a_square)
        sigma_isotrop = 0.5*(sigma_x+sigma_y)
        sigma_anisotrop_x = 0.5*(sigma_x-sigma_y)
        sigma_anisotrop_y = 0.5*(sigma_y-sigma_x)

        

        return a, b, sigma_isotrop, sigma_anisotrop_x, sigma_anisotrop_y
    
    sigma_xx = np.load(directory+'sigma_xx.npy')
    
    for c in range(cell_no_start_loop-1,noCells):
        print("CELL ", c+1)
        # define save container
        a_list_t = []
        b_list_t = []
        sigma_y_list_t = []
        xc_list_t = []
        yc_list_t = []

        a_list_b = []
        b_list_b = []
        sigma_y_list_b = []
        xc_list_b = []
        yc_list_b = []

        sigma_x_list = []

        (R_upper, R_lower,
         xc_upper, yc_upper,
         xc_lower, yc_lower,
         tx_top_left, ty_top_left,
         tx_top_right, ty_top_right,
         tx_bottom_right, ty_bottom_right,
         tx_bottom_left, ty_bottom_left) = read_radii_and_tangents_and_center_um(directory, c)

        # sigma_x_left, sigma_x_right = read_sigma_contribution_mat(directory, c)
        sigma_x_left = sigma_xx[:,c]
        sigma_x_right = sigma_xx[:,c]

        landa_top_left, landa_top_right, landa_bottom_left, landa_bottom_right = read_line_tension(directory, c)

        a_top_approx, b_top_approx, xc_top_approx, yc_top_approx, a_bottom_approx, b_bottom_approx, xc_bottom_approx, yc_bottom_approx, sigma_y_top_approx, sigma_y_bottom_approx, sigma_x_av = read_ellipse_center_um_sigma(
        directory, c)

        fibertrack_path = directory + "fibertracking (%s).mat" % (c+1)
        mat = sio.loadmat(fibertrack_path)
        print(mat.keys())
        # first index gives position and second index gives frame/time
        x_data_bottom = np.array(mat['Xbottom'])
        y_data_bottom = np.array(mat['Ybottom'])

        # first index gives position and second index gives frame/time
        x_data_top = np.array(mat['Xtop'])
        y_data_top = np.array(mat['Ytop'])

        for frame in np.arange(0, noFrames):
            print("FRAME ", frame)
            R_u = R_upper[frame]
            R_l = R_lower[frame]

            curvature_u = (1/R_u).item()
            curvature_l = (1/R_l).item()

            xc_u = xc_upper[frame]
            yc_u = yc_upper[frame]
            xc_l = xc_lower[frame]
            yc_l = yc_lower[frame]

            ty_tl = np.absolute(ty_top_left[frame])
            ty_tr = np.absolute(ty_top_right[frame])
            ty_bl = np.absolute(ty_bottom_left[frame])
            ty_br = np.absolute(ty_bottom_right[frame])

            ty_top = (ty_tl + ty_tr)/2
            ty_bottom = (ty_bl + ty_br)/2

            phi_tl = np.arcsin(ty_tl)
            phi_tr = np.arcsin(ty_tr)
            phi_bl = np.arcsin(ty_bl)
            phi_br = np.arcsin(ty_br)

            phi_top = (phi_tl + phi_tr)/2
            phi_bottom = (phi_bl + phi_br)/2

            landa_tl = np.absolute(landa_top_left[frame])
            landa_tr = np.absolute(landa_top_right[frame])
            landa_bl = np.absolute(landa_bottom_left[frame])
            landa_br = np.absolute(landa_bottom_right[frame])

            landa_top = (landa_tl + landa_tr)/2
            landa_bottom = (landa_bl + landa_br)/2

            sigma_x_l = sigma_x_left[frame]
            sigma_x_r = sigma_x_right[frame]

            # sigma_x_av = (sigma_x_l + sigma_x_r)/2

            # data from ellipse approximation necessary to set start values for fit
            sigma_y_top_estimate = sigma_y_top_approx[frame]
            sigma_y_bottom_estimate = sigma_y_bottom_approx[frame]
            sigma_x_estimate = sigma_x_av[frame]

            xc_top_ellipse_estimate = xc_top_approx[frame]
            yc_top_ellipse_estimate = yc_top_approx[frame]

            xc_bottom_ellipse_estimate = xc_bottom_approx[frame]
            yc_bottom_ellipse_estimate = yc_bottom_approx[frame]

            # first index gives position and second index gives frame/time
            x_top_track = x_data_top[:, frame]*uM_per_pix
            y_top_track = y_data_top[:, frame]*uM_per_pix

            # first index gives position and second index gives frame/time
            x_bottom_track = x_data_bottom[:, frame]*uM_per_pix
            y_bottom_track = y_data_bottom[:, frame]*uM_per_pix

            def dist_top(p):

                x = x_top_track
                y = y_top_track
                sigma_x = sigma_x_estimate
                phi = phi_top
                landa = landa_top  # nN

                sigma_y, xc, yc = p

                b = np.sqrt((landa/sigma_x)**2 * (1+(sigma_x/sigma_y)* np.tan(phi)**2)/(1+np.tan(phi)**2))
                a = np.sqrt((landa)**2/(sigma_x*sigma_y) *(1+(sigma_x/sigma_y)*np.tan(phi)**2)/(1+np.tan(phi)**2))

                dist_container_u = []

                # print("No of tracking points ", len(x_top_track), frame)
                for j in range(len(x_top_track)):
                    dist_u, t_out_reached = estimate_distance_with_timeout(
                    x_top_track[j], y_top_track[j], a, b, xc, yc, angle=0)
                    # print("dist_u ", dist_u, "Data point ", j, frame)

                    # store squared distance
                    dist_container_u.append(dist_u**2)

                return np.sum(dist_container_u)

            def dist_bottom(p):

                x = x_bottom_track
                y = y_bottom_track
                sigma_x = sigma_x_estimate
                phi = phi_bottom
                landa = landa_bottom  # nN

                sigma_y, xc, yc = p

                b = np.sqrt((landa/sigma_x)**2 * (1+(sigma_x/sigma_y)* np.tan(phi)**2)/(1+np.tan(phi)**2))
                a = np.sqrt((landa)**2/(sigma_x*sigma_y) * (1+(sigma_x/sigma_y)*np.tan(phi)**2)/(1+np.tan(phi)**2))

                dist_container_l = []
                # print("No of tracking points ", len(x_bottom_track), frame)
                for j in range(len(x_bottom_track)):
                    dist_l, t_out_reached = estimate_distance_with_timeout(
                        x_bottom_track[j], y_bottom_track[j], a, b, xc, yc, angle=0)
                    # print("dist_l ",dist_l,"Data point ",j,frame)

                    # store squared distance
                    dist_container_l.append(dist_l**2)

                return np.sum(dist_container_l)

            params0_top = [sigma_y_top_estimate, xc_top_ellipse_estimate, yc_top_ellipse_estimate]
            params0_bottom = [sigma_y_bottom_estimate, xc_bottom_ellipse_estimate, yc_bottom_ellipse_estimate]

            # perform the fit
            result_top = optimize.minimize(dist_top, params0_top, method='Nelder-Mead', options={'disp': True, 'adaptive': True})  # options={'maxiter': 10000, 'disp': True}
            result_bottom = optimize.minimize(dist_bottom, params0_bottom, method='Nelder-Mead', options={'disp': True, 'adaptive': True})

            # print(result.fit_report())
            sigma_y_top, xc_ellipse_top, yc_ellipse_top = result_top.x
            a_top, b_top, sigma_isotrop_top, sigma_anisotrop_x_top, sigma_anisotrop_y_top = return_results(sigma_y_top, landa_top, phi_top, sigma_x_estimate, R_u, xc_u, yc_u, arc='top')
            print('sigma_x ', sigma_x_estimate)
            print('sigma_y_top ', sigma_y_top)
            print('sigma_anisotrop_x_top ', sigma_anisotrop_x_top)
            print('sigma_anisotrop_y_top ', sigma_anisotrop_y_top)
            print('a_top ', a_top)
            print('b_top ', b_top)
            print("xc_ellipse_top ", xc_ellipse_top)
            print("yc_ellipse_top ", yc_ellipse_top)

            sigma_y_bottom, xc_ellipse_bottom, yc_ellipse_bottom = result_bottom.x
            a_bottom, b_bottom, sigma_isotrop_bottom, sigma_anisotrop_x_bottom, sigma_anisotrop_y_bottom = return_results(sigma_y_bottom, landa_bottom, phi_bottom, sigma_x_estimate, R_l, xc_l, yc_l, arc='bottom')
            print('sigma_x ', sigma_x_estimate)
            print('sigma_y_bottom ', sigma_y_bottom)
            print('sigma_anisotrop_x_bottom ', sigma_anisotrop_x_bottom)
            print('sigma_anisotrop_y_bottom ', sigma_anisotrop_y_bottom)
            print('a_bottom ', a_bottom)
            print('b_bottom ', b_bottom)
            print("xc_ellipse_bottom ", xc_ellipse_bottom)
            print("yc_ellipse_bottom ", yc_ellipse_bottom)

            a_list_t.append(a_top)
            b_list_t.append(b_top)
            sigma_y_list_t.append(sigma_y_top)
            xc_list_t.append(xc_ellipse_top)
            yc_list_t.append(yc_ellipse_top)

            a_list_b.append(a_bottom)
            b_list_b.append(b_bottom)
            sigma_y_list_b.append(sigma_y_bottom)
            xc_list_b.append(xc_ellipse_bottom)
            yc_list_b.append(yc_ellipse_bottom)

            sigma_x_list.append(sigma_x_estimate)

        data_dict = {'sigma x [nN/um]': sigma_x_list,
                     'sigma y top [nN/um]': sigma_y_list_t,
                     'a top [um]': a_list_t,
                     'b top [um]': b_list_t,
                     'xc top [um]': xc_list_t,
                     'yc top [um]': yc_list_t,
                     'a bottom [um]': a_list_b,
                     'b bottom [um]': b_list_b,
                     'sigma y bottom [nN/um]': sigma_y_list_b,
                     'xc bottom [um]': xc_list_b,
                     'yc bottom [um]': yc_list_b}
        save_dict_to_csv(data_dict, directory + '/ellipse_data_fit (%s).csv' % (c+1))
        # estimate std for each cell
        std_t, std_b = estimate_ellipse_fit_std(c)

        # save again to add std estimation
        data_dict = {'sigma x [nN/um]': sigma_x_list,
                     'sigma y top [nN/um]': sigma_y_list_t,
                     'a top [um]': a_list_t,
                     'b top [um]': b_list_t,
                     'xc top [um]': xc_list_t,
                     'yc top [um]': yc_list_t,
                     'std top [um]': std_t,
                     'a bottom [um]': a_list_b,
                     'b bottom [um]': b_list_b,
                     'sigma y bottom [nN/um]': sigma_y_list_b,
                     'xc bottom [um]': xc_list_b,
                     'yc bottom [um]': yc_list_b,
                     'std bottom [um]': std_b}


        save_dict_to_csv(data_dict, directory+'/ellipse_data_fit (%s).csv' % (c+1))


    return None


# run main_ellipse_fit
if do_ellipse_fit is True:
    main_ellipse_fit(directory)
else:
    print("main_ellipse_fit is not called!")
