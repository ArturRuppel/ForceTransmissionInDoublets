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
import pickle
import matplotlib.pyplot as plt
import math

''' Main Analysis Script for TFM-Opto Experiment for Cells on H-shaped Micropattern (Artur Ruppel, Dennis Wörthmüller)

    Copyright: Dennis Wörthmüller, Artur Ruppel
    Date: April 29, 2021  

    External python modules which are necessary: numpy, pandas scipy, lmfit

'''


# %%####################################################################################
# Step 1: fit circles to ellipses

def main_circle_fit(x_top, x_bottom, y_top, y_bottom):
    ''' Read fibertracking and fit circles to it using the hyper fit algorithm

        Input: directory where fibertrack (n).mat sits 
        Output: circle_fit (n).csv -> contains: 
    
    '''

    def fit_data_in_file(x_data, y_data, arc='left'):

        R_list = []
        error_list = []
        xc_list = []
        yc_list = []

        for frame in np.arange(0, x_data.shape[1]):
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

    xc_up = np.zeros((x_top.shape[1], x_top.shape[2]))  # initialize array with noFrames by noCells
    yc_up = np.zeros((x_top.shape[1], x_top.shape[2]))
    R_up = np.zeros((x_top.shape[1], x_top.shape[2]))
    error_up = np.zeros((x_top.shape[1], x_top.shape[2]))

    xc_lo = np.zeros((x_top.shape[1], x_top.shape[2]))
    yc_lo = np.zeros((x_top.shape[1], x_top.shape[2]))
    R_lo = np.zeros((x_top.shape[1], x_top.shape[2]))
    error_lo = np.zeros((x_top.shape[1], x_top.shape[2]))

    for c in range(x_top.shape[2]):
        # remove lines with 0 first
        x_top_current = x_top[:, :, c]
        x_bottom_current = x_bottom[:, :, c]
        y_top_current = y_top[:, :, c]
        y_bottom_current = y_bottom[:, :, c]

        x_top_current = x_top_current[~np.all(x_top_current == 0, axis=1)]
        x_bottom_current = x_bottom_current[~np.all(x_bottom_current == 0, axis=1)]
        y_top_current = y_top_current[~np.all(y_top_current == 0, axis=1)]
        y_bottom_current = y_bottom_current[~np.all(y_bottom_current == 0, axis=1)]

        # for each cell return fitted data (as list with entries meaning the frame) for all arcs of all frames 
        xc_up[:, c], yc_up[:, c], R_up[:, c], error_up[:, c] = fit_data_in_file(x_top_current, y_top_current, 'top')
        xc_lo[:, c], yc_lo[:, c], R_lo[:, c], error_lo[:, c] = fit_data_in_file(x_bottom_current, y_bottom_current, 'bottom')

    std = (error_up + error_lo) / 2
    std_baseline = np.nanmean(std[0:20, :], axis=0)
    # create data dictionary
    data_dict = {'Radius [px] (upper)': R_up,
                 'Radius Std [px] (upper)': error_up,
                 'x-pos center [px] (upper)': xc_up,
                 'y-pos center [px] (upper)': yc_up,
                 'Radius [px] (lower)': R_lo,
                 'Radius Std [px] (lower)': error_lo,
                 'x-pos center [px] (lower)': xc_lo,
                 'y-pos center [px] (lower)': yc_lo,
                 'std [px]': std,
                 'std_baseline [px]': std_baseline}

    print("Circle Fit terminated successfully!")
    return data_dict


# %% Step 2: calculate tangents of circles at fibertrack start and end point


def main_calc_tangents(x_top, x_bottom, y_top, y_bottom,
                       XC_upper_all, YC_upper_all, XC_lower_all, YC_lower_all, R_upper_all, R_lower_all):
    ''' Calculate tangents at start and end point of fiber track

        Input: fibertracks and circlefits
        Output: tangents
    
    '''

    # tl = top left, other "pos" are tr, bl, br
    # def calc_tangent(X, Y, XC, YC, R, pos='tl'):
    #     '''caluclate the tangents of point (X,Y) but also return closest point on circle (x,y)'''
    #     theta = np.abs(np.arctan((Y-YC)/(X-XC)))

    #     if pos == 'tr' or pos == 'br':
    #         theta+=np.pi/2

    #     x = XC + R*np.cos(theta)
    #     y = YC + R*np.sin(theta)
    #     tx = np.cos(theta)
    #     ty = np.sin(theta)

    #     if pos == 'tl' or pos == 'tr':
    #         ty = -ty

    #     return tx, ty, x, y

    def calc_tangent(X, Y, XC, YC, R, pos='tl'):
        '''caluclate the tangents of point (X,Y) but also return closest point on circle (x,y)'''
        theta = np.arctan((Y - YC) / (X - XC))

        # if pos == 'tr' or pos == 'br':
        #     theta = theta + np.pi
        # # elif pos == 'tl':
        # #     theta = np.pi+theta
        # theta_degree = theta*180/np.pi

        # x = XC + R*np.cos(theta)
        # y = YC + R*np.sin(theta)
        # tx = np.cos(theta)
        # ty = np.sin(theta)

        # # if pos == 'tl' or pos == 'tr':
        # #     ty = -ty

        if pos == 'bl':
            theta = theta + np.pi
        elif pos == 'tl':
            theta = np.pi + theta

        x = XC + R * np.cos(theta)
        y = YC + R * np.sin(theta)
        tx = -np.sin(theta)
        ty = np.cos(theta)

        if pos == 'tl':
            theta = theta + np.pi
            tx = -tx
            ty = -ty

        elif pos == 'br':
            theta = np.pi + theta
            tx = -tx
            ty = -ty

        return tx, ty, x, y

    tx_top_left = np.zeros((x_top.shape[1], x_top.shape[2]))  # initialize array with noFrames by noCells
    ty_top_left = np.zeros((x_top.shape[1], x_top.shape[2]))
    x_top_left = np.zeros((x_top.shape[1], x_top.shape[2]))
    y_top_left = np.zeros((x_top.shape[1], x_top.shape[2]))

    tx_top_right = np.zeros((x_top.shape[1], x_top.shape[2]))  # initialize array with noFrames by noCells
    ty_top_right = np.zeros((x_top.shape[1], x_top.shape[2]))
    x_top_right = np.zeros((x_top.shape[1], x_top.shape[2]))
    y_top_right = np.zeros((x_top.shape[1], x_top.shape[2]))

    tx_bottom_left = np.zeros((x_top.shape[1], x_top.shape[2]))  # initialize array with noFrames by noCells
    ty_bottom_left = np.zeros((x_top.shape[1], x_top.shape[2]))
    x_bottom_left = np.zeros((x_top.shape[1], x_top.shape[2]))
    y_bottom_left = np.zeros((x_top.shape[1], x_top.shape[2]))

    tx_bottom_right = np.zeros((x_top.shape[1], x_top.shape[2]))  # initialize array with noFrames by noCells
    ty_bottom_right = np.zeros((x_top.shape[1], x_top.shape[2]))
    x_bottom_right = np.zeros((x_top.shape[1], x_top.shape[2]))
    y_bottom_right = np.zeros((x_top.shape[1], x_top.shape[2]))

    for c in range(x_top.shape[2]):
        x_top_current = x_top[:, :, c]
        x_bottom_current = x_bottom[:, :, c]
        y_top_current = y_top[:, :, c]
        y_bottom_current = y_bottom[:, :, c]

        # remove lines with 0
        x_top_current = x_top_current[~np.all(x_top_current == 0, axis=1)]
        x_bottom_current = x_bottom_current[~np.all(x_bottom_current == 0, axis=1)]
        y_top_current = y_top_current[~np.all(y_top_current == 0, axis=1)]
        y_bottom_current = y_bottom_current[~np.all(y_bottom_current == 0, axis=1)]

        # get end points for tangent calculation
        X_top_left = x_top_current[0, :]
        X_top_right = x_top_current[-1, :]
        X_bottom_left = x_bottom_current[0, :]
        X_bottom_right = x_bottom_current[-1, :]

        Y_top_left = y_top_current[0, :]
        Y_top_right = y_top_current[-1, :]
        Y_bottom_left = y_bottom_current[0, :]
        Y_bottom_right = y_bottom_current[-1, :]

        XC_upper = XC_upper_all[:, c]
        YC_upper = YC_upper_all[:, c]
        XC_lower = XC_lower_all[:, c]
        YC_lower = YC_lower_all[:, c]
        R_upper = R_upper_all[:, c]
        R_lower = R_lower_all[:, c]

        # tangents top left
        tx_top_left[:, c], ty_top_left[:, c], x_top_left[:, c], y_top_left[:, c] = calc_tangent(X_top_left, Y_top_left, XC_upper, YC_upper,
                                                                                                R_upper, pos='tl')
        # tangents top right
        tx_top_right[:, c], ty_top_right[:, c], x_top_right[:, c], y_top_right[:, c] = calc_tangent(X_top_right, Y_top_right, XC_upper,
                                                                                                    YC_upper, R_upper, pos='tr')
        # tangents bottom left
        tx_bottom_left[:, c], ty_bottom_left[:, c], x_bottom_left[:, c], y_bottom_left[:, c] = calc_tangent(X_bottom_left, Y_bottom_left,
                                                                                                            XC_lower, YC_lower, R_lower,
                                                                                                            pos='bl')
        # tangents bottom right
        tx_bottom_right[:, c], ty_bottom_right[:, c], x_bottom_right[:, c], y_bottom_right[:, c] = calc_tangent(X_bottom_right,
                                                                                                                Y_bottom_right, XC_lower,
                                                                                                                YC_lower, R_lower, pos='br')

    # define data dictionary for each cell
    data_dict = {'tx top left': tx_top_left,
                 'ty top left': ty_top_left,
                 'tx top right': tx_top_right,
                 'ty top right': ty_top_right,
                 'tx bottom left': tx_bottom_left,
                 'ty bottom left': ty_bottom_left,
                 'tx bottom right': tx_bottom_right,
                 'ty bottom right': ty_bottom_right,
                 'xTouch top left': x_top_left,
                 'yTouch top left': y_top_left,
                 'xTouch top right': x_top_right,
                 'yTouch top right': y_top_right,
                 'xTouch bottom left': x_bottom_left,
                 'yTouch bottom left': y_bottom_left,
                 'xTouch bottom right': x_bottom_right,
                 'yTouch bottom right': y_bottom_right}

    print("Calculating tangents terminated successfully!")
    return data_dict


# %%###################################################################################
# Step 3: perform a tfm analysis based on the circle fits 

def main_TEM_circles(R_upper, R_lower, tx_top_left, ty_top_left, tx_top_right, ty_top_right,
                     tx_bottom_right, ty_bottom_right, tx_bottom_left, ty_bottom_left,
                     Fx_top_left, Fy_top_left, Fx_top_right, Fy_top_right, Fx_bottom_right, Fy_bottom_right, Fx_bottom_left, Fy_bottom_left,
                     pixelsize):
    '''Desciption'''

    # calculate forces coming from free (landa) and adherent fiber (f)
    def calc_landa_and_f(tx, ty, Fx, Fy):
        # landa = Fx + Fy*np.abs(tx)
        # f = Fy*np.abs(ty)
        # return landa, f
        landa = Fx / tx
        f = Fy - Fx * ty / tx
        return landa, f

    landa_top_left = np.zeros((R_upper.shape[0], R_upper.shape[1]))  # initialize array with noFrames by noCells
    landa_top_right = np.zeros((R_upper.shape[0], R_upper.shape[1]))
    landa_bottom_left = np.zeros((R_upper.shape[0], R_upper.shape[1]))
    landa_bottom_right = np.zeros((R_upper.shape[0], R_upper.shape[1]))

    f_top_left = np.zeros((R_upper.shape[0], R_upper.shape[1]))
    f_top_right = np.zeros((R_upper.shape[0], R_upper.shape[1]))
    f_bottom_left = np.zeros((R_upper.shape[0], R_upper.shape[1]))
    f_bottom_right = np.zeros((R_upper.shape[0], R_upper.shape[1]))

    # sigma_top_left = np.zeros((R_upper.shape[0],R_upper.shape[1]))
    # sigma_top_right = np.zeros((R_upper.shape[0],R_upper.shape[1]))
    # sigma_bottom_left = np.zeros((R_upper.shape[0],R_upper.shape[1]))
    # sigma_bottom_right = np.zeros((R_upper.shape[0],R_upper.shape[1]))

    for c in range(R_upper.shape[1]):
        landa_top_left[:, c], f_top_left[:, c] = calc_landa_and_f(tx_top_left[:, c], ty_top_left[:, c], Fx_top_left[:, c],
                                                                  Fy_top_left[:, c])  # in N
        landa_top_right[:, c], f_top_right[:, c] = calc_landa_and_f(tx_top_right[:, c], ty_top_right[:, c], Fx_top_right[:, c],
                                                                    Fy_top_right[:, c])
        landa_bottom_left[:, c], f_bottom_left[:, c] = calc_landa_and_f(tx_bottom_left[:, c], ty_bottom_left[:, c], Fx_bottom_left[:, c],
                                                                        Fy_bottom_left[:, c])
        landa_bottom_right[:, c], f_bottom_right[:, c] = calc_landa_and_f(tx_bottom_right[:, c], ty_bottom_right[:, c],
                                                                          Fx_bottom_right[:, c], Fy_bottom_right[:, c])

        # sigma_top_left[:,c] = landa_top_left[:,c]/(R_upper[:,c]*pixelsize)  # convert radius to m to obtain N/m
        # sigma_top_right[:,c] = -landa_top_right[:,c]/(R_upper[:,c]*pixelsize) 
        # sigma_bottom_left[:,c] = landa_bottom_left[:,c]/(R_lower[:,c]*pixelsize) 
        # sigma_bottom_right[:,c] = -landa_bottom_right[:,c]/(R_lower[:,c]*pixelsize) 

    # # make some calculations
    # f_adherent_baseline = np.nanmean((f_top_left[0:20,:]+f_top_right[0:20,:]-f_bottom_left[0:20,:]-f_bottom_right[0:20,:])/4,axis=0)
    # landa_baseline = np.nanmean((landa_top_left[0:20,:]+landa_top_right[0:20,:]+landa_bottom_left[0:20,:]+landa_bottom_right[0:20,:])/4,axis=0)
    # f_adherent_baseline = np.nanmean(f_bottom_left[0:20,:],axis=0)
    # landa_baseline = np.nanmean(f_bottom_left[0:20,:],axis=0)

    data_dict = {'line tension top left [nN]': landa_top_left,
                 'f top left [nN]': f_top_left,
                 'line tension top right [nN]': landa_top_right,
                 'f top right [nN]': f_top_right,
                 'line tension bottom left [nN]': landa_bottom_left,
                 'f bottom left [nN]': f_bottom_left,
                 'line tension bottom right [nN]': landa_bottom_right,
                 'f bottom right [nN]': f_bottom_right, }
    # 'f bottom right': f_bottom_right,
    # 'surface tension top left': sigma_top_left,
    # 'surface tension top right': sigma_top_right,
    # 'surface tension bottom left': sigma_bottom_left,
    # 'surface tension bottom right': sigma_bottom_right}

    print("TEM circle analysis terminated successfully!")
    return data_dict


# %%###################################################################################
# Step 4: given TEM_circle data and sigma_x contribution estimate ellipse based on curvature of circle

def main_ellipse_approx(x_top, x_bottom, y_top, y_bottom,
                        R_upper, R_lower,
                        xc_upper, yc_upper, xc_lower, yc_lower,
                        tx_top_left, ty_top_left, tx_top_right, ty_top_right,
                        tx_bottom_right, ty_bottom_right, tx_bottom_left, ty_bottom_left,
                        landa_top_left, landa_top_right, landa_bottom_left, landa_bottom_right,
                        sigma_x, pixelsize, folder):
    '''Description'''

    def return_results(sigma_y, landa, phi, sigma_x, R, xc, yc, arc='top'):

        b_square = (landa / sigma_x) ** 2 * (1 + (sigma_x / sigma_y) * np.tan(phi) ** 2) / (1 + np.tan(phi) ** 2)
        a_square = (landa) ** 2 / (sigma_x * sigma_y) * (1 + (sigma_x / sigma_y) * np.tan(phi) ** 2) / (1 + np.tan(phi) ** 2)
        b = np.sqrt(b_square)
        a = np.sqrt(a_square)
        sigma_isotrop = 0.5 * (sigma_x + sigma_y)
        sigma_anisotrop_x = 0.5 * (sigma_x - sigma_y)
        sigma_anisotrop_y = 0.5 * (sigma_y - sigma_x)

        delta = np.absolute(b - R)
        xc_ellipse = xc
        if arc == 'top':
            if b > R:
                yc_ellipse = yc - delta
            elif b <= R:
                yc_ellipse = yc + delta
        elif arc == 'bottom':
            if b > R:
                yc_ellipse = yc + delta
            elif b <= R:
                yc_ellipse = yc - delta

        return a, b, sigma_isotrop, sigma_anisotrop_x, sigma_anisotrop_y, xc_ellipse, yc_ellipse

    def estimate_ellipse_approx_std(x_top, x_bottom, y_top, y_bottom,
                                    R_upper, R_lower, xc_upper, yc_upper, xc_lower, yc_lower,
                                    a_top, b_top, xc_top, yc_top, a_bottom, b_bottom, xc_bottom, yc_bottom,
                                    pixelsize):

        a_u = a_top
        b_u = b_top

        a_l = a_bottom
        b_l = b_bottom

        std_u_list = []
        std_l_list = []
        for frame in range(R_upper.shape[0]):
            # first read in tracks for each frame
            # first index gives position and second index gives frame/time
            x_top_track = x_top[:, frame]
            y_top_track = y_top[:, frame]

            # first index gives position and second index gives frame/time
            x_bottom_track = x_bottom[:, frame]
            y_bottom_track = y_bottom[:, frame]

            # loop over all data pointa in one track
            dist_container_u = []
            dist_container_l = []

            dist_container_circle_u = []
            dist_container_circle_l = []
            # print("No of tracking points ", len(x_top_track), frame)
            # print("No of tracking points ", len(x_bottom_track), frame)
            for j in range(len(x_top_track)):
                # dist_u = estimate_distance(x_top_track[j], y_top_track[j], a_u[frame], b_u[frame], xc_top[frame], yc_top[frame], angle=0)
                # dist_circle_u = ((xc_upper[frame]-x_top_track[j])**2 +(yc_upper[frame]-y_top_track[j])**2)**0.5 - R_upper[frame]

                dist_u = estimate_distance_new(a_u[frame], b_u[frame], (x_top_track[j] - xc_top[frame], y_top_track[j] - yc_top[frame]))
                # store squared distance
                dist_container_u.append(dist_u ** 2)
                # dist_container_circle_u.append(dist_circle_u**2)

            for j in range(len(x_bottom_track)):
                # dist_l = estimate_distance(x_bottom_track[j], y_bottom_track[j], a_l[frame], b_l[frame], xc_bottom[frame], yc_bottom[frame], angle=0)
                # dist_circle_l = ((xc_lower[frame]-x_bottom_track[j])**2 + (yc_lower[frame]-y_bottom_track[j])**2)**0.5 - R_lower[frame]

                dist_l = estimate_distance_new(a_l[frame], b_l[frame],
                                               (x_bottom_track[j] - xc_bottom[frame], y_bottom_track[j] - yc_bottom[frame]))
                # store squared distance
                dist_container_l.append(dist_l ** 2)
                # dist_container_circle_l.append(dist_circle_l**2)

                std_u = np.sqrt(np.sum(dist_container_u) / len(dist_container_u))
                std_l = np.sqrt(np.sum(dist_container_l) / len(dist_container_l))

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

    # def estimate_distance(x, y, rx, ry, x0=0, y0=0, angle=0, error=1e-5):
    #     '''Given a point (x, y), and an ellipse with major - minor axis (rx, ry),
    #        its center at (x0, y0), and with a counter clockwise rotation of
    #        `angle` degrees, will return the distance between the ellipse and the
    #        closest point on the ellipses boundary.
    #     '''
    #     x -= x0
    #     y -= y0
    #     if angle:
    #         # rotate the points onto an ellipse whose rx, and ry lay on the x, y
    #         # axis
    #         angle = -np.pi / 180. * angle
    #         x, y = x * np.cos(angle) - y * np.sin(angle), x * np.sin(angle) + y * np.cos(angle)

    #     theta = np.arctan2(rx * y, ry * x)
    #     while np.absolute(ellipe_tan_dot(rx, ry, x, y, theta)) > error:
    #         theta -= ellipe_tan_dot(rx, ry, x, y, theta) / ellipe_tan_dot_derivative(rx, ry, x, y, theta)

    #     px, py = rx * np.cos(theta), ry * np.sin(theta)
    #     return ((x - px) ** 2 + (y - py) ** 2) ** .5

    def estimate_distance_new(semi_major, semi_minor, p):
        '''Given a point p, and an ellipse with major - minor axis (semi_major, semi_minor),
        will return the distance between the ellipse and the
        closest point on the ellipses boundary. code was taken from https://wet-robots.ghost.io/simple-method-for-distance-to-ellipse/
        '''
        px = abs(p[0])
        py = abs(p[1])

        t = math.pi / 4

        a = semi_major
        b = semi_minor

        for x in range(0, 3):
            x = a * math.cos(t)
            y = b * math.sin(t)

            ex = (a * a - b * b) * math.cos(t) ** 3 / a
            ey = (b * b - a * a) * math.sin(t) ** 3 / b

            rx = x - ex
            ry = y - ey

            qx = px - ex
            qy = py - ey

            r = math.hypot(ry, rx)
            q = math.hypot(qy, qx)

            delta_c = r * math.asin((rx * qy - ry * qx) / (r * q))
            delta_t = delta_c / math.sqrt(a * a + b * b - x * x - y * y)

            t += delta_t
            t = min(math.pi / 2, max(0, t))
        p_ellipse = (math.copysign(x, p[0]), math.copysign(y, p[1]))
        distance = math.sqrt(((p_ellipse[0] - p[0]) ** 2) + ((p_ellipse[1] - p[1]) ** 2))
        return distance

    theta_max = 15  # in degree
    theta_max_rad = theta_max / 180 * np.pi

    # initialize arrays with noFrames by noCells
    a_top_all = np.zeros((R_upper.shape[0], R_upper.shape[1]))
    b_top_all = np.zeros((R_upper.shape[0], R_upper.shape[1]))
    sigma_y_top_all = np.zeros((R_upper.shape[0], R_upper.shape[1]))
    xc_top_all = np.zeros((R_upper.shape[0], R_upper.shape[1]))
    yc_top_all = np.zeros((R_upper.shape[0], R_upper.shape[1]))
    std_top_all = np.zeros((R_upper.shape[0], R_upper.shape[1]))

    a_bottom_all = np.zeros((R_upper.shape[0], R_upper.shape[1]))
    b_bottom_all = np.zeros((R_upper.shape[0], R_upper.shape[1]))
    sigma_y_bottom_all = np.zeros((R_upper.shape[0], R_upper.shape[1]))
    xc_bottom_all = np.zeros((R_upper.shape[0], R_upper.shape[1]))
    yc_bottom_all = np.zeros((R_upper.shape[0], R_upper.shape[1]))
    std_bottom_all = np.zeros((R_upper.shape[0], R_upper.shape[1]))

    for c in range(R_upper.shape[1]):
        print("Ellipse approximation: cell " + str(c + 1))
        x_data_top = x_top[:, :, c]
        x_data_bottom = x_bottom[:, :, c]
        y_data_top = y_top[:, :, c]
        y_data_bottom = y_bottom[:, :, c]
        # remove lines with 0 first
        x_data_top = x_data_top[~np.all(x_data_top == 0, axis=1)] * pixelsize  # convert to µm
        y_data_top = y_data_top[~np.all(y_data_top == 0, axis=1)] * pixelsize
        x_data_bottom = x_data_bottom[~np.all(x_data_bottom == 0, axis=1)] * pixelsize
        y_data_bottom = y_data_bottom[~np.all(y_data_bottom == 0, axis=1)] * pixelsize

        for frame in np.arange(R_upper.shape[0]):
            R_u = R_upper[frame, c] * pixelsize  # convert to µm
            R_l = R_lower[frame, c] * pixelsize

            curvature_u = (1 / R_u).item()
            curvature_l = (1 / R_l).item()

            xc_u = xc_upper[frame, c] * pixelsize
            yc_u = yc_upper[frame, c] * pixelsize
            xc_l = xc_lower[frame, c] * pixelsize
            yc_l = yc_lower[frame, c] * pixelsize

            ty_tl = np.absolute(ty_top_left[frame, c])
            ty_tr = np.absolute(ty_top_right[frame, c])
            ty_bl = np.absolute(ty_bottom_left[frame, c])
            ty_br = np.absolute(ty_bottom_right[frame, c])

            ty_top = (ty_tl + ty_tr) / 2
            ty_bottom = (ty_bl + ty_br) / 2

            phi_tl = np.arcsin(ty_tl)
            phi_tr = np.arcsin(ty_tr)
            phi_bl = np.arcsin(ty_bl)
            phi_br = np.arcsin(ty_br)

            phi_top = (phi_tl + phi_tr) / 2
            phi_bottom = (phi_bl + phi_br) / 2

            landa_tl = np.absolute(landa_top_left[frame, c])
            landa_tr = np.absolute(landa_top_right[frame, c])
            landa_bl = np.absolute(landa_bottom_left[frame, c])
            landa_br = np.absolute(landa_bottom_right[frame, c])

            landa_top = (landa_tl + landa_tr) / 2
            landa_bottom = (landa_bl + landa_br) / 2

            sigma_x_av = sigma_x[frame, c]

            def k_top(x, sigma_y):
                phi = phi_top
                landa = landa_top  # N
                sigma_x = sigma_x_av  # N/m
                # print("phi ", phi)
                # print("landa ", landa)
                # print("sigma_x ", sigma_x)
                b_square = (landa / sigma_x) ** 2 * (1 + (sigma_x / sigma_y) * np.tan(phi) ** 2) / (1 + np.tan(phi) ** 2)
                a_square = (landa) ** 2 / (sigma_x * sigma_y) * (1 + (sigma_x / sigma_y) * np.tan(phi) ** 2) / (1 + np.tan(phi) ** 2)
                a = np.sqrt(a_square)
                b = np.sqrt(b_square)
                # print("a ", a)
                # print("b ", b)
                # returns the curvature over x range
                return a * b / (a ** 2 * np.sin(x) ** 2 + b ** 2 * np.cos(x) ** 2) ** (3 / 2)

            def k_bottom(x, sigma_y):
                phi = phi_bottom
                landa = landa_bottom  # N
                sigma_x = sigma_x_av  # N/m
                b_square = (landa / sigma_x) ** 2 * (1 + (sigma_x / sigma_y) * np.tan(phi) ** 2) / (1 + np.tan(phi) ** 2)
                a_square = (landa) ** 2 / (sigma_x * sigma_y) * (1 + (sigma_x / sigma_y) * np.tan(phi) ** 2) / (1 + np.tan(phi) ** 2)
                a = np.sqrt(a_square)
                b = np.sqrt(b_square)
                # print("a ", a)
                # print("b ",b)
                # returns the curvature over x range
                return a * b / (a ** 2 * np.sin(x) ** 2 + b ** 2 * np.cos(x) ** 2) ** (3 / 2)

            x = np.linspace(np.pi / 2 - (theta_max_rad / np.pi) * np.pi, np.pi / 2 + (theta_max_rad / np.pi) * np.pi, 1000)

            model_top = Model(k_top, independent_vars=['x'], nan_policy='omit')

            model_bottom = Model(k_bottom, independent_vars=['x'], nan_policy='omit')

            params_top = model_top.make_params(sigma_y=landa_top / R_u)
            params_bottom = model_bottom.make_params(sigma_y=landa_bottom / R_l)

            # print(model_top.param_names, model_top.independent_vars)

            # params.add(name="factor", min=0.0, max=1)
            # model.print_param_hints(colwidth=8)

            # perform the fit
            result_top = model_top.fit(curvature_u, params_top, x=x, method='leastsq')
            result_bottom = model_bottom.fit(curvature_l, params_bottom, x=x, method='leastsq')

            # print(result.fit_report())
            sigma_y_top = result_top.params['sigma_y'].value
            a_top, b_top, sigma_isotrop_top, sigma_anisotrop_x_top, sigma_anisotrop_y_top, xc_top, yc_top = return_results(sigma_y_top,
                                                                                                                           landa_top,
                                                                                                                           phi_top,
                                                                                                                           sigma_x_av, R_u,
                                                                                                                           xc_u, yc_u,
                                                                                                                           arc='top')

            sigma_y_bottom = result_bottom.params['sigma_y'].value
            a_bottom, b_bottom, sigma_isotrop_bottom, sigma_anisotrop_x_bottom, sigma_anisotrop_y_bottom, xc_bottom, yc_bottom = return_results(
                sigma_y_bottom, landa_bottom, phi_bottom, sigma_x_av, R_l, xc_l, yc_l, arc='bottom')

            a_top_all[frame, c] = a_top
            b_top_all[frame, c] = b_top
            sigma_y_top_all[frame, c] = sigma_y_top
            xc_top_all[frame, c] = xc_top
            yc_top_all[frame, c] = yc_top

            a_bottom_all[frame, c] = a_bottom
            b_bottom_all[frame, c] = b_bottom
            sigma_y_bottom_all[frame, c] = sigma_y_bottom
            xc_bottom_all[frame, c] = xc_bottom
            yc_bottom_all[frame, c] = yc_bottom

            if frame == 1:
                # plot ellipses and data for quality check
                t = np.linspace(0, 2 * np.pi, 100)
                plt.figure(figsize=(5, 5))

                plt.plot(xc_top + a_top * np.cos(t), yc_top + b_top * np.sin(t))
                plt.plot(x_data_top[:, frame], y_data_top[:, frame], 'bo')
                plt.grid(color='lightgray', linestyle='--')

                plt.plot(xc_bottom + a_bottom * np.cos(t), yc_bottom + b_bottom * np.sin(t))
                plt.plot(x_data_bottom[:, frame], y_data_bottom[:, frame], 'bo')
                plt.grid(color='lightgray', linestyle='--')

                plt.xlim((0, 100))
                plt.ylim((0, 100))

                if not os.path.exists(str.replace(folder, '.dat', '')):
                    os.mkdir(str.replace(folder, '.dat', ''))

                savefolder = str.replace(folder, '.dat', '/ellipseapproximation')

                if not os.path.exists(savefolder):
                    os.mkdir(savefolder)

                plt.savefig(savefolder + '/cell' + str(c) + '.png')
                plt.close()

        # estimate std for each cell
        std_t, std_b = estimate_ellipse_approx_std(x_data_top, x_data_bottom, y_data_top, y_data_bottom,
                                                   R_upper[:, c], R_lower[:, c], xc_upper[:, c], yc_upper[:, c], xc_lower[:, c],
                                                   yc_lower[:, c],
                                                   a_top_all[:, c], b_top_all[:, c], xc_top_all[:, c], yc_top_all[:, c],
                                                   a_bottom_all[:, c], b_bottom_all[:, c], xc_bottom_all[:, c], yc_bottom_all[:, c],
                                                   pixelsize)
        std_top_all[:, c] = std_t
        std_bottom_all[:, c] = std_b

    # do some calculations
    std = (std_bottom_all + std_bottom_all) / 2
    std_baseline = np.nanmean(std[0:20, :], axis=0)

    data_dict = {'sigma x [nN/µm]': sigma_x,
                 'sigma y top [nN/µm]': sigma_y_top_all,
                 'a top [µm]': a_top_all,
                 'b top [µm]': b_top_all,
                 'xc top [µm]': xc_top_all,
                 'yc top [µm]': yc_top_all,
                 'std top [µm]': std_top_all,
                 'sigma y bottom [nN/µm]': sigma_y_bottom_all,
                 'a bottom [µm]': a_bottom_all,
                 'b bottom [µm]': b_bottom_all,
                 'xc bottom [µm]': xc_bottom_all,
                 'yc bottom [µm]': yc_bottom_all,
                 'std bottom [µm]': std_bottom_all,
                 'std [µm]': std,
                 'std_baseline [µm]': std_baseline}

    print("Ellipse approximation terminated successfully!")
    return data_dict


# %%###################################################################################
# Step 5: given Ellipse approximation data as initial minimization values, fit ellipse to arc and minimize std 
# Input files needed, tangents (n).csv, sigma_contribution.mat,TEM_circles (n).csv,ellipse_approx (n).csv, Output ellipse_data_fit (n).csv

def main_ellipse_fit(R_upper_all, R_lower_all, XC_upper_all, YC_upper_all, XC_lower_all, YC_lower_all,
                     tx_top_left_all, ty_top_left_all,
                     tx_top_right_all, ty_top_right_all,
                     tx_bottom_right_all, ty_bottom_right_all,
                     tx_bottom_left_all, ty_bottom_left_all,
                     sigma_x_all,
                     landa_top_left_all, landa_top_right_all, landa_bottom_left_all, landa_bottom_right_all,
                     x_data_bottom_all, y_data_bottom_all, x_data_top_all, y_data_top_all,
                     a_top_approx_all, b_top_approx_all, xc_top_approx_all, yc_top_approx_all,
                     a_bottom_approx_all, b_bottom_approx_all, xc_bottom_approx_all, yc_bottom_approx_all,
                     sigma_y_all, folder):
    def estimate_ellipse_fit_std(x_data_top, y_data_top, x_data_bottom, y_data_bottom,
                                 R_upper, R_lower, xc_upper, yc_upper, xc_lower, yc_lower,
                                 a_top, b_top, xc_top, yc_top, a_bottom, b_bottom, xc_bottom, yc_bottom,
                                 pixelsize):

        a_u = a_top
        b_u = b_top

        a_l = a_bottom
        b_l = b_bottom

        std_u_list = []
        std_l_list = []
        for frame in range(R_upper.shape[0]):
            # first read in tracks for each frame
            # first index gives position and second index gives frame/time
            x_top_track = x_data_top[:, frame] * pixelsize
            y_top_track = y_data_top[:, frame] * pixelsize

            # first index gives position and second index gives frame/time
            x_bottom_track = x_data_bottom[:, frame] * pixelsize
            y_bottom_track = y_data_bottom[:, frame] * pixelsize

            # loop over all data pointa in one track
            dist_container_u = []
            dist_container_l = []

            # dist_container_circle_u = []
            # dist_container_circle_l = []
            # print("No of tracking points ", len(x_top_track), frame)
            # print("No of tracking points ", len(x_bottom_track), frame)
            for j in range(len(x_top_track)):
                # dist_u = estimate_distance(
                #     x_top_track[j], y_top_track[j], a_u[frame], b_u[frame], xc_top[frame], yc_top[frame], angle=0)
                dist_u = estimate_distance_new(a_u[frame], b_u[frame], (x_top_track[j] - xc_top[frame], y_top_track[j] - yc_top[frame]))
                # dist_circle_u = ((xc_upper[frame]-x_top_track[j])**2 + (
                #     yc_upper[frame]-y_top_track[j])**2)**0.5 - R_upper[frame]
                # store squared distance
                dist_container_u.append(dist_u ** 2)
                # dist_container_circle_u.append(dist_circle_u**2)

            for j in range(len(x_bottom_track)):
                # dist_l = estimate_distance(
                #     x_bottom_track[j], y_bottom_track[j], a_l[frame], b_l[frame], xc_bottom[frame], yc_bottom[frame], angle=0)
                dist_l = estimate_distance_new(a_l[frame], b_l[frame],
                                               (x_bottom_track[j] - xc_bottom[frame], y_bottom_track[j] - yc_bottom[frame]))
                # dist_circle_l = ((xc_lower[frame]-x_bottom_track[j])**2 + (
                #     yc_lower[frame]-y_bottom_track[j])**2)**0.5 - R_lower[frame]

                # store squared distance
                dist_container_l.append(dist_l ** 2)
                # dist_container_circle_l.append(dist_circle_l**2)

                std_u = np.sqrt(np.sum(dist_container_u) / len(dist_container_u))
                std_l = np.sqrt(np.sum(dist_container_l) / len(dist_container_l))

            # arrays of length 60 which have std of top/bottom ellipse for each frame
            std_u_list.append(std_u)
            std_l_list.append(std_l)

        return std_u_list, std_l_list

    def ellipe_tan_dot(rx, ry, px, py, theta):
        # Dot product of the equation of the line formed by the point
        # with another point on the ellipse's boundary and the tangent of the ellipse
        # at that point on the boundary.

        return ((rx ** 2 - ry ** 2) * np.cos(theta) * np.sin(theta) - px * rx * np.sin(theta) + py * ry * np.cos(theta))

    def ellipe_tan_dot_derivative(rx, ry, px, py, theta):
        # derivative of ellipse_tan_dot.

        return ((rx ** 2 - ry ** 2) * (np.cos(theta) ** 2 - np.sin(theta) ** 2) - px * rx * np.cos(theta) - py * ry * np.sin(theta))

    # def estimate_distance(x, y, rx, ry, x0=0, y0=0, angle=0, error=1e-5):
    #     # '''Given a point (x, y), and an ellipse with major - minor axis (rx, ry),
    #     #    its center at (x0, y0), and with a counter clockwise rotation of
    #     #    `angle` degrees, will return the distance between the ellipse and the
    #     #    closest point on the ellipses boundary.
    #     # '''
    #     x -= x0
    #     y -= y0
    #     if angle:
    #         # rotate the points onto an ellipse whose rx, and ry lay on the x, y
    #         # axis
    #         angle = -np.pi / 180. * angle
    #         x, y = x * np.cos(angle) - y * np.sin(angle), x * \
    #             np.sin(angle) + y * np.cos(angle)

    #     theta = np.arctan2(rx * y, ry * x)
    #     while np.absolute(ellipe_tan_dot(rx, ry, x, y, theta)) > error:
    #         theta -= ellipe_tan_dot(rx, ry, x, y, theta) / \
    #             ellipe_tan_dot_derivative(rx, ry, x, y, theta)

    #     px, py = rx * np.cos(theta), ry * np.sin(theta)
    #     return ((x - px) ** 2 + (y - py) ** 2) ** .5

    # # this function has a time out included to not get stucked in the while loop
    # def estimate_distance_with_timeout(x, y, rx, ry, x0=0, y0=0, angle=0, error=1e-5):
    #     '''Given a point (x, y), and an ellipse with major - minor axis (rx, ry),
    #     its center at (x0, y0), and with a counter clockwise rotation of
    #     `angle` degrees, will return the distance between the ellipse and the
    #     closest point on the ellipses boundary.
    #     '''
    #     # timeout = time.time() + 20*1
    #     timeout = time.time() + 2*1
    #     x -= x0
    #     y -= y0
    #     if angle:
    #         # rotate the points onto an ellipse whose rx, and ry lay on the x, y
    #         # axis
    #         angle = -np.pi / 180. * angle
    #         x, y = x * np.cos(angle) - y * np.sin(angle), x * \
    #             np.sin(angle) + y * np.cos(angle)

    #     theta = np.arctan2(rx * y, ry * x)
    #     while np.absolute(ellipe_tan_dot(rx, ry, x, y, theta)) > error:
    #         theta -= ellipe_tan_dot(
    #             rx, ry, x, y, theta) / \
    #             ellipe_tan_dot_derivative(rx, ry, x, y, theta)
    #         if time.time() > timeout:
    #             break

    #     px, py = rx * np.cos(theta), ry * np.sin(theta)
    #     t_out_reached = False
    #     if True: #time.time() < timeout:
    #         dist = ((x - px) ** 2 + (y - py) ** 2) ** .5
    #     else:
    #         dist = 0
    #         print("TIME OUT REACHED")

    #     return dist, t_out_reached

    def estimate_distance_new(semi_major, semi_minor, p):
        '''Given a point p, and an ellipse with major - minor axis (semi_major, semi_minor),
        will return the distance between the ellipse and the
        closest point on the ellipses boundary. code was taken from https://wet-robots.ghost.io/simple-method-for-distance-to-ellipse/
        '''
        px = abs(p[0])
        py = abs(p[1])

        t = math.pi / 4

        a = semi_major
        b = semi_minor

        for x in range(0, 3):
            x = a * math.cos(t)
            y = b * math.sin(t)

            ex = (a * a - b * b) * math.cos(t) ** 3 / a
            ey = (b * b - a * a) * math.sin(t) ** 3 / b

            rx = x - ex
            ry = y - ey

            qx = px - ex
            qy = py - ey

            r = math.hypot(ry, rx)
            q = math.hypot(qy, qx)

            delta_c = r * math.asin((rx * qy - ry * qx) / (r * q))
            delta_t = delta_c / math.sqrt(a * a + b * b - x * x - y * y)

            t += delta_t
            t = min(math.pi / 2, max(0, t))
        p_ellipse = (math.copysign(x, p[0]), math.copysign(y, p[1]))
        distance = math.sqrt(((p_ellipse[0] - p[0]) ** 2) + ((p_ellipse[1] - p[1]) ** 2))
        return distance

    def return_results(sigma_y, landa, phi, sigma_x, R, xc, yc, arc='top'):

        b_square = (landa / sigma_x) ** 2 * (1 + (sigma_x / sigma_y) * np.tan(phi) ** 2) / (1 + np.tan(phi) ** 2)
        a_square = (landa) ** 2 / (sigma_x * sigma_y) * (1 + (sigma_x / sigma_y) * np.tan(phi) ** 2) / (1 + np.tan(phi) ** 2)
        b = np.sqrt(b_square)
        a = np.sqrt(a_square)
        sigma_isotrop = 0.5 * (sigma_x + sigma_y)
        sigma_anisotrop_x = 0.5 * (sigma_x - sigma_y)
        sigma_anisotrop_y = 0.5 * (sigma_y - sigma_x)

        return a, b, sigma_isotrop, sigma_anisotrop_x, sigma_anisotrop_y

    # sigma_xx = np.load(directory+'sigma_xx.npy')

    # initialize arrays to store final result
    sigma_x_t_final = np.zeros((R_upper_all.shape[0], R_upper_all.shape[1]))
    sigma_y_t_final = np.zeros((R_upper_all.shape[0], R_upper_all.shape[1]))
    a_t_final = np.zeros((R_upper_all.shape[0], R_upper_all.shape[1]))
    b_t_final = np.zeros((R_upper_all.shape[0], R_upper_all.shape[1]))
    xc_t_final = np.zeros((R_upper_all.shape[0], R_upper_all.shape[1]))
    yc_t_final = np.zeros((R_upper_all.shape[0], R_upper_all.shape[1]))
    std_t_final = np.zeros((R_upper_all.shape[0], R_upper_all.shape[1]))

    sigma_x_b_final = np.zeros((R_upper_all.shape[0], R_upper_all.shape[1]))
    sigma_y_b_final = np.zeros((R_upper_all.shape[0], R_upper_all.shape[1]))
    a_b_final = np.zeros((R_upper_all.shape[0], R_upper_all.shape[1]))
    b_b_final = np.zeros((R_upper_all.shape[0], R_upper_all.shape[1]))
    xc_b_final = np.zeros((R_upper_all.shape[0], R_upper_all.shape[1]))
    yc_b_final = np.zeros((R_upper_all.shape[0], R_upper_all.shape[1]))
    std_b_final = np.zeros((R_upper_all.shape[0], R_upper_all.shape[1]))

    for c in range(R_upper_all.shape[1]):
        print("CELL ", c + 1)
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

        sigma_x_list_t = []
        sigma_x_list_b = []

        # std_t_list = []
        # std_b_list = []

        R_upper = R_upper_all[:, c] * pixelsize  # convert to µm
        R_lower = R_lower_all[:, c] * pixelsize
        XC_upper = XC_upper_all[:, c] * pixelsize
        YC_upper = YC_upper_all[:, c] * pixelsize
        XC_lower = XC_lower_all[:, c] * pixelsize
        YC_lower = YC_lower_all[:, c] * pixelsize
        # tx_top_left  = tx_top_left_all[:,c]
        ty_top_left = ty_top_left_all[:, c]
        # tx_top_right  = tx_top_right_all[:,c]
        ty_top_right = ty_top_right_all[:, c]
        # tx_bottom_left  = tx_bottom_left_all[:,c]
        ty_bottom_left = ty_bottom_left_all[:, c]
        # tx_bottom_right  = tx_bottom_right_all[:,c]
        ty_bottom_right = ty_bottom_right_all[:, c]

        # sigma_x_left, sigma_x_right = read_sigma_contribution_mat(directory, c)
        # sigma_x_left = sigma_x_all[:,c]
        # sigma_x_right = sigma_x_all[:,c]

        landa_top_left = landa_top_left_all[:, c]
        landa_top_right = landa_top_right_all[:, c]
        landa_bottom_left = landa_bottom_left_all[:, c]
        landa_bottom_right = landa_bottom_right_all[:, c]

        # a_top_approx = a_top_approx_all[:,c]
        # b_top_approx = b_top_approx_all[:,c]
        xc_top_approx = xc_top_approx_all[:, c]
        yc_top_approx = yc_top_approx_all[:, c]

        # a_bottom_approx = a_bottom_approx_all[:,c]
        # b_bottom_approx = b_bottom_approx_all[:,c]
        xc_bottom_approx = xc_bottom_approx_all[:, c]
        yc_bottom_approx = yc_bottom_approx_all[:, c]

        sigma_y_top_approx = sigma_y_all[:, c]
        sigma_y_bottom_approx = sigma_y_all[:, c]
        sigma_x_av = sigma_x_all[:, c]

        x_data_top = x_data_top_all[:, :, c]
        y_data_top = y_data_top_all[:, :, c]

        x_data_bottom = x_data_bottom_all[:, :, c]
        y_data_bottom = y_data_bottom_all[:, :, c]

        # remove lines with 0 values
        x_data_top = x_data_top[~np.all(x_data_top == 0, axis=1)]
        x_data_bottom = x_data_bottom[~np.all(x_data_bottom == 0, axis=1)]
        y_data_top = y_data_top[~np.all(y_data_top == 0, axis=1)]
        y_data_bottom = y_data_bottom[~np.all(y_data_bottom == 0, axis=1)]

        for frame in np.arange(R_upper.shape[0]):
            print("FRAME ", frame)
            R_u = R_upper[frame]
            R_l = R_lower[frame]

            # curvature_u = (1/R_u).item()
            # curvature_l = (1/R_l).item()

            xc_u = XC_upper[frame]
            yc_u = YC_upper[frame]
            xc_l = XC_lower[frame]
            yc_l = YC_lower[frame]

            ty_tl = np.absolute(ty_top_left[frame])
            ty_tr = np.absolute(ty_top_right[frame])
            ty_bl = np.absolute(ty_bottom_left[frame])
            ty_br = np.absolute(ty_bottom_right[frame])

            # ty_top = (ty_tl + ty_tr)/2
            # ty_bottom = (ty_bl + ty_br)/2

            phi_tl = np.arcsin(ty_tl)
            phi_tr = np.arcsin(ty_tr)
            phi_bl = np.arcsin(ty_bl)
            phi_br = np.arcsin(ty_br)

            phi_top = (phi_tl + phi_tr) / 2
            phi_bottom = (phi_bl + phi_br) / 2

            landa_tl = np.absolute(landa_top_left[frame])
            landa_tr = np.absolute(landa_top_right[frame])
            landa_bl = np.absolute(landa_bottom_left[frame])
            landa_br = np.absolute(landa_bottom_right[frame])

            landa_top = (landa_tl + landa_tr) / 2
            landa_bottom = (landa_bl + landa_br) / 2

            # sigma_x_l = sigma_x_left[frame]
            # sigma_x_r = sigma_x_right[frame]

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
            x_top_track = x_data_top[:, frame] * pixelsize
            y_top_track = y_data_top[:, frame] * pixelsize

            # first index gives position and second index gives frame/time
            x_bottom_track = x_data_bottom[:, frame] * pixelsize
            y_bottom_track = y_data_bottom[:, frame] * pixelsize

            def dist_top(p):

                # x = x_top_track
                # y = y_top_track
                # sigma_x = sigma_x_estimate
                xc = xc_top_ellipse_estimate

                phi = phi_top
                landa = landa_top  # nN

                sigma_x, sigma_y, yc = p

                b = np.sqrt((landa / sigma_x) ** 2 * (1 + (sigma_x / sigma_y) * np.tan(phi) ** 2) / (1 + np.tan(phi) ** 2))
                a = np.sqrt((landa) ** 2 / (sigma_x * sigma_y) * (1 + (sigma_x / sigma_y) * np.tan(phi) ** 2) / (1 + np.tan(phi) ** 2))

                dist_container_u = []

                # print("No of tracking points ", len(x_top_track), frame)
                for j in range(len(x_top_track)):
                    # dist_u, t_out_reached = estimate_distance_with_timeout(
                    # x_top_track[j], y_top_track[j], a, b, xc, yc, angle=0)
                    dist_u = estimate_distance_new(a, b, (x_top_track[j] - xc, y_top_track[j] - yc))
                    # print("dist_u ", dist_u, "Data point ", j, frame)

                    # store squared distance
                    dist_container_u.append(dist_u ** 2)

                return np.sum(dist_container_u)

            def dist_bottom(p):

                # x = x_bottom_track
                # y = y_bottom_track
                # sigma_x = sigma_x_estimate
                xc = xc_bottom_ellipse_estimate

                phi = phi_bottom
                landa = landa_bottom  # nN

                sigma_x, sigma_y, yc = p  # add sigma x here maybe

                b = np.sqrt((landa / sigma_x) ** 2 * (1 + (sigma_x / sigma_y) * np.tan(phi) ** 2) / (1 + np.tan(phi) ** 2))
                a = np.sqrt((landa) ** 2 / (sigma_x * sigma_y) * (1 + (sigma_x / sigma_y) * np.tan(phi) ** 2) / (1 + np.tan(phi) ** 2))

                dist_container_l = []
                # print("No of tracking points ", len(x_bottom_track), frame)
                for j in range(len(x_bottom_track)):
                    # dist_l, t_out_reached = estimate_distance_with_timeout(
                    #     x_bottom_track[j], y_bottom_track[j], a, b, xc, yc, angle=0)
                    dist_l = estimate_distance_new(a, b, (x_bottom_track[j] - xc, y_bottom_track[j] - yc))
                    # print("dist_l ",dist_l,"Data point ",j,frame)

                    # store squared distance
                    dist_container_l.append(dist_l ** 2)

                return np.sum(dist_container_l)

            if frame == 0:
                params0_top = [sigma_x_estimate, sigma_y_top_estimate, yc_top_ellipse_estimate]
                params0_bottom = [sigma_x_estimate, sigma_y_bottom_estimate, yc_bottom_ellipse_estimate]
            else:
                params0_top = [sigma_x_list_t[0], sigma_y_list_t[0], yc_list_t[0]]
                params0_bottom = [sigma_x_list_b[0], sigma_y_list_b[0], yc_list_b[0]]
            # perform the fit
            # add sigma x here maybe
            result_top = optimize.minimize(dist_top, params0_top, method='Nelder-Mead', options={'disp': False,
                                                                                                 'adaptive': True})  # , 'maxiter':2000})  # options={'maxiter': 10000, 'disp': True}
            result_bottom = optimize.minimize(dist_bottom, params0_bottom, method='Nelder-Mead',
                                              options={'disp': False, 'adaptive': True})  # , 'maxiter':2000})

            # print(result.fit_report())
            sigma_x_top, sigma_y_top, yc_ellipse_top = result_top.x  # add sigma x here maybe
            a_top, b_top, sigma_isotrop_top, sigma_anisotrop_x_top, sigma_anisotrop_y_top = return_results(sigma_y_top, landa_top, phi_top,
                                                                                                           sigma_x_top, R_u, xc_u, yc_u,
                                                                                                           arc='top')

            sigma_x_bottom, sigma_y_bottom, yc_ellipse_bottom = result_bottom.x
            a_bottom, b_bottom, sigma_isotrop_bottom, sigma_anisotrop_x_bottom, sigma_anisotrop_y_bottom = return_results(sigma_y_bottom,
                                                                                                                          landa_bottom,
                                                                                                                          phi_bottom,
                                                                                                                          sigma_x_bottom,
                                                                                                                          R_l, xc_l, yc_l,
                                                                                                                          arc='bottom')

            xc_ellipse_top = xc_top_ellipse_estimate
            xc_ellipse_bottom = xc_bottom_ellipse_estimate

            if frame == 1:
                # plot ellipses and data for quality check
                t = np.linspace(0, 2 * np.pi, 100)
                plt.figure(figsize=(5, 5))

                plt.plot(xc_ellipse_top + a_top * np.cos(t), yc_ellipse_top + b_top * np.sin(t))
                plt.plot(x_top_track, y_top_track, 'bo')
                plt.grid(color='lightgray', linestyle='--')

                plt.plot(xc_ellipse_bottom + a_bottom * np.cos(t), yc_ellipse_bottom + b_bottom * np.sin(t))
                plt.plot(x_bottom_track, y_bottom_track, 'bo')
                plt.grid(color='lightgray', linestyle='--')

                plt.xlim((0, 100))
                plt.ylim((0, 100))

                if not os.path.exists(str.replace(folder, '.dat', '')):
                    os.mkdir(str.replace(folder, '.dat', ''))

                savefolder = str.replace(folder, '.dat', '/ellipsefit')

                if not os.path.exists(savefolder):
                    os.mkdir(savefolder)

                plt.savefig(savefolder + '/cell' + str(c) + '.png')
                plt.close()

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

            sigma_x_list_t.append(sigma_x_top)
            sigma_x_list_b.append(sigma_x_bottom)

        std_t, std_b = estimate_ellipse_fit_std(x_data_top, y_data_top, x_data_bottom, y_data_bottom,
                                                R_upper, R_lower, XC_upper, YC_upper, XC_lower, YC_lower,
                                                a_list_t, b_list_t, xc_list_t, yc_list_t, a_list_b, b_list_b, xc_list_b, yc_list_b,
                                                pixelsize)
        # print(std_t)
        # print(std_b)
        sigma_x_t_final[:, c] = sigma_x_list_t
        sigma_y_t_final[:, c] = sigma_y_list_t
        a_t_final[:, c] = a_list_t
        b_t_final[:, c] = b_list_t
        xc_t_final[:, c] = xc_list_t
        yc_t_final[:, c] = yc_list_t
        std_t_final[:, c] = std_t

        sigma_x_b_final[:, c] = sigma_x_list_b
        sigma_y_b_final[:, c] = sigma_y_list_b
        a_b_final[:, c] = a_list_b
        b_b_final[:, c] = b_list_b
        xc_b_final[:, c] = xc_list_b
        yc_b_final[:, c] = yc_list_b
        std_b_final[:, c] = std_t
    # save data in structure
    data_dict = {'sigma x top [nN/um]': sigma_x_t_final,
                 'sigma y top [nN/um]': sigma_y_t_final,
                 'a top [um]': a_t_final,
                 'b top [um]': b_t_final,
                 'xc top [um]': xc_t_final,
                 'yc top [um]': yc_t_final,
                 'std top [um]': std_t_final,
                 'a bottom [um]': a_b_final,
                 'b bottom [um]': b_b_final,
                 'sigma x bottom [nN/um]': sigma_x_b_final,
                 'sigma y bottom [nN/um]': sigma_y_b_final,
                 'xc bottom [um]': xc_b_final,
                 'yc bottom [um]': yc_b_final,
                 'std bottom [um]': std_b_final}

    # save_dict_to_csv(data_dict, directory+'/ellipse_data_fit (%s).csv' % (c+1))

    return data_dict


def calculate_various_stuff(circle_fit_data, tangent_data, TEM_data, ellipse_data_approx, ellipse_data):
    sigma_x_fit = (ellipse_data['sigma x top [nN/um]'] + ellipse_data['sigma x bottom [nN/um]']) / 2
    sigma_x_fit_baseline = np.nanmean(sigma_x_fit[0:20, :], axis=0)

    sigma_y_approx = (ellipse_data_approx["sigma y top [nN/µm]"] + ellipse_data_approx["sigma y bottom [nN/µm]"]) / 2
    sigma_y_approx_baseline = np.nanmean(sigma_y_approx[0:20, :], axis=0)
    sigma_y_fit = (ellipse_data['sigma y top [nN/um]'] + ellipse_data['sigma y bottom [nN/um]']) / 2
    sigma_y_fit_baseline = np.nanmean(sigma_y_fit[0:20, :], axis=0)

    std_approx = (ellipse_data_approx['std top [µm]'] + ellipse_data_approx['std bottom [µm]']) / 2
    std_approx_baseline = np.nanmean(std_approx[0:20, :], axis=0)
    std_fit = (ellipse_data['std top [um]'] + ellipse_data['std bottom [um]']) / 2
    std_fit_baseline = np.nanmean(std_fit[0:20, :], axis=0)

    ellipse_data_approx['sigma_y'] = sigma_y_approx
    ellipse_data_approx['sigma_y_baseline'] = sigma_y_approx_baseline
    ellipse_data_approx['std'] = std_approx
    ellipse_data_approx['std_baseline'] = std_approx_baseline

    ellipse_data['sigma_x'] = sigma_x_fit
    ellipse_data['sigma_x_baseline'] = sigma_x_fit_baseline
    ellipse_data['sigma_y'] = sigma_y_fit
    ellipse_data['sigma_y_baseline'] = sigma_y_fit_baseline
    ellipse_data['std'] = std_fit
    ellipse_data['std_baseline'] = std_fit_baseline

    ellipse_data['a_baseline'] = np.nanmean((ellipse_data['a top [um]'][0:20, :] + ellipse_data['a bottom [um]'][0:20, :]) / 2, axis=0)
    ellipse_data['b_baseline'] = np.nanmean((ellipse_data['b top [um]'][0:20, :] + ellipse_data['b bottom [um]'][0:20, :]) / 2, axis=0)

    landa_top_left = TEM_data['line tension top left [nN]']
    landa_top_right = TEM_data['line tension top right [nN]']
    landa_bottom_right = TEM_data['line tension bottom right [nN]']
    landa_bottom_left = TEM_data['line tension bottom left [nN]']

    f_top_left = TEM_data['f top left [nN]']
    f_top_right = TEM_data['f top right [nN]']
    f_bottom_right = TEM_data['f bottom right [nN]']
    f_bottom_left = TEM_data['f bottom left [nN]']

    f_adherent_baseline = np.nanmean((f_top_left[0:20, :] + f_top_right[0:20, :] - f_bottom_left[0:20, :] - f_bottom_right[0:20, :]) / 4,
                                     axis=0)
    landa_baseline = np.nanmean(
        (landa_top_left[0:20, :] + landa_top_right[0:20, :] + landa_bottom_left[0:20, :] + landa_bottom_right[0:20, :]) / 4, axis=0)

    TEM_data["f adherent baseline [nN]"] = f_adherent_baseline
    TEM_data["line tension baseline [nN]"] = landa_baseline

    return circle_fit_data, tangent_data, TEM_data, ellipse_data_approx, ellipse_data


def main(inputpath, outputpath, pixelsize, shortcut=False):
    if shortcut == False:
        # load data
        data = pickle.load(open(inputpath, "rb"))

        # pull needed data for analysis out of dataset
        x_top = data['shape_data']['Xtop']
        x_bottom = data['shape_data']['Xbottom']

        y_top = data['shape_data']['Ytop']
        y_bottom = data['shape_data']['Ybottom']

        Fx_top_left = data['TFM_data']['Fx_topleft'] * 1e9  # convert to nN
        Fx_top_right = data['TFM_data']['Fx_topright'] * 1e9
        Fx_bottom_right = data['TFM_data']['Fx_bottomright'] * 1e9
        Fx_bottom_left = data['TFM_data']['Fx_bottomleft'] * 1e9

        Fy_top_left = data['TFM_data']['Fy_topleft'] * 1e9
        Fy_top_right = data['TFM_data']['Fy_topright'] * 1e9
        Fy_bottom_right = data['TFM_data']['Fy_bottomright'] * 1e9
        Fy_bottom_left = data['TFM_data']['Fy_bottomleft'] * 1e9

        sigma_x = data['MSM_data']['sigma_xx_average'] * 1e3  # convert to nN/µm
        sigma_y = data['MSM_data']['sigma_yy_average'] * 1e3  # convert to nN/µm

        # remove the loaded dataset to save memory    
        del data

        # run circle_fit function
        circle_fit_data = main_circle_fit(x_top, x_bottom, y_top, y_bottom)

        XC_upper = circle_fit_data['x-pos center [px] (upper)']
        YC_upper = circle_fit_data['y-pos center [px] (upper)']
        XC_lower = circle_fit_data['x-pos center [px] (lower)']
        YC_lower = circle_fit_data['y-pos center [px] (lower)']
        R_upper = circle_fit_data['Radius [px] (upper)']
        R_lower = circle_fit_data['Radius [px] (lower)']

        # run main_tangent_calc function
        tangent_data = main_calc_tangents(x_top, x_bottom, y_top, y_bottom,
                                          XC_upper, YC_upper, XC_lower, YC_lower, R_upper, R_lower)

        tx_top_left = tangent_data['tx top left']
        tx_top_right = tangent_data['tx top right']
        tx_bottom_right = tangent_data['tx bottom right']
        tx_bottom_left = tangent_data['tx bottom left']

        ty_top_left = tangent_data['ty top left']
        ty_top_right = tangent_data['ty top right']
        ty_bottom_right = tangent_data['ty bottom right']
        ty_bottom_left = tangent_data['ty bottom left']

        # run main_TEM_circles
        TEM_data = main_TEM_circles(R_upper, R_lower,
                                    tx_top_left, ty_top_left, tx_top_right, ty_top_right, tx_bottom_right, ty_bottom_right, tx_bottom_left,
                                    ty_bottom_left,
                                    Fx_top_left, Fy_top_left, Fx_top_right, Fy_top_right, Fx_bottom_right, Fy_bottom_right, Fx_bottom_left,
                                    Fy_bottom_left, pixelsize)

        landa_top_left = TEM_data['line tension top left [nN]']
        landa_top_right = TEM_data['line tension top right [nN]']
        landa_bottom_left = TEM_data['line tension bottom left [nN]']
        landa_bottom_right = TEM_data['line tension bottom right [nN]']

        ellipse_data_approx = main_ellipse_approx(x_top, x_bottom, y_top, y_bottom,
                                                  R_upper, R_lower,
                                                  XC_upper, YC_upper, XC_lower, YC_lower,
                                                  tx_top_left, ty_top_left, tx_top_right, ty_top_right,
                                                  tx_bottom_right, ty_bottom_right, tx_bottom_left, ty_bottom_left,
                                                  landa_top_left, landa_top_right, landa_bottom_left, landa_bottom_right,
                                                  sigma_x, pixelsize, inputpath)

        a_top_approx = ellipse_data_approx['a top [µm]']
        b_top_approx = ellipse_data_approx['b top [µm]']
        xc_top_approx = ellipse_data_approx['xc top [µm]']
        yc_top_approx = ellipse_data_approx['yc top [µm]']

        a_bottom_approx = ellipse_data_approx['a bottom [µm]']
        b_bottom_approx = ellipse_data_approx['b bottom [µm]']
        xc_bottom_approx = ellipse_data_approx['xc bottom [µm]']
        yc_bottom_approx = ellipse_data_approx['yc bottom [µm]']

        sigma_y_top_approx = ellipse_data_approx['sigma y top [nN/µm]']
        sigma_y_bottom_approx = ellipse_data_approx['sigma y bottom [nN/µm]']

        ellipse_data = main_ellipse_fit(R_upper, R_lower,
                                        XC_upper, YC_upper, XC_lower, YC_lower,
                                        tx_top_left, ty_top_left, tx_top_right, ty_top_right,
                                        tx_bottom_right, ty_bottom_right, tx_bottom_left, ty_bottom_left,
                                        sigma_x,
                                        landa_top_left, landa_top_right, landa_bottom_left, landa_bottom_right,
                                        x_bottom, y_bottom, x_top, y_top,
                                        a_top_approx, b_top_approx, xc_top_approx, yc_top_approx,
                                        a_bottom_approx, b_bottom_approx, xc_bottom_approx, yc_bottom_approx,
                                        sigma_y, inputpath)

        # calculate some stuff
        circle_fit_data, tangent_data, TEM_data, ellipse_data_approx, ellipse_data = calculate_various_stuff(circle_fit_data, tangent_data,
                                                                                                             TEM_data, ellipse_data_approx,
                                                                                                             ellipse_data)


    # load previously analysed data to make some additional calculations
    elif shortcut:
        CM_data = pickle.load(open(outputpath, "rb"))
        circle_fit_data = CM_data["circle_fit_data"]
        tangent_data = CM_data["tangent_data"]
        TEM_data = CM_data["TEM_data"]
        ellipse_data_approx = CM_data["ellipse_data_approx"]
        ellipse_data = CM_data["ellipse_data"]

        circle_fit_data, tangent_data, TEM_data, ellipse_data_approx, ellipse_data = calculate_various_stuff(circle_fit_data, tangent_data,
                                                                                                             TEM_data, ellipse_data_approx,
                                                                                                             ellipse_data)

    CM_data = {'circle_fit_data': circle_fit_data,
               'tangent_data': tangent_data,
               'TEM_data': TEM_data,
               'ellipse_data_approx': ellipse_data_approx,
               'ellipse_data': ellipse_data}

    with open(outputpath, 'wb') as outfile:
        pickle.dump(CM_data, outfile, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    pixelsize = 0.108  # in µm
    shortcut = True

    path = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/analysed_data/AR1to1d_fullstim_long"
    main(path + ".dat", path + "_CM.dat", pixelsize, shortcut)

    path = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/analysed_data/AR1to1s_fullstim_long"
    main(path + ".dat", path + "_CM.dat", pixelsize, shortcut)

    path = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/analysed_data/AR1to1d_fullstim_short"
    main(path + ".dat", path + "_CM.dat", pixelsize, shortcut)

    path = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/analysed_data/AR1to1s_fullstim_short"
    main(path + ".dat", path + "_CM.dat", pixelsize, shortcut)

    path = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/analysed_data/AR1to2d_halfstim"
    main(path + ".dat", path + "_CM.dat", pixelsize, shortcut)

    path = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/analysed_data/AR1to1d_halfstim"
    main(path + ".dat", path + "_CM.dat", pixelsize, shortcut)

    path = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/analysed_data/AR1to1s_halfstim"
    main(path + ".dat", path + "_CM.dat", pixelsize, shortcut)

    path = "C:/Users/Balland/Documents/_forcetransmission_in_cell_doublets_alldata/analysed_data/AR2to1d_halfstim"
    main(path + ".dat", path + "_CM.dat", pixelsize, shortcut)
