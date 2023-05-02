import pandas as pd
import openpyxl
import numpy as np
from scipy.interpolate import interp1d, CubicSpline, UnivariateSpline
import matplotlib.pyplot as plt
# s = 17.5
s = 25/2
x = np.linspace(-s,s,200)
print(s)

def shifted_ellipse(a,b,x):

    y_standard = -np.sqrt(b**2*(1-x**2/a**2))
    y_x_shift = -np.sqrt(b**2*(1-(x-s)**2/a**2))
    y_shift = -np.sqrt(b**2*(1-s**2/a**2))
    print(y_shift)
    return y_standard - y_shift

def slope_angle_adhesion(a,b,x):
    y_x_shift = -np.sqrt(b ** 2 * (1 - (x - s) ** 2 / a ** 2))
    theta = np.arctan(-b**2/a**2*(x-s)/y_x_shift)
    print("theta ",np.degrees(theta))
    return theta


# 2to1
s_x = 1.1
s_y = 1.6
a = 25
b = a*np.sqrt(s_y/s_x)
print('b = ',b)
typ = "2to1"
# # 1to2
# s_x = 0.55
# s_y = 0.5
# a = 150
# b = a*np.sqrt(s_y/s_x)
# print('b = ',b)
# typ = "1to2"
# # singlet
# s_x = 3.21
# s_y = 0.89
# a = 53.86
# b = a*np.sqrt(s_y/s_x)
# print('b = ',b)
# typ = "singlet"
# doublet
# s_x = 0.92
# s_y = 1.12
# a = 61.94
# b = a*np.sqrt(s_y/s_x)
# print('b = ',b)
# typ = "doublet"

theta = slope_angle_adhesion(a,b,0)
print("Theta in rad ",theta)

landa = np.sqrt(s_x*s_y)*a *np.sqrt((1+np.tan(theta)**2)/(1+s_x/s_y*np.tan(theta)**2))
print("landa ",landa)

y = shifted_ellipse(a,b,x)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
# plt.plot(u_x_array_init, u_y_array_init,color='k', label='isotropic')
ax.plot(x+s, y,color='gray', label=r'contour exp')

ax.set_xlim(0, 2*s)
ax.set_aspect('equal')
ax.set_ylim(-18.0, 18.0)
ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("y")

fig.savefig("ellipse_contour_%s.pdf"%(typ),tight_layout=True)
np.savetxt("contour_%s.txt"%(typ), np.c_[y], header="y(x) [um]")