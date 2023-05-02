"""
This demo program solves Poisson's equation
    (u^2/2)_x - u_xx = f(x)
on the unit interval with source f given by
    f(x) = 9*pi^2*sin(3*pi*x[0])
and boundary conditions given by
    u(0) = u(1) = 0
"""

from dolfin import *

import numpy as np
import matplotlib.pyplot as plt
import circle_fit as cf
from scipy.interpolate import interp1d, CubicSpline, UnivariateSpline

parameters['reorder_dofs_serial'] = False
# Create mesh and function space
mesh = IntervalMesh(199, 0, 35)
l = np.linspace(0,35,200)
# mesh = Mesh("mesh.xml")
# Define function space for system of concentrations
P1 = FiniteElement('CG', interval, 1)
element = MixedElement([P1, P1])
V = FunctionSpace(mesh, element)

# Define test functions
v_x, v_y = TestFunctions(V)

tol = 1E-14


def DirichletBoundary_y(x, on_boundary):
    return near(x[0], 0.0, tol) or near(x[0], 35.0, tol)


def DirichletBoundary_x_left(x, on_boundary):
    return near(x[0], 0.0, tol)


def DirichletBoundary_x_right(x, on_boundary):
    return near(x[0], 35., tol)


# Define the isotropic model based on TEM Model with an additional active line tension
u = Function(V)

R = 50 / 1.0
L = 2 * R * np.arcsin(35 / (2 * R))
landa_act = 50 - 100 * (L / 35 - 1)
print("Stretch ", L / 35)
print("Landa_act ", landa_act)
u_x, u_y = split(u)
sigma_x = Constant(1.0)
sigma_y = Constant(1.0)
EA = Constant(100)  # nN
landa = Constant(50)
nu = Constant(L / 35)
F = (- landa / nu * dot(grad(u_x), grad(v_x)) * dx + sigma_x * u_y.dx(0) * v_x * dx \
     - landa / nu * dot(grad(u_y), grad(v_y)) * dx - sigma_y * u_x.dx(0) * v_y * dx)

# Define boundary condition
u_x_0 = Constant(0.0)
u_x_L = Constant(35.0)
u_y_0_L = Constant(0.0)

bc_y = DirichletBC(V.sub(1), u_y_0_L, DirichletBoundary_y)
bc_x_left = DirichletBC(V.sub(0), u_x_0, DirichletBoundary_x_left)
bc_x_right = DirichletBC(V.sub(0), u_x_L, DirichletBoundary_x_right)

bcs = [bc_y, bc_x_right, bc_x_left]

# solve the isotropic model
solve(F == 0, u, bcs)
# solver_parameters={"newton_solver":{"relative_tolerance": 1e-6}})
u_save_init = Function(V)
u_save_init.assign(u)
ux_save_init, uy_save_init = u_save_init.split(deepcopy=True)
u_x_array_init = ux_save_init.vector().get_local()
u_y_array_init = uy_save_init.vector().get_local()

# ------------------------------------------- Here starts the main simulation ------------------------------
s_x = 32
s_y = 3
# s_x = 1
# s_y = 10
l_act = 150# determined from text file of fit result
ea = 200
a_exp = 38.99

####################### First Simulation NO PHOTO-ACTIVATION ###################################################


# Define the anisotropic model based on Anisotropic Contour model with an additional active line tension
# The solution of the isotropic model serves as an initial guess for the anisotropic model

u_x, u_y = split(u)
sigma_x = Constant(s_x)
sigma_y = Constant(s_y)
EA = Constant(ea)  # Expression("ea -ea* 0.8/(1+exp(-(x[0]-15)/0.1))*(1-1/(1+exp(-(x[0]-20)/0.1))) ",ea=ea,degree=0)# #nN
landa_act = Constant(l_act)

landa = landa_act + EA * (sqrt(u_x.dx(0) ** 2 + u_y.dx(0) ** 2) - 1)
nu = sqrt(u_x.dx(0) ** 2 + u_y.dx(0) ** 2)
F = (- landa / nu * dot(grad(u_x), grad(v_x)) * dx + sigma_x * u_y.dx(0) * v_x * dx \
     - landa / nu * dot(grad(u_y), grad(v_y)) * dx - sigma_y * u_x.dx(0) * v_y * dx)

# solver takes solution "u" from isotropic model as an inital guess and saves the new result to variable "u"
solve(F == 0, u, bcs)

print("WORKED")
u_save_ani = Function(V)
u_save_ani.assign(u)
ux_save_ani, uy_save_ani = u_save_ani.split(deepcopy=True)
u_x_array_ani = ux_save_ani.vector().get_local()
u_y_array_ani = uy_save_ani.vector().get_local()

Q = FunctionSpace(mesh, "DG", 0)
deriv = ux_save_ani.dx(0)
u_xdx_ani = project(ux_save_ani.dx(0), Q)
u_ydx_ani = project(uy_save_ani.dx(0), Q)
print(u_xdx_ani(*mesh.coordinates()[0]))
print(u_xdx_ani(*mesh.coordinates()[1]))
u_xdx_ani_array = np.empty_like(np.linspace(0, 35, 200))
u_ydx_ani_array = np.empty_like(np.linspace(0, 35, 200))
for v in vertices(mesh):
    print(v.index())
    u_xdx_ani_array[v.index()] = u_xdx_ani(*mesh.coordinates()[v.index()])
    u_ydx_ani_array[v.index()] = u_ydx_ani(*mesh.coordinates()[v.index()])

stretch_ani = np.sqrt(u_xdx_ani_array ** 2 + u_ydx_ani_array ** 2)
tan_ani = (u_ydx_ani_array / u_xdx_ani_array)
theta_ani = np.arctan(u_ydx_ani_array / u_xdx_ani_array)
theta_ani_deg = np.degrees(theta_ani)

Space_ea = FunctionSpace(mesh, "DG", 0)
ea_interpol = interpolate(EA, Space_ea)
ea_array = np.empty_like(np.linspace(0, 35, 200))
for v in vertices(mesh):
    ea_array[v.index()] = ea_interpol(*mesh.coordinates()[v.index()])


Space_landa_act = FunctionSpace(mesh, "DG", 0)
landa_act_array = interpolate(landa_act, Space_landa_act)
landa_act_array = landa_act_array.vector().get_local()

landa_ani = (l_act + ea * (stretch_ani - 1))
################################################ END OF FIRST SIMULATION ##############################################################




################################################ Second Simulation add PHOTO-ACTIVATION ###############################################

# Define the anisotropic model based on Anisotropic Contour model with an additional active line tension and PA ACTIVATION
# The solution of the isotropic model serves as an initial guess for the anisotropic model
activation_profile_sigma_x = Constant(0.12)#Expression("1/(1+exp(0.6*(x[0]-11.5)))/10 ",degree=0)#Expression("(0.1-0.09)*1/(1+exp(0.8*(x[0]-11.5)))+0.09",degree=0)#Expression("1/(1+exp(0.6*(x[0]-11.5)))/10 ",degree=0)
activation_profile_sigma_y = Constant(0.21)#Expression("1/(1+exp(0.6*(x[0]-11.5)))/5 ",degree=0)#Expression("(0.1-0.09)*1/(1+exp(0.8*(x[0]-11.5)))+0.09",degree=0)#Expression("1/(1+exp(0.6*(x[0]-11.5)))/10 ",degree=0)#Constant(0.12)
activation_profile_landa = Constant(0.00)  # Expression("1/(1+exp(0.6*(x[0]-11.5)))/20 ",degree=0)Constant(0.00)
u_x, u_y = split(u)
sigma_x = Constant(s_x) + Constant(s_x) * activation_profile_sigma_x
sigma_y = Constant(s_y) + Constant(s_y) * activation_profile_sigma_y
EA = Constant(ea)  # Expression("ea -ea* 0.8/(1+exp(-(x[0]-15)/0.1))*(1-1/(1+exp(-(x[0]-20)/0.1))) ",ea=ea,degree=0)#Constant(ea) #nN
landa_act = Constant(l_act)
alpha = 1000
landa = landa_act + landa_act * activation_profile_landa + EA * (sqrt(u_x.dx(0) ** 2 + u_y.dx(0) ** 2) - 1)
nu = sqrt(u_x.dx(0) ** 2 + u_y.dx(0) ** 2)
F = (- landa / nu * dot(grad(u_x), grad(v_x)) * dx + sigma_x * u_y.dx(0) * v_x * dx \
     - landa / nu * dot(grad(u_y), grad(v_y)) * dx - sigma_y * u_x.dx(0) * v_y * dx)

solve(F == 0, u, bcs)
File("primal.pvd") << u

u_save = Function(V)
u_save.assign(u)

ux_save, uy_save = u_save.split(deepcopy=True)
u_x_array = ux_save.vector().get_local()
u_y_array = uy_save.vector().get_local()
np.savetxt("ydisplacement.txt", np.c_[u_x_array, u_y_array], header="")

# create function mesh to calculate derivatives du_x/dl etc
Q = FunctionSpace(mesh, "DG", 0)
u_xdx = project(u_x.dx(0), Q)
u_ydx = project(u_y.dx(0), Q)

u_xdx_array = np.empty_like(np.linspace(0,35,200))
u_ydx_array = np.empty_like(np.linspace(0,35,200))
for v in vertices(mesh):
    u_xdx_array[v.index()] = u_xdx(*mesh.coordinates()[v.index()])
    u_ydx_array[v.index()] = u_ydx(*mesh.coordinates()[v.index()])


stretch = np.sqrt(u_xdx_array ** 2 + u_ydx_array ** 2)
tan = (u_ydx_array / u_xdx_array)
theta = np.arctan(u_ydx_array / u_xdx_array)
theta_deg = np.degrees(theta)


Space = FunctionSpace(mesh, "DG", 0)
activation_profile_landa_interpol = interpolate(activation_profile_landa, Space)
activation_profile_sigma_x_interpol = interpolate(activation_profile_sigma_x, Space)
activation_profile_sigma_y_interpol = interpolate(activation_profile_sigma_y, Space)

activation_profile_landa_array = np.empty_like(np.linspace(0,35,200))
activation_profile_sigma_x_array = np.empty_like(np.linspace(0,35,200))
activation_profile_sigma_y_array = np.empty_like(np.linspace(0,35,200))
for v in vertices(mesh):
    activation_profile_landa_array[v.index()] = activation_profile_landa_interpol(*mesh.coordinates()[v.index()])
    activation_profile_sigma_x_array[v.index()] = activation_profile_sigma_x_interpol(*mesh.coordinates()[v.index()])
    activation_profile_sigma_y_array[v.index()] = activation_profile_sigma_y_interpol(*mesh.coordinates()[v.index()])

landa = (l_act + l_act * activation_profile_landa_array + ea * (stretch - 1))
print("Theta Ani 0 ",theta_ani[0])
print("Theta Ani 0 degrees ",theta_ani_deg[0])
print("Theta 0 ",theta[0])
print("Theta 0 degrees ",theta_deg[0])
print(landa)
theta_after_opto = np.arctan(u_ydx_array / u_xdx_array)
# print("Force balance x ", landa[0]*np.cos(theta_after_opto[0]) - landa[-1]*np.cos(theta_after_opto[-1]))
# print("F_x left = ",landa[0]*np.cos(theta_after_opto[0]))
# print("F_x right = ",landa[-1]*np.cos(theta_after_opto[-1]))
print(theta_after_opto[0],theta_after_opto[-1])
print(theta_after_opto)


relative_landa_increase_left = (landa[0] - landa_ani[0]) / landa_ani[0]
relative_landa_increase_right = (landa[-1] - landa_ani[-1]) / landa_ani[-1]
landa_left = landa[0]
landa_right = landa[-1]

landa_left_ani = landa_ani[0]
landa_right_ani = landa_ani[-1]
print((ea * (stretch_ani - 1))[0])
print((ea * (stretch - 1))[0])
integral = assemble(nu*dx)
print("Length of stretched fiber ",integral)
cf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
region = AutoSubDomain(lambda x, on: x[0] <= 17.5)
region.mark(cf, 1)
dx_sub = Measure('dx', subdomain_data=cf)
integral_sub = assemble(nu*dx_sub(0))
print("Half length of stretched fiber ",integral_sub)
##################################################### END OF SECOND SIMULATION ###################################################

def interpolate_contour(u_x_array, u_y_array, u_x_array_ani, u_y_array_ani):
    x_init = u_x_array_ani
    y_init = u_y_array_ani
    x = u_x_array
    y = u_y_array

    f_init = interp1d(x_init, y_init, kind='cubic')
    f = interp1d(x, y, kind='cubic')  # interpolated arc

    x_range = np.linspace(0, 35, 200)
    # arc_curvature = curvature_splines(xnew, f_array)
    l_0 = np.absolute(f_init(x_range) + 17.5)
    l = np.absolute(f(x_range) + 17.5)

    epsilon_yy_array = 1 - l_0 / l
    print(len(x_range))
    print(len(epsilon_yy_array))
    return x_range, epsilon_yy_array


x_range, epsilon_yy_array = interpolate_contour(u_x_array, u_y_array, u_x_array_ani, u_y_array_ani)
landa_expected = np.sqrt(s_x * s_y) * a_exp * np.sqrt((1 + np.tan(theta_ani)**2)/(1 + s_x / s_y * np.tan(theta_ani)**2))
y_halfstim = np.loadtxt("contour_doublet.txt", skiprows=1)
np.savez('contour_sim_data_doublet.npz',l=l,x=u_x_array,y=u_y_array,landa_expected=landa_expected,landa_bPA=landa_ani,landa_aPA=landa)
# print(y_halfstim)
# plotting results section
# fig = plt.figure()
# # plt.plot(u_x_array_init, u_y_array_init,color='k', label='isotropic')
# plt.plot(np.linspace(0, 35, 200), y_halfstim, color='gray', label=r'contour exp')
# plt.plot(u_x_array_ani, u_y_array_ani, label='before PA')
# plt.plot(u_x_array, u_y_array, label='after PA')
#
# plt.xlim(0, 35)
# plt.axes().set_aspect('equal')
# plt.ylim(-18.0, 18.0)
# plt.legend()
# plt.xlabel("x")
# plt.ylabel("y")
# plt.figure(figsize=(5, 5))
# fig.savefig("plots_doublet/y(x).pdf", tight_layout=True)
#
# fig = plt.figure()
# # plt.plot(u_x_array_init, u_y_array_init,color='k', label='isotropic')
# plt.plot(np.linspace(0, 35, 200), y_halfstim, color='gray', label=r'contour exp')
# plt.plot(u_x_array_ani, u_y_array_ani, label='before PA')
# plt.plot(u_x_array, u_y_array, label='after PA')
#
# plt.xlim(0, 35)
# # plt.axes().set_aspect('equal')
# plt.ylim(-6.0, 6.0)
# plt.legend()
# plt.xlabel("x")
# plt.ylabel("y")
# plt.figure(figsize=(5, 5))
# fig.savefig("plots_doublet/y(x)_zoomed.pdf", tight_layout=True)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax2 = ax.twinx()
# # plt.plot(u_x_array_init, u_y_array_init,color='k', label='isotropic')
# ax.plot(u_x_array_ani, u_y_array_ani, label='before PA')
# ax.plot(u_x_array, u_y_array, label='after PA')
# ax2.plot(u_x_array,activation_profile_sigma_x_array,linestyle=":",color='gray',label='act. profile')
# ax.set_xlim(0, 35)
# # plt.axes().set_aspect('equal')
# ax.set_ylim(-6, 6)
# ax2.set_ylim(0.0, 0.1)
# plt.legend()
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax2.set_ylabel("activation sigmoid")
# plt.figure(figsize=(5, 5))
# fig.savefig("plots_doublet/y(x)_zoomed_with_activation.pdf", tight_layout=True)
#
#
# fig = plt.figure()
# plt.plot(l, stretch, label='y')
# plt.xlim(-1, 36)
# # plt.axes().set_aspect('equal')
# # plt.ylim(-1.0, 1.0)
# plt.legend()
# plt.xlabel("l")
# plt.ylabel("stretch")
# plt.figure(figsize=(5, 5))
# fig.savefig("plots_doublet/stretch(l).pdf", tight_layout=True)
#
# fig = plt.figure()
# plt.plot(l, u_xdx_array, label='y')
# # plt.xlim(-1, 36)
# # plt.axes().set_aspect('equal')
# # plt.ylim(-1.0, 1.0)
# plt.legend()
# plt.xlabel("l")
# plt.ylabel("du_x/dl")
# plt.figure(figsize=(5, 5))
# fig.savefig("plots_doublet/du_x_dl.pdf", tight_layout=True)
#
# fig = plt.figure()
# plt.plot(u_x_array, u_xdx_array / stretch, label='y')
# # plt.xlim(-1, 36)
# # plt.axes().set_aspect('equal')
# # plt.ylim(-1.0, 1.0)
# plt.legend()
# plt.xlabel("l")
# plt.ylabel("du_x/dl")
# plt.figure(figsize=(5, 5))
# fig.savefig("plots_doublet/du_x_ds(x).pdf", tight_layout=True)
#
# fig = plt.figure()
# plt.plot(l, u_ydx_array, label='y')
# # plt.xlim(-1, 36)
# # plt.axes().set_aspect('equal')
# # plt.ylim(-1.0, 1.0)
# plt.legend()
# plt.xlabel("l")
# plt.ylabel("du_y/dl")
# plt.figure(figsize=(5, 5))
# fig.savefig("plots_doublet/du_y_dl.pdf", tight_layout=True)
#
# fig = plt.figure()
# plt.plot(l, u_ydx_array / stretch, label='y')
# # plt.xlim(-1, 36)
# # plt.axes().set_aspect('equal')
# # plt.ylim(-1.0, 1.0)
# plt.legend()
# plt.xlabel("l")
# plt.ylabel("du_y/dl")
# plt.figure(figsize=(5, 5))
# fig.savefig("plots_doublet/du_y_ds(l).pdf", tight_layout=True)
#
# fig = plt.figure()
# plt.plot(l, tan, label='y')
# # plt.xlim(-1, 36)
# # plt.axes().set_aspect('equal')
# # plt.ylim(-1.0, 1.0)
# plt.legend()
# plt.xlabel("l")
# plt.ylabel("tan")
# plt.figure(figsize=(5, 5))
# fig.savefig("plots_doublet/tan(l).pdf", tight_layout=True)
#
# fig = plt.figure()
# plt.plot(l, theta_deg, label='y')
# # plt.xlim(-1, 36)
# # plt.axes().set_aspect('equal')
# # plt.ylim(-1.0, 1.0)
# plt.legend()
# plt.xlabel("l")
# plt.ylabel("theta")
# plt.figure(figsize=(5, 5))
# fig.savefig("plots_doublet/theta(l).pdf", tight_layout=True)
#
# fig = plt.figure()
# plt.plot(l, u_x_array, label='y')
# # plt.xlim(-1, 36)
# # plt.axes().set_aspect('equal')
# # plt.ylim(-1.0, 1.0)
# plt.legend()
# plt.xlabel("l")
# plt.ylabel("u_x")
# plt.figure(figsize=(5, 5))
# fig.savefig("plots_doublet/x(l).pdf", tight_layout=True)
#
# fig = plt.figure()
# plt.plot(l, u_y_array, label='y')
# # plt.xlim(-1, 36)
# # plt.axes().set_aspect('equal')
# # plt.ylim(-1.0, 1.0)
# plt.legend()
# plt.xlabel("l")
# plt.ylabel("u_y")
# plt.figure(figsize=(5, 5))
# fig.savefig("plots_doublet/y(l).pdf", tight_layout=True)
#
# fig = plt.figure()
# plt.plot(l, landa_expected, color='k',linewidth=3, label='landa_expected')
# plt.plot(l, landa_ani, label='before PA')
# plt.plot(l, landa, label='after PA')
# # plt.xlim(-1, 36)
# # plt.axes().set_aspect('equal')
# # plt.ylim(-1.0, 1.0)
# plt.legend()
# plt.xlabel("l")
# plt.ylabel("landa")
# plt.figure(figsize=(5, 5))
# fig.savefig("plots_doublet/landa.pdf", tight_layout=True)
# # fig.savefig("landa.svg", tight_layout=True)
#
# fig = plt.figure()
# plt.plot(theta_deg, landa_expected, color='k', label='landa_expected')
# plt.plot(theta_deg, landa_ani, label='before PA')
# plt.plot(theta_deg, landa, label='after PA')
# # plt.xlim(-1, 36)
# # plt.axes().set_aspect('equal')
# # plt.ylim(-1.0, 1.0)
# plt.legend()
# plt.xlabel(r"$\theta$ [deg]")
# plt.ylabel("landa")
# plt.figure(figsize=(5, 5))
# fig.savefig("plots_doublet/landa(theta).pdf", tight_layout=True)
# # fig.savefig("landa.svg", tight_layout=True)
#
# fig = plt.figure()
# plt.plot(l, (landa_ani-landa_expected), color='k', label='landa_sim-landa_theo')
# # plt.xlim(-1, 36)
# # plt.axes().set_aspect('equal')
# # plt.ylim(-1.0, 1.0)
# plt.legend()
# plt.xlabel("l")
# plt.ylabel("landa")
# plt.figure(figsize=(5, 5))
# fig.savefig("plots_doublet/difference_landa.pdf", tight_layout=True)
#
#
# fig = plt.figure()
# plt.plot(l, activation_profile_sigma_x_array, label=r'')
# plt.axvline(x=17.5, ymin=0, ymax=1, linestyle=':', color='gray')
# # plt.xlim(-1, 36)
# # plt.axes().set_aspect('equal')
# plt.ylim(0.0, 0.2)
# plt.legend()
# plt.xlabel("l")
# plt.ylabel("PA act")
# plt.figure(figsize=(5, 5))
# fig.savefig("plots_doublet/activation_profile_sigma_x.pdf", tight_layout=True)
#
# fig = plt.figure()
# plt.plot(l, activation_profile_sigma_y_array, label=r'')
# plt.axvline(x=17.5, ymin=0, ymax=1, linestyle=':', color='gray')
# # plt.xlim(-1, 36)
# # plt.axes().set_aspect('equal')
# plt.ylim(0.0, 0.2)
# plt.legend()
# plt.xlabel("l")
# plt.ylabel("PA act")
# plt.figure(figsize=(5, 5))
# fig.savefig("plots_doublet/activation_profile_sigma_y.pdf", tight_layout=True)
#
# fig = plt.figure()
# plt.plot(l, activation_profile_landa_array,
#          label=r'$\mathrm{P}(l) = \frac{1}{20(1+e^{0.6(x-11.5)})}$')
# plt.axvline(x=17.5, ymin=0, ymax=1, linestyle=':', color='gray')
# # plt.xlim(-1, 36)
# # plt.axes().set_aspect('equal')
# # plt.ylim(-1.0, 1.0)
# plt.legend()
# plt.xlabel("l")
# plt.ylabel("PA act")
# plt.figure(figsize=(5, 5))
# fig.savefig("plots_doublet/activation_profile_landa.pdf", tight_layout=True)
#
# fig = plt.figure()
# plt.plot(u_x_array, epsilon_yy_array, label=r'')
# # plt.axvline(x=17.5,ymin=0,ymax=1,linestyle=':',color='gray')
# # plt.xlim(-1, 36)
# # plt.axes().set_aspect('equal')
# plt.ylim(-.07, 0.0)
# plt.legend()
# plt.xlabel("x")
# plt.ylabel("epsilon_yy")
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# plt.figure(figsize=(5, 5))
# fig.savefig("plots_doublet/strain_profile.pdf", tight_layout=True)
# fig.savefig("plots_doublet/strain_profile.svg", tight_layout=True)
#
# names = [r'$\lambda_l$', r'$\lambda_r$']
#
# values_landa = [landa_left, landa_right]
# values_landa_ani = [landa_left_ani, landa_right_ani]
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.scatter(names, values_landa_ani, marker='o', edgecolor='gray', facecolor='gray', label="before PA")
# plt.scatter(names, values_landa, marker='x', edgecolor='gray', facecolor='gray', label="after PA")
# # ax.plot(1, R1_str, 'ok',markersize=3)
# plt.xlim(-.5, 1.5)
# plt.legend()
# # plt.xlabel("l")
# textstr1 = '\n'.join((
#     r'$\lambda$ inc. left %s' % (format(relative_landa_increase_left * 100, '.4f')),
#     r'$\lambda$ inc. right %s' % (format(relative_landa_increase_right * 100, '.4f'))))
#
# # these are matplotlib.patch.Patch properties
# props = dict(boxstyle='round', facecolor='white', edgecolor='white', alpha=1)
# ax.text(0.75, 0.15, textstr1, transform=ax.transAxes, fontsize=6,
#         verticalalignment='top', bbox=props, linespacing=2)
# plt.ylabel(r"$\lambda$ [nN]")
# plt.figure(figsize=(5, 5))
# fig.savefig("plots_doublet/landa_before_and_after_PA.pdf", tight_layout=True)
#
#
# def fit_data():
#     x = u_x_array
#     y = u_y_array
#
#     coords = [[x[i], y[i]] for i in range(len(x))]
#     xc, yc, R, s = cf.least_squares_circle(coords)
#     arc = 2.0
#     phi = np.arange(0, arc * np.pi, 0.01)
#     x_circle = xc + R * np.cos(phi)
#     y_circle = yc + R * np.sin(phi)
#
#     # latexify(columns=1)
#     fig1 = plt.figure()
#     # number of rows, number of columns, plot number
#     ax = fig1.add_subplot(1, 1, 1)
#     # legend_elements = [Line2D([0], [0], color='k', lw=0.5, label=r'$\sigma_y > \sigma_x$'),
#     #                    Line2D([0], [0], color='k', lw=0.5,
#     #                           linestyle='--', label=r'$\sigma_y < \sigma_x$'),
#     #                    Line2D([0], [0], color='k', lw=0.5, linestyle=':', label=r'$\sigma_x = \sigma_y$')]
#
#     ax.plot(x, y, 'o')
#     ax.plot(x_circle, y_circle, linestyle='-', label='R = %s' % (R))
#     #
#     # ax.legend(['data', 'cubic', 'initial pos.'], loc='best')
#     # ax.set_ylim(20, 45)
#     ax.set_aspect('equal')
#     ax.set_ylabel(r" ")
#     ax.set_xlabel(r"")
#     legend = ax.legend(loc='best', framealpha=1)
#
#     fig1.tight_layout()
#     # format_axes(ax)
#     fig1.savefig("plots_doublet/circle_fit.pdf")
#     plt.show()

# fit_data()


#
# fig=plt.figure()
# plt.plot(np.linspace(0,35,35), y_halfstim, label=r'')
# plt.xlim(0, 35)
# plt.axes().set_aspect('equal')
# plt.ylim(-18.0, 18.0)
# plt.xlabel("x")
# plt.ylabel("y(x)")
# plt.figure(figsize=(5, 5))
# fig.savefig("contour_halfstim.pdf",tight_layout=True)
