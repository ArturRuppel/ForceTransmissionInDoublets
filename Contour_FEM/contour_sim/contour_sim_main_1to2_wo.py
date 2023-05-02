from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import circle_fit as cf
from scipy.interpolate import interp1d, CubicSpline, UnivariateSpline
from scipy import optimize
import pickle

keyDict = {"EA300"}

dict_allEA = dict([(key, []) for key in keyDict])
dict_allEA_with_fit = dict([(key, []) for key in keyDict])# {"EA50": [], "EA60": [],"EA70": [], "EA80": [],"EA90": [], "EA100": [],"EA150": [], "EA200": [],, "EA250": [], "EA300": []}
EA_array = [300]



s_x = 0.55
s_y = 0.5
a_exp = 150
b = a_exp*np.sqrt(s_y/s_x)
sigma = np.amax([s_x, s_y])
landa_0_exp = np.sqrt(s_x * s_y) * a_exp
landa_exp = 100 # nN


def doublet_fit(ea_input):
    d = 50

    def interpolate_contour(u_x_array, u_y_array, d):
        x = u_x_array
        y = u_y_array

        f = interp1d(x, y, kind='cubic')  # interpolated arc

        x_range = np.linspace(0, d, 200)

        return x_range, f(x_range)


    ea = ea_input


    # Create mesh and function space
    mesh = IntervalMesh(199, 0, d)
    # mesh = Mesh("mesh.xml")
    # Define function space for system of concentrations
    P1 = FiniteElement('CG', interval, 1)
    element = MixedElement([P1, P1])
    V = FunctionSpace(mesh, element)

    # Define test functions
    v_x, v_y = TestFunctions(V)

    tol = 1E-14

    def DirichletBoundary_y(x, on_boundary):
        return near(x[0], 0.0, tol) or near(x[0], d, tol)

    def DirichletBoundary_x_left(x, on_boundary):
        return near(x[0], 0.0, tol)

    def DirichletBoundary_x_right(x, on_boundary):
        return near(x[0], d, tol)

    # Define the isotropic model based on TEM Model with an additional active line tension
    u = Function(V)

    # calculate non-zero initial conditions
    R = 200 / 2
    L = 2 * R * np.arcsin(d / (2 * R))
    landa_act = 200 - 100 * (L / d - 1)
    print("Stretch ", L / d)
    print("Landa_act ", landa_act)
    u_x, u_y = split(u)
    sigma_x = Constant(2)
    sigma_y = Constant(2)
    EA = Constant(100)  # nN
    landa = Constant(200)
    nu = Constant(L / d)
    F = (- landa / nu * dot(grad(u_x), grad(v_x)) * dx + sigma_x * u_y.dx(0) * v_x * dx \
         - landa / nu * dot(grad(u_y), grad(v_y)) * dx - sigma_y * u_x.dx(0) * v_y * dx)

    # Define boundary condition
    u_x_0 = Constant(0.0)
    u_x_L = Constant(d)
    u_y_0_L = Constant(0.0)

    bc_y = DirichletBC(V.sub(1), u_y_0_L, DirichletBoundary_y)
    bc_x_left = DirichletBC(V.sub(0), u_x_0, DirichletBoundary_x_left)
    bc_x_right = DirichletBC(V.sub(0), u_x_L, DirichletBoundary_x_right)

    bcs = [bc_y, bc_x_left, bc_x_right]

    # solve the isotropic model
    solve(F == 0, u, bcs)
    # solver_parameters={"newton_solver":{"relative_tolerance": 1e-6}})

    u_save_init = Function(V)
    u_save_init.assign(u)
    ux_save_init, uy_save_init = u_save_init.split(deepcopy=True)
    u_x_array_init = ux_save_init.vector().get_local()
    u_y_array_init = uy_save_init.vector().get_local()


    def f(x, *args):
        u.assign(u_save_init)

        # fit only active lambda for a fixed modulus
        l_act = x[0]

        u_x, u_y = split(u)
        sigma_x = Constant(s_x)
        sigma_y = Constant(s_y)
        EA = Constant(ea)  # nN
        landa_act = l_act
        nu = sqrt(u_x.dx(0) ** 2 + u_y.dx(0) ** 2)
        landa = landa_act + EA * (sqrt(u_x.dx(0) ** 2 + u_y.dx(0) ** 2) - 1)
        # *sqrt((1 + (u_y.dx(0) / u_x.dx(0)) ** 2) / (1 + sigma_x / sigma_y * (u_y.dx(0) / u_x.dx(0)) ** 2))
        F = (- landa / nu * dot(grad(u_x), grad(v_x)) * dx + sigma_x * u_y.dx(0) * v_x * dx \
             - landa / nu * dot(grad(u_y), grad(v_y)) * dx - sigma_y * u_x.dx(0) * v_y * dx)

        # solver takes solution "u" from isotropic model as an initial guess and saves the new result to variable "u"
        solve(F == 0, u, bcs, solver_parameters={"newton_solver": {"relative_tolerance": 1e-6, "absolute_tolerance": 1e-6}})

        u_save_ani = Function(V)
        u_save_ani.assign(u)
        ux_save_ani, uy_save_ani = u_save_ani.split(deepcopy=True)
        u_x_array_ani = ux_save_ani.vector().get_local()
        u_y_array_ani = uy_save_ani.vector().get_local()

        Q = FunctionSpace(mesh, "DG", 0)
        u_xdx_ani = project(ux_save_ani.dx(0), Q)
        u_ydx_ani = project(uy_save_ani.dx(0), Q)

        u_xdx_ani_array = u_xdx_ani.vector().get_local()
        u_ydx_ani_array = u_ydx_ani.vector().get_local()
        tan_l0_squared = ((u_ydx_ani_array / u_xdx_ani_array) ** 2)[0]
        stretch_ani = np.sqrt(u_xdx_ani_array ** 2 + u_ydx_ani_array ** 2)
        print("THETA ", np.degrees(np.arctan(np.sqrt(tan_l0_squared))))
        theta = np.sqrt(tan_l0_squared)
        landa_ani = (l_act + ea * (stretch_ani - 1))[0]
        landa_0_theo = (l_act + ea * (stretch_ani - 1))[100]
        # *np.sqrt((1 + (u_ydx_ani_array / u_xdx_ani_array) ** 2) / (1 + s_x / s_y * (u_ydx_ani_array / u_xdx_ani_array) ** 2))
        print(landa_ani - (ea * (stretch_ani - 1))[0])
        print((ea * (stretch_ani - 1))[0])

        print("landa_ani = ", landa_ani)
        print("landa_exp = ", landa_exp)

        print("landa_0_theo = ", landa_0_theo)
        print("landa_0_exp = ", landa_0_exp)

        x_range, y_theo = interpolate_contour(u_x_array_ani, u_y_array_ani, d)
        y_halfstim_exp = np.loadtxt("contour_1to2.txt", skiprows=1)
        # theta = np.deg2rad(17.338777101253832)
        print("theta in rad ", theta)
        print("s_x ", s_x)
        print("s_y ", s_y)
        print("a_exp ", a_exp)
        landa_expected = np.sqrt(s_x * s_y) * a_exp * np.sqrt((1 + np.tan(theta) ** 2) / (1 + s_x / s_y * np.tan(theta) ** 2))
        print(landa_expected)
        dist = np.sum((y_halfstim_exp - y_theo) ** 2)  # +(landa_0_exp-landa_0_theo)**2),+(landa_exp-landa_ani)**2
        print("DIST SHAPE ", np.sum((y_halfstim_exp - y_theo) ** 2))
        print("DIST ", dist)


        np.savetxt("l_act_1to2.txt", x)
        # assign solution of one optimization step to u_save_init in order to set this as initial condition for the next iteration step
        u_save_init.assign(u)
        print(x)
        return dist

    # set up the solver
    ### Load parameters ###
    x0 = np.loadtxt("input.txt")
    args = ([''])

    ### Optimize ###
    res1 = optimize.minimize(f, x0, args=args, method='Nelder-Mead', options={'maxiter': 1000, 'disp': True, 'adaptive': True, 'fatol': 0.001})
    print(res1)

    l_act = np.loadtxt("l_act_1to2.txt")


    return ea,l_act











all_fitted_ea_l_act= []
for ea_input in EA_array:

    ea,l_act = doublet_fit(ea_input)

print(ea,l_act)
