from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import circle_fit as cf
from scipy.interpolate import interp1d, CubicSpline, UnivariateSpline
from scipy import optimize
import pickle

keyDict = {"EA50", "EA60","EA70", "EA80","EA90", "EA100","EA150", "EA200", "EA250", "EA300"}

dict_allEA = dict([(key, []) for key in keyDict])

keyDict_FB = {"FB0.05", "FB0.1","FB0.2", "FB0.3","Fb0.4", "FB0.5","FB0.6", "FB0.7", "FB0.8", "FB0.9"}

dict_allFeedback = dict([(key, []) for key in keyDict_FB])

Feedback_array = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
EA_array = [50,60,70,80,90,100,150,200,250,300]



s_x = 0.92
s_y = 1.12
a_exp = 61.94
b = a_exp*np.sqrt(s_y/s_x)
sigma = np.amax([s_x, s_y])
landa_0_exp = np.sqrt(s_x * s_y) * a_exp
landa_exp = 102.69293373872952 # nN

sx_increase = 0.042
sx_increase_CI = 0.072
sy_increase = 0.15
sy_increase_CI =  0.095
landa_increase = 0.0

with open('all_results/fullstim_doublet_simulation_strain_fit.pickle', "rb") as data:
    ds_fit = pickle.load(data)



def doublet_simulation(fb,sx_increase,sy_increase,ea_input,l_act_input):
    parameters['reorder_dofs_serial'] = False
    # Create mesh and function space
    mesh = IntervalMesh(199, 0, 35)
    l = np.linspace(0, 35, 200)
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

    l_act = l_act_input  # determined from text file of fit result
    ea = ea_input



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
    feedback_x = fb*sx_increase
    feedback_y = fb * sy_increase
    activation_profile_sigma_x = Expression("(amp-fb)*1/(1+exp(0.65*(x[0]-8.2)))+fb",degree=0,amp=sx_increase,fb=feedback_x)  # Expression("1/(1+exp(0.6*(x[0]-11.5)))/10 ",degree=0)#Expression("1/(1+exp(0.6*(x[0]-11.5)))/10 ",degree=0)
    activation_profile_sigma_y = Expression("(amp-fb)*1/(1+exp(0.65*(x[0]-8.2)))+fb",degree=0,amp=sy_increase,fb=feedback_y) # Expression("1/(1+exp(0.6*(x[0]-11.5)))/5 ",degree=0)#Expression("(0.1-0.09)*1/(1+exp(0.8*(x[0]-11.5)))+0.09",degree=0)#Expression("1/(1+exp(0.6*(x[0]-11.5)))/10 ",degree=0)#Constant(0.12)
    activation_profile_landa = Constant(landa_increase)  # Expression("1/(1+exp(0.6*(x[0]-11.5)))/20 ",degree=0)Constant(0.00)
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

    u_xdx_array = np.empty_like(np.linspace(0, 35, 200))
    u_ydx_array = np.empty_like(np.linspace(0, 35, 200))
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

    activation_profile_landa_array = np.empty_like(np.linspace(0, 35, 200))
    activation_profile_sigma_x_array = np.empty_like(np.linspace(0, 35, 200))
    activation_profile_sigma_y_array = np.empty_like(np.linspace(0, 35, 200))
    for v in vertices(mesh):
        activation_profile_landa_array[v.index()] = activation_profile_landa_interpol(*mesh.coordinates()[v.index()])
        activation_profile_sigma_x_array[v.index()] = activation_profile_sigma_x_interpol(*mesh.coordinates()[v.index()])
        activation_profile_sigma_y_array[v.index()] = activation_profile_sigma_y_interpol(*mesh.coordinates()[v.index()])

    landa = (l_act + l_act * activation_profile_landa_array + ea * (stretch - 1))
    print("Theta Ani 0 ", theta_ani[0])
    print("Theta Ani 0 degrees ", theta_ani_deg[0])
    print("Theta 0 ", theta[0])
    print("Theta 0 degrees ", theta_deg[0])
    print(landa)
    theta_after_opto = np.arctan(u_ydx_array / u_xdx_array)
    # print("Force balance x ", landa[0]*np.cos(theta_after_opto[0]) - landa[-1]*np.cos(theta_after_opto[-1]))
    # print("F_x left = ",landa[0]*np.cos(theta_after_opto[0]))
    # print("F_x right = ",landa[-1]*np.cos(theta_after_opto[-1]))
    print(theta_after_opto[0], theta_after_opto[-1])
    print(theta_after_opto)

    relative_landa_increase_left = (landa[0] - landa_ani[0]) / landa_ani[0]
    relative_landa_increase_right = (landa[-1] - landa_ani[-1]) / landa_ani[-1]
    landa_left = landa[0]
    landa_right = landa[-1]

    landa_left_ani = landa_ani[0]
    landa_right_ani = landa_ani[-1]
    print((ea * (stretch_ani - 1))[0])
    print((ea * (stretch - 1))[0])
    integral = assemble(nu * dx)
    print("Length of stretched fiber ", integral)
    cf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    region = AutoSubDomain(lambda x, on: x[0] <= 17.5)
    region.mark(cf, 1)
    dx_sub = Measure('dx', subdomain_data=cf)
    integral_sub = assemble(nu * dx_sub(0))
    print("Half length of stretched fiber ", integral_sub)

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
    landa_expected = np.sqrt(s_x * s_y) * a_exp * np.sqrt((1 + np.tan(theta_ani) ** 2) / (1 + s_x / s_y * np.tan(theta_ani) ** 2))
    y_halfstim = np.loadtxt("contour_doublet.txt", skiprows=1)


    data = {'exp_contour': y_halfstim,
            'l': l,
            'x_bPA': u_x_array_ani, 'y_bPA': u_y_array_ani,
            'x_aPA': u_x_array, 'y_aPA': u_y_array,
            'dxdl_bPA': u_xdx_ani_array, 'dydl_bPA': u_ydx_ani_array, 'dxdl_aPA': u_xdx_array, 'dydl_aPA': u_ydx_array,
            'act_profile_landa': activation_profile_landa_array, 'act_profile_s_x': activation_profile_sigma_x_array, 'act_profile_s_y': activation_profile_sigma_y_array,
            'stretch_bPA': stretch_ani, 'stretch_aPA': stretch,
            'theta_bPA': theta_ani_deg, 'theta_aPA': theta_deg,
            'epsilon_yy': epsilon_yy_array,
            'landa_expected': landa_expected, 'landa_bPA': landa_ani, 'landa_aPA': landa, 'l_act': l_act,'ea':ea,'sx_increase':sx_increase,'sy_increase':sy_increase,'fb':fb}

    dict_allFeedback['FB%s' % (fb)] = data

all_fitted_ea_l_act= []
for ea in EA_array:
    for fb in Feedback_array:
        l_act_input = ds_fit['EA%s' % (ea)]['l_act']
        sx_increase_input = ds_fit['EA%s' % (ea)]['sx_increase']
        sy_increase_input = ds_fit['EA%s' % (ea)]['sy_increase']

        doublet_simulation(fb, sx_increase_input, sy_increase_input, ea, l_act_input)
        dict_allEA['EA%s' % (ea)] = dict_allFeedback

        all_fitted_ea_l_act.append([fb, ea, l_act_input ,dict_allEA['EA%s'%(ea)]['FB%s'%(fb)]['sx_increase'],dict_allEA['EA%s'%(ea)]['FB%s'%(fb)]['sy_increase'],np.amin(dict_allEA['EA%s'%(ea)]['FB%s'%(fb)]['epsilon_yy'])])
        names = ["ea", 'l_act', "sx_increase", 'sy_increase', 'epsilon_yy middle']
        np.savetxt("all_fitted_ea_halfstim.txt", all_fitted_ea_l_act)


with open('all_results/halfstim_doublet_simulation.dat',"wb") as f:
    pickle.dump(dict_allEA, f,protocol=pickle.HIGHEST_PROTOCOL)

with open('all_results/halfstim_doublet_simulation.pickle',"wb") as f:
    pickle.dump(dict_allEA, f,protocol=pickle.HIGHEST_PROTOCOL)


