from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from scipy import optimize
from scipy.interpolate import griddata,Rbf
import pickle
import sys
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("rothemain.rothe_utils")
logging.getLogger('UFL').setLevel(logging.WARNING)
logging.getLogger('FFC').setLevel(logging.WARNING)

set_log_active(False)
tol_x = DOLFIN_EPS
tol_y = DOLFIN_EPS
def interpolate_simulation_maps_rbf(x, y, grid_x,grid_y,data1,data2):
    '''x_start, x_end, y_start and y_and are the coordinates of the 4 courners of the heatmap in micron and the pixelsize in micron per pixel
    determines how many pixel the stressmap will have after interpolation'''


    # data1_all[:, :] = griddata((x, y), data1, (grid_x, grid_y), method='linear',fill_value=0)
    # data2_all[:, :] = griddata((x, y), data2, (grid_x, grid_y), method='linear',fill_value=0)
    rbf1 = Rbf(x, y, data1,function='gaussian')
    rbf2 = Rbf(x, y, data2,function='gaussian')
    data1_all = rbf1(grid_x, grid_y)
    data2_all = rbf2(grid_x, grid_y)

    return data1_all, data2_all

# Strain
def eps(v):
    return sym(grad(v))

# Stress
def sigma(v, lmbda, mu):
    return 2.0 * mu * eps(v) + lmbda * tr(eps(v)) * Identity(len(v))

# Active stress tensor
def active(sx, sy, sxy=0.0):
    delta = Identity(2)
    return as_tensor([[sx, sxy], [sxy, sy]])#0.1 * delta[i,j], (i,j))

# Calculation of the penetration length on a thick substrate
def penetrationLength_thick_subs(Ec, hc,p):
    '''
    Paper: Banerjee & Marchetti (2012): Contractile stresses in cohesive cell layers on finite-thickness substrates
    '''
    # ka = 2.5e-3     # Stiffness of focal adhesion bonds [N/m]
    # L = 50e-6       # Cell length (1d), diameter (2d) [m]
    # lc0 = 1e-6      # Length of sarcomeric subunit [m]
    hs = p['hs']       # Thickness of the substrate [m]
    nus = p['vs']    # Poisson's ratio of the substrate
    Es = p['Es']    # Elastic modulus of the substrate [N/m^2]
    nuc = p['vc']
    Lc = p['Lc']
    #hc = 1e-6       # Thickness of the cell [m]
    heff = Lc#(1. / (hs * 2 * np.pi * (1 + nus)))**-1 # Oakes 2014 paper
    Ya = p['Ya']# N/m^3
    # Ys = Es/(2*(1+nus)*hs)# N/m^3
    Ys = np.pi*Es / heff  # N/m^3
    Y = (1.0 / Ys)**(-1) # N/m^3 1.0 / Ya +
    #print Ya, Ys
    lp = np.sqrt(Ec * hc / (Y*(1-nuc**2))) # m # Edwards und Schwarz
    return lp, Ys*1e-12,Y*1e-12 # return Y's with N/(m*um^2) gives strain energy in pJ

# only relevant if symmetry if pattern is present and if PA is on full cell
def DirichletBoundary(x, on_boundary):
    return near(x[0], 0.0, tol_x) and near(x[1], 0.0, tol_y)

def calculateTotalForce(u, kN, F, V, assigner_V_to_F, mesh):
    ux = Function(F)
    uy = Function(F)
    u0 = Function(V)
    u0.assign(u)
    u0.vector()[:] *= u0.vector()
    # Split so that ux = ux**2, uy = uy**2
    assigner_V_to_F.assign([ux, uy], u0)
    # ux will hold |u|**2 = ux**2 + uy**2
    ux.vector().axpy(1, uy.vector())
    # ux will hold |u|
    ux.vector().set_local(np.sqrt(ux.vector().get_local()))
    ux.vector().apply('')
    return assemble(kN*ux*dx(mesh))

def calculateStrainEnergy(u, kN, Ys, F, V, assigner_V_to_F, mesh):
    ux = Function(F)
    uy = Function(F)
    u0 = Function(V)
    u0.assign(u)
    u0.vector()[:] *= u0.vector()
    # Split so that ux = ux**2, uy = uy**2
    assigner_V_to_F.assign([ux, uy], u0)
    # ux will hold |u|**2 = ux**2 + 1 * uy**2
    ux.vector().axpy(1, uy.vector())
    #ux.vector().apply('')
    return assemble(0.5*kN**2/Ys*ux*dx(mesh))

# Definition of the position-dependent spring constants on an |-| shaped pattern
# class KNExpression(UserExpression):
#
#     def __init__(self, lmbdaE, muE, lp, armWidth, degree=2):
#         print("BIS HIER")
#         super().__init__()
#         self.lmbdaE = lmbdaE
#         self.muE = muE
#         self.lp = lp
#         self.armWidth = armWidth
#     def eval(self, value, x):
#         d = 45
#         if (x[0] <= -(d/2)+self.armWidth or x[0] >= (d/2)-self.armWidth or between(x[1],(-self.armWidth/2,self.armWidth/2))):
#             value[0] = (self.lmbdaE + 2 * self.muE) / (self.lp * 1e6)**2
#         else:
#             value[0] = 0.0

class KNExpression(UserExpression):

    def __init__(self,Y, armWidth, degree=2):
        print("BIS HIER")
        super().__init__()
        self.armWidth = armWidth
        self.Y =Y
    def eval(self, value, x):
        d = 2*np.sqrt(45*45/2)
        if (x[0] <= -(d/2)+self.armWidth or x[0] >= (d/2)-self.armWidth or between(x[1],(-self.armWidth/2,self.armWidth/2))):
            value[0] = self.Y
        else:
            value[0] = 0.0
# ----------------------------------------------------------------------------------------------------------------------
# main adjustments of script parameters are done here
ar = "ar1to2d"

p = {'Ec':10*1e3,'Etac':100e3,'hc': 1e-6, 'vc': 0.5,'Es':20e3, 'hs': 50e-6, 'vs': 0.5,'Ya':1e9,'AIC':0.46,'Lc':50e-6}
# define the main optimization function f
# f returns the distance between experimental strain energy and simulated strain energy from FEM
def f(x_init, *args):

    # __________________________________________________________________________________________ Loading Data and Params
    # Load experimental data
    # exp_data = np.loadtxt("mean_energy_baseline_%s.txt" % (ar), skiprows=1)
    with open('raw_data/mean_stress_and_Es_low_protocol.dat', "rb") as data:
        ds = pickle.load(data)

    ar1to2d = ds['1to2d_halfstim']
    exp_data = ar1to2d['Es'][0:19]
    T_array = np.arange(0,60)[0:19]
    sigma_x_left_data = ar1to2d['sigma_xx_left'][0:19]
    sigma_y_left_data = ar1to2d['sigma_yy_left'][0:19]
    sigma_left = np.sqrt(sigma_x_left_data ** 2 + sigma_y_left_data ** 2)

    # read in the fit parameters which are stored in x
    s= x_init[0]
    print("!!!!!!!!!!!!!!!!!!!!!! ",s)

    # Define mesh
    mesh = Mesh('h12.xml')
    # Extract coordinates of mesh points
    coords = mesh.coordinates()

    # _________________________________________________________________________________________ Set Fixed cell paramters
    E3D = p['Ec'] # Elastic modulus Pa
    eta3D = p['Etac'] # Viscous modulus Pa/s


    h = p['hc']  # m
    # two dimensional stresses i.e. surface tensions

    sigma_x = s  # N/m
    sigma_y = s*(1-p['AIC'])/(1+p['AIC']) # N/m
    # conversion to 2D constants for plane stress and thin layer approximation
    Eh = E3D * h  # N / m = Pa * m
    etah = eta3D * h  # Ns / m = Pa * m
    nu = 0.5
    lmbdaE = Eh * nu / ((1 - nu) * (1 + nu))  # for 3d: ((1 + nu) * (1 - 2 * nu))
    muE = Eh / (2 * (1 + nu))
    lmbdaEta = etah * nu / ((1 - nu) * (1 + nu))  # for 3d: ((1 + nu) * (1 - 2 * nu))
    muEta = etah / (2 * (1 + nu))

    lp, Ys, Y = penetrationLength_thick_subs(E3D, h,p)  # Unit [m]. Here, only Ys is calculated!
    # Ys = 0.00031
    # lp = 0.97e-06  # [m] fit parameter of baseline optimization

    armWidth = 5.0 # um width of the H-bars of the micro pattern
    kN = KNExpression(Y, armWidth,degree=2) # Spring stiffness density kN in [N/m/ um**2] -> get u in [um]

    print("Ec: ",p['Ec']*1e-3,'kPa')
    print("hc: ", p['hc'] * 1e6, 'um')
    print("Es: ", p['Es'] * 1e-3, 'kPa')
    print("hs: ", p['hs'] * 1e6, 'um')
    print("Spring constant density substrate:", Ys, "N/(m*um^2)")
    print("Spring constant density total:", Y, "N/(m*um^2)")
    print("Localization length:", lp * 1e6, "um")
    print("Total cell area:", assemble(Constant(1.0) * dx(mesh)), "um**2")


    # _____________________________________________________________________________________________ Define Time Stepping
    dt = Constant(60) # Time constant [s]
    T = T_array[-1]*60+60# Total simulation time [s] for Full Circle experiment

    # __________________________________________________________________________Define lag time and opto activation time
    lag_time = 12 * etah / Eh  # The cell needs this time to arrive in its 'ground state'.
    act_times = np.array([1200, T + 60]) + lag_time  # Time points of photoactivation stress

    # _________________________________________________________________________Define function space and basis functions
    V = VectorFunctionSpace(mesh, "CG", 2)
    u = TrialFunction(V)
    v = TestFunction(V)

    # _________________________________________________________________________________________Define boundary condition
    u0 = Constant((0.0, 0.0))
    bc = DirichletBC(V, u0, DirichletBoundary)#,method="pointwise")

    # ___________________________________________________________________________________________Define variational form
    a = inner(sigma(u, lmbdaEta, muEta), sym(grad(v)))*dx + dt*inner(sigma(u, lmbdaE, muE), sym(grad(v)))*dx + dt * kN * inner(u, v) * dx
    u = Function(V)
    uinit = Constant((0.0, 0.0))
    uold = interpolate(uinit, V)
    uold.assign(u)

    F = FunctionSpace(mesh, 'CG', 2)
    assigner_V_to_F = FunctionAssigner([F, F], V)
    # _________________________________________________________ Define Elements and Function Space for Resulting Tensors
    dFE = FiniteElement("CG", mesh.ufl_cell(), 2)
    tFE = TensorElement(dFE)
    W = FunctionSpace(mesh, tFE)
    K = FunctionSpace(mesh, dFE)
    stressNorm = Function(K, name="StressNorm")
    stress = Function(W, name='Stress')
    passive_stress = Function(W, name='PassiveStress')
    disp = Function(V, name='Displacement')
    dispSubstrate = Function(V, name='Displacement Substrate')

    # _________________________Determine the save options and save resulting fields to output.xdmf to view with ParaView
    xdmf_file= XDMFFile("output.xdmf")
    xdmf_file.parameters["flush_output"] = True
    xdmf_file.parameters["functions_share_mesh"] = True
    save = True

    # Initialize lists to save simulation results
    all_times = []
    all_energies = []
    all_stresses = []
    all_stresses_left = []

    # Run simulation
    progress = ProgressBar(maxval=np.ceil(((T+lag_time)/dt)(0.0))).start()
    t = 0 * dt
    lag_counter = 0
    act_flag = False        # True, if activated
    act_time = 0.0
    act_stressX, act_stressY, act_stressXY = 0.0, 0.0, 0.0

    sigma0Xh = 0.0
    sigma0Yh = 0.0
    # main simulation loop
    while t(0.0) < T + lag_time:
        if t(0.0) < lag_time:
            lag_counter += 1
        elif near(t(0.0), act_times[0]) and act_flag == False:
            act_flag = True

        if act_flag == True:
            # print("GOT ACTIVATED")
            act_time = act_times[0]
            act_stressX = act_stress
            act_stressY = act_stress

            newT = t(0.0) - act_time
            sigma0Xh = act_stressX / (1 + np.exp(-(newT - center_act)/ tau_stress_act)) * (1 - 1 / (1 + np.exp(-(newT - center_rel)/ tau_stress_rel)))
            sigma0Yh = act_stressY / (1 + np.exp(-(newT - center_act)/ tau_stress_act)) * (1 - 1 / (1 + np.exp(-(newT - center_rel)/ tau_stress_rel)))

        if sigma0Xh < 1e-08:
            sigma0Xh = 0.0
        if sigma0Yh < 1e-08:
            sigma0Yh = 0.0

        # Right side of variational form
        L = inner(sigma(uold, lmbdaEta, muEta),sym(grad(v)))*dx - \
            dt * inner(active(sigma0Xh+sigma_x, sigma0Yh+sigma_y), sym(grad(v))) * dx
        # Solve problem with boundary conditions bc
        solve(a == L, u, bc)
        uold.assign(u)
        total_energy = calculateStrainEnergy(u, kN, Ys, F, V, assigner_V_to_F, mesh)   # Unit [pJ]
        all_times.append(t(0.0))
        all_energies.append(total_energy)
        all_stresses.append(sigma0Yh)

        # calculate total stress/strain tensors
        eps = sym(grad(u))
        sig = active(sigma0Xh+sigma_x, sigma0Yh+sigma_y) + Eh/(1+nu)*eps + nu*Eh/(1-nu**2)*tr(eps)*Identity(2)
        sig_passive = Eh / (1 + nu) * eps + nu * Eh / (1 - nu ** 2) * tr(eps) * Identity(2)
        stress.assign(project(sig, W))
        passive_stress.assign(project(sig_passive, W))
        disp.assign(u)
        dispSubstrate.assign(project(u*kN/Ys, V))
        stressNorm.assign(project(sqrt(inner(sig, sig)), K)) # Frobeniusnorm


        # mesh_points = mesh.coordinates()
        # _x = mesh_points[:, 0]
        # _y = mesh_points[:, 1]
        #
        # stress_x = stress.sub(0)
        # stress_y = stress.sub(3)
        # stress_x_array = np.array([stress_x(Point(_x, _y)) for _x, _y in mesh_points])
        # stress_y_array = np.array([stress_y(Point(_x, _y)) for _x, _y in mesh_points])
        #
        # # interpolate on regular grid
        # x_start = -31.82  # left most x-value in micron
        # x_end = 31.82  # right most x-value in micron
        # y_start = -15.91  # left most y-value in micron
        # y_end = 15.91  # right most y-value in micron
        # pixelsize = 0.4  # desired final pixelsize in micron per pixel
        # #
        # grid_x, grid_y = np.mgrid[x_start:x_end:(x_end - x_start) / pixelsize * 1j, y_start:y_end:(y_end - y_start) / pixelsize * 1j]
        # sigma_xx_int, sigma_yy_int = interpolate_simulation_maps_rbf(_x, _y, grid_x, grid_y,stress_x_array,stress_y_array)
        # sigma_xx_int, sigma_yy_int = sigma_xx_int.T, sigma_yy_int.T
        # sigma_sim = np.sqrt(sigma_xx_int**2+sigma_yy_int**2)
        # sigma_sim_left = np.nanmedian(sigma_sim[:,0:int(159/2)])
        # # print(sigma_xx_int)
        # # print(sigma_yy_int)
        # all_stresses_left.append(sigma_sim_left)
        # print('sigma_sim_left ',sigma_sim_left)
        # print('sigma_left ', sigma_left)

        # save tensors at each time step
        if save:
            # print("TIME ", t(0.0))
            xdmf_file.write(disp, t(0.0))
            xdmf_file.write(dispSubstrate, t(0.0))
            xdmf_file.write(stress, t(0.0))
            xdmf_file.write(passive_stress, t(0.0))
            xdmf_file.write(stressNorm, t(0.0))

        progress.update((t/dt)(0.0))
        t += dt
    progress.finish()

    # __________________________________________________________________________________________________Calculate result
    theo_times = np.array(all_times)[lag_counter:] - lag_time  # all times without lag time
    theo_energies = np.array(all_energies)[lag_counter:]
    # theo_stresses_left = np.array(all_stresses_left)[lag_counter:]
    # dist_stress = np.sum((theo_stresses_left[:] - sigma_left[:]) ** 2)  # dist is the variable which is minimized
    dist_energy = np.sum((theo_energies[:] - exp_data[:] * 1e12) ** 2)  # dist is the variable which is minimized
    dist = dist_energy
    # print("Baseline: ", np.mean(theo_energies[0::21]))
    print("Distance: ", dist)

    # fig, ax = plt.subplots()
    # fig.tight_layout()
    # ax.plot(theo_times, theo_stresses_left, label="Model")
    # ax.plot(T_array * 60., sigma_left[:], label="Data")
    # ax.set_xlabel("Time [sec]")
    # ax.set_ylabel("Stress Left [pJ]")
    # ax.legend(loc=0)
    # fig.savefig("Stress_left.pdf", bbox_inches='tight')

    fig, ax = plt.subplots()
    fig.tight_layout()
    ax.plot(theo_times, theo_energies, label="Model")
    ax.plot(T_array * 60., exp_data[:] * 1e12, label="Data")
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Strain energy [pJ]")
    ax.legend(loc=0)
    fig.savefig("Energies.pdf", bbox_inches='tight')

    # ____________________________________________________________________save computed strain energy and updated params
    np.savetxt("strain_energy_%s.txt"%(ar), np.array([np.array(all_times)-lag_time, all_energies]).T, \
               header="Time\tModel")
    np.savetxt("updated_params_baseline_%s.txt" % (ar), x_init)
    print('Maximal Displacement field:',np.nanmax(u.vector().get_local()),"um")
    return dist

# Main function
if __name__ == "__main__":
    ### Load parameters
    x0 = np.loadtxt("initial_params_baseline_%s.txt"%(ar))
    args = (p)

    # ________________________________________________________________________________________________________Optimize f
    res1 = optimize.minimize(f, x0, args=args, method='Nelder-Mead', options={'maxiter': 1000, 'disp': True,'fatol': 0.000001})
    print(res1)





    ### Single iteration and plot ###
##    theo_times,theo_energies = f(x0,[''])
##    exp_data = np.loadtxt("area_traces/mean_energy_full_circle_1000.txt", skiprows=1)
##    fig, ax = plt.subplots()
##    fig.tight_layout()
##    ax.plot(theo_times, theo_energies, label="Model")
##    ax.plot(exp_data[:, 0]*60., exp_data[:, 1]*1e12, label="Data")
##    ax.set_xlabel("Time [sec]")
##    ax.set_ylabel("Strain energy [pJ]")
##    ax.legend(loc=0)
##    fig.savefig("Energies.pdf",bbox_inches='tight')
