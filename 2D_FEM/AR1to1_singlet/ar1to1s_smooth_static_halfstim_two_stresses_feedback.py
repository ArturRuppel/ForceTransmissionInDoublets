from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from scipy import optimize
import sys
import logging
import pickle
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("rothemain.rothe_utils")
logging.getLogger('UFL').setLevel(logging.WARNING)
logging.getLogger('FFC').setLevel(logging.WARNING)
parameters['reorder_dofs_serial'] = False
set_log_active(False)
tol_x = DOLFIN_EPS
tol_y = DOLFIN_EPS

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

def calculateStrainEnergy_subdomain(u, kN, Ys, F, V, assigner_V_to_F, domain_measure):
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
    return assemble(0.5*kN**2/Ys*ux*domain_measure)


def calculate_ux_and_uy(u,F, V, assigner_V_to_F, mesh):
    ux = Function(F)
    uy = Function(F)
    u0 = Function(V)
    u0.assign(u)
    assigner_V_to_F.assign([ux, uy], u0)

    return ux,uy

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
        d = 45
        if (x[0] <= -(d/2)+self.armWidth or x[0] >= (d/2)-self.armWidth or between(x[1],(-self.armWidth/2,self.armWidth/2))):
            value[0] = self.Y
        else:
            value[0] = 0.0

def new_boundary(xval, u, l0):
    newX = xval + u
    newBoundary = xval[np.where(newX <= l0)[0][-1]]
    return newBoundary

# ----------------------------------------------------------------------------------------------------------------------
# main adjustments of script parameters are done here
ar = "ar1to1s"

s = np.loadtxt("updated_params_baseline_%s.txt"%(ar))

p = {'Ec':10*1e3,'Etac':100e3,'hc': 1e-6, 'vc': 0.5,'Es':20e3, 'hs': 50e-6, 'vs': 0.5,'Ya':1e9,'AIC':0.4,'Lc':50e-6,'s':s}


# define the main optimization function f
# f returns the distance between experimental strain energy and simulated strain energy from FEM
def f(fb,*args):


    tau_stress_act, tau_stress_rel, center, act_stress = x0
    # __________________________________________________________________________________________ Loading Data and Params
    print("Feedback is ",fb, "Act stress is ", act_stress)
    # Define mesh
    mesh = Mesh('h.xml')
    # Extract coordinates of mesh points
    coords = mesh.coordinates().T

    # x_ = Expression("x[0]", degree=0)
    # xval = x_.compute_vertex_values(mesh)

    # _________________________________________________________________________________________ Set Fixed cell paramters
    E3D = p['Ec']  # Elastic modulus Pa
    eta3D = p['Etac']  # Viscous modulus Pa/s

    h = p['hc']  # m

    # two dimensional stresses i.e. surface tensions
    sigma_x = p['s']  # N/m
    sigma_y = p['s'] * (1 - p['AIC']) / (1 + p['AIC'])  # N/m
    # conversion to 2D constants for plane stress and thin layer approximation
    Eh = E3D * h  # N / m = Pa * m
    etah = eta3D * h  # Ns / m = Pa * m
    nu = 0.5
    lmbdaE = Eh * nu / ((1 - nu) * (1 + nu))  # for 3d: ((1 + nu) * (1 - 2 * nu))
    muE = Eh / (2 * (1 + nu))
    lmbdaEta = etah * nu / ((1 - nu) * (1 + nu))  # for 3d: ((1 + nu) * (1 - 2 * nu))
    muEta = etah / (2 * (1 + nu))

    lp, Ys, Y = penetrationLength_thick_subs(E3D, h, p)  # Unit [m].

    armWidth = 5.0  # um width of the H-bars of the micro pattern
    kN = KNExpression(Y, armWidth, degree=2)  # Spring stiffness density kN in [N/m/ um**2] -> get u in [um]
    print("Ec: ", p['Ec'] * 1e-3, 'kPa')
    print("hc: ", p['hc'] * 1e6, 'um')
    print("Es: ", p['Es'] * 1e-3, 'kPa')
    print("hs: ", p['hs'] * 1e6, 'um')
    print("Spring constant density substrate:", Ys, "N/(m*um^2)")
    print("Spring constant density total:", Y, "N/(m*um^2)")
    print("Localization length:", lp * 1e6, "um")
    print("Total cell area:", assemble(Constant(1.0) * dx(mesh)), "um**2")


    # _____________________________________________________________________________________________ Define Time Stepping
    dt = Constant(60) # Time constant [s]
    T = T_array[-1] * 60 + 60# Total simulation time [s] for Full Circle experiment

    # __________________________________________________________________________Define lag time and opto activation time
    lag_time = 12 * etah / Eh  # The cell needs this time to arrive in its 'ground state'.
    act_times = np.array([1200, T + 60]) + lag_time  # Time points of photoactivation stress

    # _________________________________________________________________________Define function space and basis functions
    V = VectorFunctionSpace(mesh, "CG", 2)
    u = TrialFunction(V)
    v = TestFunction(V)

    # _________________________________________________________________________________________Define boundary condition
    u0 = Constant((0.0, 0.0))
    bc = []

    # ___________________________________________________________________________________________Define variational form
    a = inner(sigma(u, lmbdaEta, muEta), sym(grad(v)))*dx + dt*inner(sigma(u, lmbdaE, muE), sym(grad(v)))*dx + dt * kN * inner(u, v) * dx
    u = Function(V)
    uinit = Constant((0.0, 0.0))
    uold = interpolate(uinit, V)
    uold.assign(u)

    F = FunctionSpace(mesh, 'CG', 2)
    assigner_V_to_F = FunctionAssigner([F, F], V)
    # _________________________________________________________ Define Elements and Function Space for Resulting Tensors
    dFE = FiniteElement("DG", mesh.ufl_cell(), 0)
    tFE = TensorElement(dFE)
    W = FunctionSpace(mesh, tFE)
    K = FunctionSpace(mesh, dFE)



    stressNorm = Function(K, name="StressNorm")
    stress = Function(W, name='Stress_%s'%(fb))
    stress_left = Function(W, name='StressLeft')
    stress_right = Function(W, name='StressRight')
    passive_stress = Function(W, name='PassiveStress_%s'%(fb))
    disp = Function(V, name='Displacement_%s'%(fb))
    defGrad = Function(W, name='DeformationGradientTensor')
    Jacobian = Function(K, name='detF')
    activationProfile = Function(K, name="ActivationProfile_%s"%(fb))
    springstiffness_density = Function(V, name='kN_%s'%(fb))
    Traction = Function(V, name='Traction')
    dispSubstrate = Function(V, name='Displacement Substrate_%s'%(fb))


    save = True

    # Initialize lists to save simulation results
    all_times = []
    all_energies = []
    all_energies_left = []
    all_energies_right = []
    all_stresses = []

    stressNorm_list = []
    stress_xx_list = []
    stress_yy_list = []

    stress_xx_left_list = []
    stress_yy_left_list = []

    stress_xx_right_list = []
    stress_yy_right_list = []

    # Run simulation
    progress = ProgressBar(maxval=np.ceil(((T+lag_time)/dt)(0.0))).start()
    t = 0 * dt
    lag_counter = 0
    act_flag = False        # True, if activated
    act_time = 0.0
    act_stressX, act_stressY, act_stressXY = 0.0, 0.0, 0.0

    sigma0Xh = 0.0
    sigma0Yh = 0.0



    l0 = -9.32

    act_counter = 0
    g = Constant(0.0)


    # main simulation loop
    while t(0.0) < T + lag_time:
        print("TIME ", t(0.0))
        print("LAG TIME ", lag_time)
        print("TIME WITHOUT LAGTIME ", t(0.0) - lag_time)

        if t(0.0) < lag_time:
            lag_counter += 1
        if near(t(0.0), act_times[0]) and act_flag == False:
            act_flag = True
            act_counter += 1

        if act_flag == True:
            print("GOT ACTIVATED")
            act_time = act_times[0]
            #
            g = Expression("(amp-fb)*1/(1+exp(a*(x[0]+ux-b)))+fb",degree=0,a=0.64978052,amp=1,b=l0,fb=fb,ux=u.sub(0),domain=mesh)#Expression("s/(1+exp(a*(x[0]+ux-b))) ", degree=0, s=1, a=0.64978052, b=l0,ux=u.sub(0),domain=mesh)
            # g = Expression("s/(1+exp(a*(x[0]-b))) ", degree=0, s=1, a=0.64978052, b=l0, ux=u.sub(0), domain=mesh)
            act_stressX = act_stress * g
            act_stressY = act_stress * g

            newT = t(0.0) - act_time
            print('acttime ', act_time)
            print("Time since last activation point", newT)
            sigma0Xh = act_stressX * (1 - np.exp(-newT / tau_stress_act)) * (1 - 1 / (1 + np.exp(-(newT - center) / tau_stress_rel)))
            sigma0Yh = act_stressY * (1 - np.exp(-newT / tau_stress_act)) * (1 - 1 / (1 + np.exp(-(newT - center) / tau_stress_rel)))


        # if sigma0Xh < 1e-08:
        #     sigma0Xh = 0.0
        # if sigma0Yh < 1e-08:
        #     sigma0Yh = 0.0

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
        sig = active(sigma0Xh + sigma_x, sigma0Yh + sigma_y) + Eh / (1 + nu) * eps + nu * Eh / (1 - nu ** 2) * tr(
            eps) * Identity(2)
        sig_passive = Eh / (1 + nu) * eps + nu * Eh / (1 - nu ** 2) * tr(eps) * Identity(2)
        I = Identity(2)  # Identity tensor
        F_tensor = I + grad(u)  # Deformation gradient
        J = det(F_tensor)
        T_stress = kN * u * 1e3  # traction stress kPa

        stress.assign(project(sig, W))
        passive_stress.assign(project(sig_passive, W))
        disp.assign(u)

        stressNorm.assign(project(sqrt(inner(sig, sig)), K))  # Frobeniusnorm
        defGrad.assign(project(F_tensor, W))
        Jacobian.assign(project(J, K))
        springstiffness_density.assign(project(kN, K))
        activationProfile.assign(project(g, K))
        # assign(principalStressVector, [stress.sub(0),stress.sub(3)])
        Traction.assign(project(T_stress, V))
        dispSubstrate.assign(project(u * kN / Ys, V))

        mesh_tol = (mesh.hmax() / 2 * np.sqrt(3))

        # define sub_domain for both cells
        class LeftObstacle(SubDomain):
            def inside(self, x, on_boundary):
                return between(x[0],(-22.5,0))

        class RightObstacle(SubDomain):
            def inside(self, x, on_boundary):
                return between(x[0],(0,22.5))

        print(mesh_tol)
        subdomains = MeshFunction('size_t', mesh, 2)
        # subdomains = MeshFunction("size_t", mesh, "h_interface_physical_region.xml")
        subdomains.set_all(0)
        # Mark subdomains with numbers 1 and 2
        subdomain1 = LeftObstacle()
        subdomain2 = RightObstacle()
        subdomain1.mark(subdomains, 1)
        submesh1 = SubMesh(mesh, subdomains, 1)
        subdomain2.mark(subdomains, 2)
        submesh2 = SubMesh(mesh, subdomains, 2)

        dx_new = Measure('dx', subdomain_data=subdomains)
        # dx_sub_right = Measure('dx', subdomain_data=subdomains)

        print("Halfcell area left:",assemble(Constant(1) * dx(submesh1)), "um**2")
        print("Halfcell area right:",assemble(Constant(1) * dx(submesh2)), "um**2")
        # print("Halfcell area left:", assemble(const*dx(submesh1,metadata={"quadrature_rule":"vertex","quadrature_degree":1})), "um**2")
        # print("Halfcell area right:", assemble(const*dx(submesh2, metadata={"quadrature_rule":"vertex","quadrature_degree":1})),"um**2")

        total_energy_left = calculateStrainEnergy_subdomain(u, kN, Ys, F, V, assigner_V_to_F, dx(submesh1))
        total_energy_right = calculateStrainEnergy_subdomain(u, kN, Ys, F, V, assigner_V_to_F, dx(submesh2))
        print(total_energy_left)
        print(total_energy_right)
        all_energies_left.append(total_energy_left)
        all_energies_right.append(total_energy_right)

        # for cell_no in range(len(subdomains.array())):
        #     subdomain_no = subdomains.array()[cell_no]
        #     if subdomain_no ==1:
        #         stress_left.vector()[cell_no] = [0.0,0.0]
        #     if subdomain_no ==0:
        #         stress_right.vector()[cell_no] = [0.0,0.0]


        stressNorm_list.append(stressNorm.vector().get_local())
        stress_xx_list.append(stress.sub(0).vector().get_local())
        stress_yy_list.append(stress.sub(3).vector().get_local())

        # save tensors at each time step
        if save:
            print("TIME ", t(0.0))
            xdmf_file.write(disp, t(0.0))
            xdmf_file.write(stress, t(0.0))
            xdmf_file.write(passive_stress, t(0.0))
            # xdmf_file.write(stressNorm, t(0.0))
            xdmf_file.write(activationProfile, t(0.0))
            xdmf_file.write(springstiffness_density, t(0.0))
            xdmf_file.write(dispSubstrate, t(0.0))
            # xdmf_file.write(defGrad, t(0.0))
            # xdmf_file.write(Jacobian, t(0.0))
            # xdmf_file.write(Traction, t(0.0))

        progress.update((t / dt)(0.0))
        t += dt
    progress.finish()

    # __________________________________________________________________________________________________Calculate result
    theo_times = np.array(all_times)[lag_counter:] - lag_time  # all times without lag time
    theo_energies = np.array(all_energies)[lag_counter:]
    dist = np.sum((theo_energies[:] - exp_data[:] * 1e12) ** 2)  # dist is the variable which is minimized
    print("Baseline: ", np.mean(theo_energies[0::21]))
    print("Distance: ", dist)

    fig, ax = plt.subplots()
    fig.tight_layout()
    ax.plot(theo_times, theo_energies, label="Model")
    ax.plot(T_array * 60., exp_data[:] * 1e12, label="Data")
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Strain energy [pJ]")
    ax.legend(loc=0)
    fig.savefig("Energies.pdf", bbox_inches='tight')
    # ____________________________________________________________________save computed strain energy and updated params
    np.savetxt("strain_energy_halfstim%s.txt"%(ar), np.array([np.array(all_times)-lag_time, all_energies]).T, \
               header="Time\tModel")

    # # np.savetxt("updated_params_%s.txt" % (ar), x)
    #
    # stressNorm_array = np.array(stressNorm_list)[lag_counter:]
    # stress_xx_array = np.array(stress_xx_list)[lag_counter:]
    # stress_yy_array = np.array(stress_yy_list)[lag_counter:]
    # strain_energy_left_array = np.array(all_energies_left)[lag_counter:]
    # strain_energy_right_array = np.array(all_energies_right)[lag_counter:]
    #
    # np.savez('stress_xx_yy_halfstim_%s.npz' % (ar), stress_xx=stress_xx_array, stress_yy=stress_yy_array,time=theo_times)  # N/m,sec
    #
    # np.savez('stress_norm_halfstim_%s.npz' % (ar), stress=stressNorm_array, time=theo_times)  # N/m,sec
    np.savez('strain_energy/strain_energy_halfstim_%s_%s.npz' % (ar,fb), energy=theo_energies, time=theo_times)  # pJ,sec
    # np.savez('strain_energy_left_halfstim_%s.npz' % (ar), energy=strain_energy_left_array, time=theo_times)  # pJ,sec
    # np.savez('strain_energy_right_halfstim_%s.npz' % (ar), energy=strain_energy_right_array, time=theo_times)  # pJ,sec

    # np.savez('smooth_exp_constants_%s.npz' % (ar), constants=np.array(x))

    return dist

# Main function
if __name__ == "__main__":
    ### Load parameters
    x0 = np.loadtxt("updated_params_smoothexp_two_stresses_%s.txt"%(ar))

    # Load experimental data
    # exp_data = np.loadtxt("mean_energy_%s.txt" % (ar), skiprows=1)
    with open('raw_data/mean_stress_and_Es_low_protocol.dat', "rb") as data:
        ds = pickle.load(data)

    ar1to1s = ds['1to1s_fullstim']
    exp_data = ar1to1s['Es']
    T_array = np.arange(0, 60)
    sigma_x_data = ar1to1s['sigma_xx'][0:19]
    sigma_y_data = ar1to1s['sigma_yy'][0:19]

    # read in the fit parameters which are stored in x
    tau_stress_act, tau_stress_rel, center, act_stress = x0

    # _________________________Determine the save options and save resulting fields to output.xdmf to view with ParaView
    xdmf_file = XDMFFile("output_halfstim_lab_two_stresses.xdmf")
    xdmf_file.parameters["flush_output"] = True
    xdmf_file.parameters["functions_share_mesh"] = True

    # two dimensional stresses i.e. surface tensions
    sigma_x = np.mean(sigma_x_data)  # N/m
    sigma_y = np.mean(sigma_y_data)  # N/m
    args = (p,x0)
    Feedback_array = [-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for fb in Feedback_array:
        print("Simulation for feedback ",fb)
        f(fb)
    # # ________________________________________________________________________________________________________Optimize f
    # res1 = optimize.minimize(f, x0, args=args, method='Nelder-Mead', options={'maxiter': 1000, 'disp': True})
    # print(res1)





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
