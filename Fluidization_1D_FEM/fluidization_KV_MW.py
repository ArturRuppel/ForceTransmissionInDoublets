from dolfin import *
import numpy as np
import copy
import matplotlib.pyplot as plt
parameters['reorder_dofs_serial'] = False
def initialize_dirichlet_boundaries(V,boundary_fct=None):
    if boundary_fct == None:
        def DirichletBoundary(x, on_boundary):
            return False
        bc = DirichletBC(V, Constant(0.0), DirichletBoundary)
        return bc
    elif boundary_fct == "on_boundary":
        def DirichletBoundary(x, on_boundary):
            return on_boundary
    elif boundary_fct == "left_boundary":
        def DirichletBoundary(x, on_boundary):
            return near(x[0], xmin, tol)
# Strain
def eps(v):
    return grad(v)
# Stress
def sigma_E(v,EA):
    return EA*eps(v)

def sigma_V(v,EA,etaA):
    return etaA*eps(v)

class Left(SubDomain):
    def __init__(self, l0, tol=DOLFIN_EPS):
        self.l0 = l0
        self.tol = tol
        SubDomain.__init__(self)
    def inside(self, x, on_boundary):
        return near(x[0], self.l0, self.tol)

class Right(SubDomain):
    def __init__(self, l0, tol=DOLFIN_EPS):
        self.l0 = l0
        self.tol = tol
        SubDomain.__init__(self)
    def inside(self, x, on_boundary):
        return near(x[0], self.l0, self.tol)

class Interior(SubDomain):
    def __init__(self, l0, tol=DOLFIN_EPS):
        self.l0 = l0
        self.tol = tol
        SubDomain.__init__(self)
    def inside(self, x, on_boundary):
        return near(x[0], self.l0, self.tol)

class RightInward(SubDomain):	# right boundary
    def __init__(self, xval, u, l0, tol=DOLFIN_EPS):
        self.xval = xval
        self.u = u
        self.l0 = l0
        self.tol = tol
        self.newX = self.xval + self.u
        self.newBoundary = self.xval[np.where(self.newX <= self.l0)[0][-1]]
        #self.newBoundary = self.l0
        SubDomain.__init__(self)
    def inside(self, x, on_boundary):
        return near(x[0], self.newBoundary, self.tol)


class Gamma(UserExpression):
    def __init__(self,g_1,g_2,new_l0, degree=0):
        print("BIS HIER")
        super().__init__()
        self.g_1, self.g_2 = g_1, g_2
        self.new_l0 = new_l0
    def eval(self, value, x):
        "Set value[0] to value at point x"
        tol = 1E-14
        if x[0] <= self.new_l0 + tol:
            value[0] = self.g_1
        else:
            value[0] = self.g_2

class Etha(UserExpression):
    def __init__(self,eleft,eright,new_l0, degree=0):
        print("BIS HIER")
        super().__init__()
        self.eleft, self.eright = eleft, eright
        self.new_l0 = new_l0
    def eval(self, value, x):
        "Set value[0] to value at point x"
        tol = 1E-14
        if x[0] <= self.new_l0 + tol:
            value[0] = self.eleft
        else:
            value[0] = self.eright


class El_Modulus(UserExpression):
    def __init__(self,mod_left,mod_right,new_l0, degree=0):
        print("BIS HIER")
        super().__init__()
        self.mod_left, self.mod_right = mod_left, mod_right
        self.new_l0 = new_l0
    def eval(self, value, x):
        "Set value[0] to value at point x"
        tol = 1E-14
        if x[0] <= self.new_l0 + tol:
            value[0] = self.mod_left
        else:
            value[0] = self.mod_right

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

# p = {'Ec':10*1e3,'Etac':100e3,'Eta_sf':5000e3,'hc': 1e-6, 'vc': 0.5,'Es':20e3, 'hs': 50e-6, 'vs': 0.5,'Lc':50e-6,'Ya':1e9} # For slides
p = {'Ec':10*1e3,'Etac':100e3,'Eta_sf':10000e3,'hc': 1e-6, 'vc': 0.5,'Es':20e3, 'hs': 50e-6, 'vs': 0.5,'Lc':50e-6,'Ya':1e9}
def sim_kv_bar(p,S,l0=None):
    L0 = Constant(S)
    nel = 1000
    tol = L0/nel/2

    # Create mesh and function space
    mesh = IntervalMesh(nel, 0, L0)
    x = Expression("x[0]", degree=0)
    # x = SpatialCoordinate(mesh)
    xval = x.compute_vertex_values(mesh)
    print(mesh.coordinates())
    # P1 = FiniteElement('CG', interval, 2)
    e = FiniteElement('CG', interval, 1)

    element = MixedElement([e, e])
    V = FunctionSpace(mesh, element)



    # initialize Neumann boundaries
    if l0 == None:
        right = Right(L0, DOLFIN_EPS)
        left = Left(0.0,DOLFIN_EPS)
        boundaries = MeshFunction("size_t",mesh,mesh.topology().dim()-1,0)
        boundaries.set_all(0)
        right.mark(boundaries,1)
        left.mark(boundaries,0)

        ds = Measure('ds', subdomain_data=boundaries)
    if l0 is not None:
        right = Right(L0, DOLFIN_EPS)
        left = Left(0.0, DOLFIN_EPS)
        boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundaries.set_all(0)
        left.mark(boundaries, 0)
        right.mark(boundaries, 1)
        ds = Measure('ds', subdomain_data=boundaries)

        right_active = Interior(l0, tol)
        # left_active = Left(0.0, DOLFIN_EPS)
        boundaries_active = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundaries_active.set_all(0)
        right_active.mark(boundaries_active, 3)
        # left_active.mark(boundaries_active, 2)
        dS = Measure('dS', subdomain_data=boundaries_active)

    # define variational problem
    DT = 60
    dt = Constant(DT)
    uinit  = Constant(0.0)#Expression("- a * sin(x[0] / xmax * pi)", a = 1, xmax = L0, pi =3.14,degree=0)
    # u = TrialFunction(V)
    dU = TrialFunction(V)
    U = Function(V)
    v1, v2 = TestFunctions(V)
    U_n = Function(V)
    U_n1 = Function(V)
    U_save = Function(V)
    uinit = Expression("0", degree=0)

    u_n = project(uinit, V.sub(0).collapse())
    u_n1 = project(uinit, V.sub(0).collapse())

    u_y_n = project(uinit, V.sub(1).collapse())
    u_y_n1 = project(uinit, V.sub(1).collapse())

    # concentration_before = assemble(c_n*dx(mesh))
    assign(U_n, [u_n, u_y_n])
    assign(U_n1, [u_n1, u_y_n1])

    u,u_y = split(U)
    u_n, u_y_n = split(U_n)
    u_n1, u_y_n1 = split(U_n1)




    # uold1= interpolate(uinit, V)
    # uold1.assign(u)
    # uold2 = interpolate(uinit, V)
    # uold2.assign(u)

    # _________________________________________________________________________________________ Set Fixed cell paramters
    E3D = p['Ec']  # Elastic modulus Pa
    eta3D = p['Etac']  # Viscous modulus Pa/s
    eta_sf3D = p['Eta_sf']

    h = p['hc']  # m

    # conversion to 2D constants for plane stress and thin layer approximation
    Eh = E3D * h  # N / m = Pa * m
    etah = eta3D * h  # Ns / m = Pa * m
    etah_sf = eta_sf3D * h  # Ns / m = Pa * m
    nu = 0.5
    lmbdaE = Eh * nu / ((1 - nu) * (1 + nu))  # for 3d: ((1 + nu) * (1 - 2 * nu))
    muE = Eh / (2 * (1 + nu))
    lmbdaEta = etah * nu / ((1 - nu) * (1 + nu))  # for 3d: ((1 + nu) * (1 - 2 * nu))
    muEta = etah / (2 * (1 + nu))
    tauc = etah/Eh

    lp, Ys, Y = penetrationLength_thick_subs(E3D, h, p)  # Unit [m].

    armWidth = 5.0  # um width of the H-bars of the micro pattern
    kN = Y  # Spring stiffness density kN in [N/m/ um**2] -> get u in [um]
    print("Ec: ", p['Ec'] * 1e-3, 'kPa')
    print("hc: ", p['hc'] * 1e6, 'um')
    print("Es: ", p['Es'] * 1e-3, 'kPa')
    print("hs: ", p['hs'] * 1e6, 'um')
    print("Spring constant density substrate:", Ys, "N/(m*um^2)")
    print("Spring constant density total:", Y, "N/(m*um^2)")
    print("Localization length:", lp * 1e6, "um")
    Eh_til = Eh/(1-nu**2)
    etah_til_sf = etah_sf/(1-nu**2)
    etah_til = etah/(1-nu**2)
    tauc=Constant(tauc)
    sigma_b = Constant(5e-3)#N/m
    sigma_a = Constant(0)
    sigma_a_n = Constant(0)
    sigma_0 = 0.0#3.0e-3 #N/m


    sigma_m = Constant(0)
    sigma_m_n = Constant(0)

    act_flag = False
    act_t = 1200
    t_off = 1680
    tau_act = 150
    tau_rel = 400
    center = 1000
    t = 0
    T =3600

    activation_profile_act = []
    activation_profile_rel = []
    def sigma_act_fct(t):
        return Constant(sigma_0*(1-np.exp(-t/tau_act)))

    def sigma_rel_fct(t):
        return Constant(sigma_0*(1-np.exp(-(t_off-act_t)/tau_act))*np.exp(-(t-(t_off-act_t))/tau_rel))#Constant(sigma_0*np.exp(-(t-act_t)/tau_rel))


    cf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    region_left = AutoSubDomain(lambda x, on: x[0] < 22.5)
    region_right = AutoSubDomain(lambda x, on: x[0] >= 22.5)
    region_left.mark(cf, 0)
    region_right.mark(cf, 1)
    dx_sub = Measure('dx', subdomain_data=cf)



    fig = plt.figure()

    # define NonlinearProblem
    class System(NonlinearProblem):
        def __init__(self, a, L):
            NonlinearProblem.__init__(self)
            self.L = L
            self.a = a

        def F(self, b, x):
            assemble(self.L, tensor=b)

        def J(self, A, x):
            assemble(self.a, tensor=A)

    u_array = []
    u_gamma_array = []
    u_y_array = []
    activation_function_array = []
    strain_energy_array = []
    strain_energy_left_array = []
    strain_energy_right_array = []
    activation_boundary_position = []

    newBoundary = l0
    for i in np.arange(0,T,DT):

        newt = t - act_t
        if near(t, act_t,1e-3):
            act_flag = True
            print("True")
        if act_flag:
            print("ACTIVATED################################################################################################")
            if newt == 0:
                sigma_a = Constant(sigma_0*(1 - np.exp(-newt / tau_act)) * (1 - 1 / (1 + np.exp(-(newt - center) / tau_rel))))
                sigma_a_n = sigma_a
            elif newt >=0:
                sigma_a =  Constant(sigma_0*(1 - np.exp(-newt / tau_act)) * (1 - 1 / (1 + np.exp(-(newt - center) / tau_rel))))
                sigma_a_python = sigma_0 * (1 - np.exp(-newt / tau_act)) * (1 - 1 / (1 + np.exp(-(newt - center) / tau_rel)))
                sigma_a_python_n = sigma_0 * (1 - np.exp(-(newt - DT) / tau_act)) * (1 - 1 / (1 + np.exp(-((newt - DT) - center) / tau_rel)))
                if sigma_a_python_n  > sigma_a_python:
                    np.savetxt('start_off_stress_decrease.txt', [newt + act_t])
                if sigma_a_python < 1e-5:
                    np.savetxt('pa_is_zero.txt',[newt+act_t])
                print("Sigma active ", sigma_a_python)
                print("Sigma active n ", sigma_a_python_n)
                sigma_a_n = Constant(sigma_0*(1 - np.exp(-(newt - DT)/ tau_act)) * (1 - 1 / (1 + np.exp(-((newt - DT) - center) / tau_rel))))
            # elif newt > 0 and newt < t_off-act_t:
            #     sigma_a = sigma_act_fct(newt)
            #     sigma_a_n = sigma_act_fct(newt-DT)
            # elif newt >= t_off-act_t:
            #     print("RELAXATION##########################################################################################")
            #     sigma_a = sigma_rel_fct(newt)
            #     sigma_a_n = sigma_rel_fct(newt - DT)


            #sigma_m  = Constant(30.0)#+ Expression("s/(1+exp(1*(x[0]+ux-b)))*(1 - exp(-t))*(1 - 1 / (1 + exp(-(t - t_rel))) ", degree=1, s=10, b=l0, ux=u,t=newT,t_rel=3)
            #sigma_m_n = Constant(30.0)#+ Expression("s/(1+exp(1*(x[0]+ux-b)))*(1 - exp(-t))*(1 - 1 / (1 + exp(-(t - t_rel))) ", degree=1, s=10, b=l0, ux=u,t=(newT-DT),t_rel=3)
            # p = Expression('s', degree=0, domain=mesh, s=30)
            # Expression("1/(1+exp(1*(x[0]-17.5)))/1 ", degree=0)

        gamma_left = 0.3
        gamma_right = 0.5*0.3
        e_left =  etah_sf/(1-nu**2)
        e_right = etah_sf/(1-nu**2)
        eh_til_left = Eh/(1-nu**2)
        eh_til_right = Eh/(1-nu**2)
        print(e_left)
        if newt <=0:
            gamma = Gamma(1e30,1e30,newBoundary)#Constant(0.5)
        else:
            if sigma_a_python<=1e-5:
                print("Sigma is zero")
                gamma = Gamma(gamma_left, gamma_right, newBoundary)  # Constant(0.5)
                etah_til_sf = Etha(e_left, e_right, newBoundary)
                Eh_til = El_Modulus(eh_til_left, eh_til_right, newBoundary)
            else:
                gamma = Gamma(gamma_left, gamma_right, newBoundary)  # Constant(0.5)
                etah_til_sf = Etha(e_left, e_right, newBoundary)
                Eh_til = El_Modulus(eh_til_left, eh_til_right, newBoundary)



        # B0 = -(sigma_b+sigma_a)/etaA*dt - tauc/etaA*(sigma_a-sigma_a_n)
        # B1 = -(sigma_b+sigma_a)/etaA*dt - tauc/etaA*(sigma_a-sigma_a_n)
        # Bi = Constant(0)#-sigma_a/etaA*dt - tauc/etaA*(sigma_a-sigma_a_n)




        u_g = u-u_y
        u_g_n = u_n-u_y_n
        u_g_n1 = u_n1-u_y_n1
        # F_KV1 = -sigma_b*v1*ds(1) + sigma_b*v1*ds(0) - Eh_til*inner(grad(u),grad(v1))*dx - etah_til*inner(grad(u-u_n)/dt,grad(v1))*dx - Y*u_y*v1*dx
        # F_KV2 = -sigma_b*v2*ds(1) + sigma_b*v2*ds(0) - Eh_til*inner(grad(u),grad(v2))*dx - etah_til*inner(grad(u-u_n)/dt,grad(v2))*dx - gamma/dt*(u-u_n-u_y+u_y_n)*v2*dx


        B_MW0 = -((sigma_a+sigma_b)+ etah_til_sf/Eh_til*(sigma_a-sigma_a_n)/dt)
        B_MW1 = sigma_b#+ tauc * (sigma_a - sigma_a_n) / dt)
        B_MW3 = -(sigma_a+ etah_til_sf/Eh_til*(sigma_a-sigma_a_n)/dt)#Constant(0)

        F_1 = B_MW1*v1*ds(1)+B_MW0*v1*ds(0)-avg(B_MW3*v1)*dt*dS(3) +Y*u_y*v1*dx + etah_til_sf/Eh_til*Y*(u_y-u_y_n)/dt*v1*dx+etah_til_sf*inner(grad(u-u_n)/dt,grad(v1))*dx
        F_2 = B_MW1*v2*ds(1)+B_MW0*v2*ds(0)-avg(B_MW3*v2)*dt*dS(3) + gamma*(u_g-u_g_n)/dt*v2*dx+ etah_til_sf/Eh_til*gamma*(u_g+u_g_n1-2*u_g_n)/dt**2*v2*dx + etah_til_sf*inner(grad(u-u_n)/dt,grad(v2))*dx

        # F1 = -B0*v1*ds(0) + B1*v1*ds(1) +avg(Bi*v1)*dS(3) - Y/etaA*u_y*v1*dt*dx - inner(grad(u-u_n),grad(v1))*dx - Y/etaA*(u_y-u_y_n)*v1*dx
        #
        #
        #
        # F2 = -B0*v2*dt*ds(0) + B1*v2*dt*ds(1)+avg(Bi*v2)*dt*dS(3) - inner(grad(u-u_n),grad(v2))*dt*dx - Y/etaA *(u-u_n-u_y+u_y_n)*dt*v2*dx - gamma/etaA*(u-u_y+u_n1-u_y_n1-2*u_n+2*u_y_n)*v2*dx
        #

        F_KV1 = -sigma_b * v1 * ds(1) + sigma_b * v1 * ds(0) - Eh_til * inner(grad(u), grad(v1)) * dx - etah_til * inner(grad(u - u_n) / dt, grad(v1)) * dx - Y * u_y * v1 * dx
        F_KV2 = -sigma_b * v2 * ds(1) + sigma_b * v2 * ds(0) - Eh_til * inner(grad(u), grad(v2)) * dx - etah_til * inner(grad(u - u_n) / dt, grad(v2)) * dx - gamma / dt * (u - u_n - u_y + u_y_n) * v2 * dx
        # F_KV1 = -(sigma_b+sigma_a) * v1 * ds(1) + sigma_b * v1 * ds(0)-avg(sigma_a*v1)*dt*dS(3) - Eh_til * inner(grad(u), grad(v1)) * dx - etah_til * inner(grad(u - u_n) / dt, grad(v1)) * dx - Y * u_y * v1 * dx
        # F_KV2 = -sigma_b * v2 * ds(1) + sigma_b * v2 * ds(0)-avg(sigma_a*v2)*dt*dS(3) - Eh_til * inner(grad(u), grad(v2)) * dx - etah_til * inner(grad(u - u_n) / dt, grad(v2)) * dx - gamma / dt * (u - u_n - u_y + u_y_n) * v2 * dx


        # L = pref*inner(uold,v)*dx + etaA * inner(grad(uold),grad(v))*dx + dt * (sigma_bck+sigma_act) * v * ds(0) - dt * sigma_bck * v * ds(1) - avg(sigma_act*v) * dt * dS(3)
        bcs = []
        if newt <= 0:
            L = F_KV1 + F_KV2
        else:
            L =  F_1+F_2

        a = derivative(L, U, dU)
        problem = System(a, L)
        solver = NewtonSolver()
        solver.parameters["linear_solver"] = "lu"
        solver.parameters["convergence_criterion"] = "incremental"
        solver.parameters["relative_tolerance"] = 1e-6

        solver.solve(problem, U.vector())
        #
        U_save.assign(U)

        _u, _u_y = U_save.split(deepcopy=True)
        strain_energy = assemble(0.5 * Y *45 * u_y**2 * dx(mesh))
        strain_energy_left = assemble(0.5 * Y *45 * u_y**2 * dx_sub(0))
        strain_energy_right = assemble(0.5 * Y * 45 * u_y ** 2 * dx_sub(1))

        print("Integral left ", assemble(u_y**2 * dx_sub(0)))#project(u_y, V.sub(0).collapse())
        print("Integral right ", assemble(u_y**2* dx_sub(1)))
        if strain_energy < 0:
            break
        # if _u_y(0) < 0:
        #     break

        print("u tot links ",_u(0))
        print("u_y links",_u_y(0))
        print("u_gamma links",_u(0)-_u_y(0))

        print("u tot rechts ", _u(S))
        print("u_y rechts", _u_y(S))
        print("u_gamma rechts", _u(S) - _u_y(S))

        right_active = RightInward(xval, _u.compute_vertex_values(), l0,tol)
        # left_active = Left(0.0, DOLFIN_EPS)
        boundaries_active = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundaries_active.set_all(0)
        right_active.mark(boundaries_active, 3)
        # left_active.mark(boundaries_active, 2)
        dS = Measure('dS', subdomain_data=boundaries_active)

        # print(_u.vector().get_local())
        # print(_u_y.vector().get_local())
        # print(len(u.vector().get_local()))

        U_n1.assign(U_n)
        U_n.assign(U)
        newX = xval + _u.vector().get_local()
        newBoundary = xval[np.where(newX <= l0)[0][-1]]
        print("New Boundary = ", newBoundary)
        # plt.axvline(ymin=0,ymax=1,x=newBoundary,lw=0.2)
        if newt>=0 and newt <= (t_off-act_t):
            plt.plot(np.linspace(0, S, nel+1), _u.vector().get_local(), label=r'',color='r',lw=0.2)
            plt.axvline(ymin=0, ymax=1, x=newBoundary, lw=0.2,color='r')
            activation_function_array.append(sigma_a)
        elif newt >= (t_off-act_t):
            plt.plot(np.linspace(0, S, nel+1), _u.vector().get_local(), label=r'',color='b',lw=0.2)
            # plt.axvline(ymin=0, ymax=1, x=newBoundary, lw=0.2, color='b')
        else:
            plt.plot(np.linspace(0, S, nel + 1), _u.vector().get_local(), label=r'', color='k',lw=0.2)
            plt.axvline(ymin=0, ymax=1, x=newBoundary, lw=0.2, color='k')

        u_array.append(_u.vector().get_local())
        u_y_array.append(_u_y.vector().get_local())
        u_gamma_array.append(_u.vector().get_local()-_u_y.vector().get_local())
        strain_energy_array.append(strain_energy)
        strain_energy_left_array.append(strain_energy_left)
        strain_energy_right_array.append(strain_energy_right)
        activation_boundary_position.append(newBoundary)





        t = t + DT

    np.savez('results.npz', u=np.array(u_array),u_y=np.array(u_y_array),u_gamma=np.array(u_gamma_array),Es=np.array(strain_energy_array),Es_l=np.array(strain_energy_left_array),Es_r=np.array(strain_energy_right_array),time=np.arange(0,T,DT),x=np.linspace(0, 45, nel+1),l0=np.array(activation_boundary_position))
    plt.xlim(0, 45)
    # plt.axes().set_aspect('equal')
    # plt.ylim(-6.0, 6.0)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.figure(figsize=(5, 5))
    fig.savefig("u(x).pdf", tight_layout=True)


sim_kv_bar(p,45,22.5-200*0.045)

