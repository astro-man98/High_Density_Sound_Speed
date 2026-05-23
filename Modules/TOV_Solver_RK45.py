from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.integrate import quad,cumulative_trapezoid
import numpy as np
from matplotlib import pyplot as plt
import Modules.unitconv as units
from scipy.interpolate import PchipInterpolator,interp1d

R0 = 1.476
density_conversion = 10**54
MeVFm3_SI = units.MEV_SI/(units.FM_SI**3)
M_sol_geom = 1477.7


class TovSolverRK45:
    """
    Class that creates a solver for the TOV equations for a single fluid model

    Attributes
    -----------
    epsilon : float
        scale factor in MeV/fm^3 used to solve the dimensionless version of the TOV equations
    eos :
        Interpolating polynomial for the total equation of state e(p)
    bary_density:
        Interpolating polynomial for the ordinary baryon density nb(p)
    """
    def __init__(self, epsilon, eos, bary_density, dpde):
        self.scale_e = epsilon
        self.eos = eos
        self.bary_density = bary_density

        self.epsilon_sol = self.scale_e * 8.9495 * 10 ** (-7)
        self.beta_sol = 4 * np.pi * self.epsilon_sol
        self.dpdrho = dpde

    def solve(self,p_central, dr, rmax, dr_max):
        p1 = p_central
        e1 = self.eos(p_central)
        m1 = (1.0 / 3.0) * self.beta_sol * (dr ** 3) * self.eos(p_central)
        a = (2.0 * R0 / 3.0) * self.beta_sol * (self.eos(p_central))
        n1_bary = 4 * np.pi * self.bary_density(p_central) * ((1 / (2 * a ** (3.0 / 2.0))) *
                                               (np.arcsin(np.sqrt(a) * dr) - np.sqrt(a) * dr * np.sqrt(
                                                   1 - a * dr ** 2))) * 10**54

        p1 = p_central - (((R0 * m1 * self.eos(p1)) / (dr ** 2.0)) * (1 + p1 / e1) * (
                    1 + (self.beta_sol * (dr ** 3) * p1 / m1)) * (1 - (2 * R0 * m1) / dr) ** (-1.0))*dr

        y0 = [p1, m1, n1_bary]

        def TOV(r, y):
            p = y[0]
            M = y[1]
            N_bary = y[2]

            if p < 0:
                return [0,0,0]
            e = self.eos(p)

            dmdr = self.beta_sol * r ** 2 * e
            dpdr = -(((R0 * M * self.eos(p)) / (r ** 2.0)) * (1 + p / e) * (
                    1 + (self.beta_sol * (r ** 3) * p / M)) * (1 - (2 * R0 * M) / r) ** (-1.0))
            dndr_bary = ((4 * np.pi * r ** 2 * self.bary_density(p)) / (np.sqrt(1 - (2 * R0 / r) * M))) * 10**(54)

            return [dpdr, dmdr, dndr_bary]

        def star_boundary(r, y):
            return y[0]

        star_boundary.terminal = True

        solution = solve_ivp(TOV, (dr, rmax), y0,
                             method='RK45', events=star_boundary,
                             dense_output=True,
                             max_step=dr_max
                             )
        return solution
    def LoveTwoFluid(self,r,y,solution):
        geom_units = units.geom_ulength(1)


        p = solution(r/1000)[0]*self.scale_e*MeVFm3_SI/geom_units.pressure
        m = solution(r/1000)[1]*M_sol_geom

        rho = self.eos(p*geom_units.pressure/(self.scale_e*MeVFm3_SI))*self.scale_e*MeVFm3_SI/geom_units.edens


        dpdrho = self.dpdrho((p*geom_units.pressure)/(self.scale_e*MeVFm3_SI))

        if dpdrho == 0:
            k = 0
        else: k = 1

        p_g = p*MeVFm3_SI/geom_units.pressure

        dydr = -1*((1 - (2*m)/r)*y**2 + y*(1 + (4*np.pi*r**2)*(rho- p_g)))*(1/(r - 2*m)) - \
                4*np.pi*r*(1/(1 - (2*m)/r))*(5*rho + 9*p_g + k*(p_g + rho)/dpdrho) + \
                6*(1/(r - 2*m)) + r*(2*(1/(1 - (2*m)/r))*(m + 4*np.pi*p_g*(r**3))/(r**2))**2

        return dydr

    def solve_tidal(self,p1_central,dr_initial=0.001,r_max = 30):
        background_sol = self.solve(p1_central,dr_initial,r_max,dr_max = 0.01)
        back_sol = background_sol.sol
        R = background_sol.t[-1]*1000
        M = background_sol.y[1][-1]
        beta = M*M_sol_geom/R

        y_initial = self.LoveTwoFluid(dr_initial*1000,2,back_sol)*dr_initial*1000 + 2

        solution = solve_ivp(self.LoveTwoFluid,(dr_initial*1000,background_sol.t[-1]*1000),[y_initial],method='RK45',
                               dense_output=True, args=(back_sol,),max_step=10)
        y_R = solution.y[0][-1]

        k_2 = (8.0/5.0)*(beta**5)*((1 - 2*beta)**2)*(2 - y_R + 2*beta*(y_R - 1))* \
              ((2*beta*(6 - 3*y_R + 3*beta*(5*y_R - 8))) + 4*(beta**3)*(13 - 11*y_R + beta*(3*y_R - 2) + 2*(beta**2)*(1 + y_R)) + \
              3*((1 - 2*beta)**2)*(2 - y_R + 2*beta*(y_R - 1))*np.log(1 - 2*beta))**(-1.0)
        lambda_tid = ((2.0/3.0)*k_2*(R**5)/((M*M_sol_geom)**5))
        return lambda_tid,k_2
    

    def ns_vol_avg(self,q_grid, r_grid, M_grid ,solution=None, nu_sol = None, p_central=1.0, dr = 0.001, rmax=30, dr_max = 0.01):

        M_r = M_grid*R0

        vol_integrand_r = 4*np.pi*r_grid**(2)*(1 - 2*M_r/r_grid)**(-1/2)*np.exp(nu_sol)
        vol_avg_integrand_r = q_grid*vol_integrand_r
        
        Vol = cumulative_trapezoid(vol_integrand_r,r_grid)
        Vol_avg_integral = cumulative_trapezoid(vol_avg_integrand_r,r_grid)
        return Vol,Vol_avg_integral/Vol
    


    def ns_mass_grav_avg(self,q_interp,p_central,r_stop,dr=0.001,rmax=30,dr_max=0.01):
        sol,nu = self.solve_nu(p_central,dr,rmax,dr_max=0.01)
        P_r = sol.y[0,:]
        rad = sol.t

        P_r_interp = interp1d(rad,P_r)

        def mass_integrand(r):
            return 4*np.pi*(r**2)*P_r_interp(r)*np.exp(nu)
        
        def mass_avg_integrand(r):
            return q_interp(r)*mass_integrand(r)
        
        Mg = quad(mass_integrand,a=1e-4,b=r_stop)[0]
        Mass_avg_integral = quad(mass_avg_integrand,a=1e-4,b=r_stop)[0]

        return Mass_avg_integral/Mg

    def solve_nu(self,p_c,dr=0.01,r_max=30):
        sol = self.solve(p_c,dr,r_max,dr_max=0.01)
        e_c = self.eos(p_c)
        rad = sol.t

        R = rad[-1]
        M = sol.y[1][-1]*R0

        nu_surf = 0.5*np.log(1 - (2*M)/R)
        nu = np.empty_like(rad)

        m_r = sol.y[1,:]*R0
        p_r = sol.y[0,:]
        eps_r = self.eos(p_r)
        
        dpdr_r = np.gradient(p_r,rad)
        
        #plt.plot(sol.t,dpdr_r)

        integrand_r = -1*(1.0/(eps_r + p_r))*dpdr_r

        integral = cumulative_trapezoid(integrand_r,rad)
        idx_stop = np.argmax(integral) - 1

        C = nu_surf - integral[idx_stop]
        print(nu_surf,integral[idx_stop] + C)
        return sol,integral[:idx_stop] + C,np.argmax(integral)






