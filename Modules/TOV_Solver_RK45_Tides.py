from scipy.integrate import solve_ivp
import numpy as np
from scipy.integrate import quad
import Modules.unitconv as units

from matplotlib import  pyplot as plt


R0 = 2.948
density_conversion = 10**54
MeVFm3_SI = units.MEV_SI/(units.FM_SI**3)
M_sol_geom = 1477.7

class TovSolverSingleFluidRK45:
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
    def __init__(self, epsilon, eos, bary_density):
        self.scale_e = epsilon
        self.eos = eos
        self.bary_density = bary_density

        self.epsilon_sol = self.scale_e * 8.9495 * 10 ** (-7)
        self.beta_sol = 4 * np.pi * self.epsilon_sol

    def dPdr(self, r, p, m):
        e = self.eos(p)
        return -((R0/2)*(e + p)*(m + self.beta_sol*(r**3)*p)) * \
                (r*(r - R0*m))**(-1)

    def dmdr(self, r, p):
        e = self.eos(p)
        return self.beta_sol*(r**2)*e

    def dndr(self,r,m,n):
        return (4 * np.pi * n * density_conversion * r**2)/(np.sqrt(1 - (R0/r)*m))

    def TOV(self,r, y):
        p = y[0]
        M = y[1]
        N_bary = y[2]

        if p < 0:
            return [0, 0, 0]
        e = self.eos(p)

        dmdr = self.dmdr(r,p)
        dpdr = self.dPdr(r,p,M)
        dndr = self.dndr(r,M,N_bary)

        return [dpdr, dmdr, dndr]

    def star_boundary(self,r, y):
        return y[0]

    star_boundary.terminal = True

    def solve(self,p_central, dr, rmax):
        p1 = p_central
        e1 = self.eos(p_central)
        m1 = (1.0 / 3.0) * self.beta_sol * (dr ** 3) * self.eos(p_central)
        a = (2.0 * R0 / 3.0) * self.beta_sol * (self.eos(p_central))

        m_initial = ((self.beta_sol / 3) * dr ** 3) * (self.eos(p_central))
        p_initial = p_central - self.dPdr(dr, p_central, m_initial) * dr
        n_initial = \
            4 * np.pi * quad(
                lambda r: ((r ** 2) * self.bary_density(p_central) / (np.sqrt(1 - (r ** 2) * R0 * self.beta_sol * e1))),
                0,dr,limit=100)[0]

        y0 = [float(p1), float(m_initial), float(n_initial)]

        solution = solve_ivp(self.TOV, (dr, rmax), y0,
                             method='RK45', events=self.star_boundary,
                             dense_output=True,
                             )
        return solution

class TovSolverTwoFluidRK45:
    """
    Class that creates a solver for the TOV equations for a two fluid system where the only interaction between the
    fluids is assumed to be gravitational

    Attributes
    -----------
    epsilon : float
        scale factor in MeV/fm^3 used to solve the dimensionless version of the TOV equations
    fluid1_eos :
        Interpolating polynomial for the equation of state describing the first fluid e_1(p_1)
    fluid2_eos:
        Interpolating polynomial for the equation of state describing the second fluid e_2(p_2)
    fluid1_density:
        Interpolating polynomial for the # density of the first fluid n1(p1)
    fluid2_density:
        interpolating polynomial for # density of the second fluid  n2(p2)
    """
    def __init__(self, epsilon, eos, dpdrho, density):
        self.scale_e = epsilon
        self.eos = eos

        self.dpdhro = dpdrho

        self.den = density

        self.epsilon_sol = self.scale_e * 8.9495 * 10 ** (-7)
        self.beta_sol = 4 * np.pi * self.epsilon_sol

    def dPdr(self, r, p, m):
        e1 = self.eos(p)
        return -((R0/2)*(e1 + p)*(m + self.beta_sol*(r**3)*p)) * \
                (r*(r - R0*m))**(-1)

    def dmdr(self, r, p1):
        e = self.eos(p1)
        return self.beta_sol*(r**2)*e

    def dndr(self,r,m,n):
        return (4 * np.pi * n * density_conversion * r**2)/(np.sqrt(1 - (R0/r)*m))

    def star_boundary(self, r, y):
        return y[0]

    star_boundary.terminal = True

    def TOV_two_fluid(self, r, y):
        p = y[0]
        if p <= 0: return [0,0,0]
        m = y[1]

        dpdr = self.dPdr(r, p, m)
        dmdr = self.dmdr(r, p)
        dNdr = self.dndr(r, m, self.den(p))

        return [dpdr, dmdr, dNdr]

    def LoveTwoFluid(self,r,y,solution):
        geom_units = units.geom_ulength(1)


        p = solution(r/1000)[0]*self.scale_e*MeVFm3_SI/geom_units.pressure
        m = solution(r/1000)[1]*M_sol_geom

        rho = self.eos(p*geom_units.pressure/(self.scale_e*MeVFm3_SI))*self.scale_e*MeVFm3_SI/geom_units.edens


        dpdrho = self.dpdhro((p*geom_units.pressure)/(self.scale_e*MeVFm3_SI))

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


    def solve(self,p1_central, dr_initial=0.001, r_max=30, dr_max=np.inf, dr_min=0):

        e = self.eos(p1_central)

        m_initial = ((self.beta_sol/3) * dr_initial ** 3) * e

        p1_initial = p1_central + self.dPdr(dr_initial, p1_central, m_initial) * dr_initial

        n1_initial = self.dndr(dr_initial, m_initial, self.den(p1_initial)) * dr_initial


        y0 = [float(p1_initial), float(m_initial), float(n1_initial)]

        solution = solve_ivp(self.TOV_two_fluid,
                             (dr_initial, r_max), y0, method='RK45', events=self.star_boundary, dense_output=True,
                             max_step=dr_max)
        return solution


