from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.integrate import quad
import numpy as np
from matplotlib import pyplot as plt
R0 = 1.476


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
    def __init__(self, epsilon, eos, bary_density):
        self.scale_e = epsilon
        self.eos = eos
        self.bary_density = bary_density

        self.epsilon_sol = self.scale_e * 8.9495 * 10 ** (-7)
        self.beta_sol = 4 * np.pi * self.epsilon_sol

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
    
    def ns_vol_avg(self,q_interp, r_stop ,solution=None, nu_sol = None, p_central=1.0, dr = 0.001, rmax=30, dr_max = 0.01):
        if solution is not None:
            sol = solution
            nu = nu_sol
        else:
            sol,nu = self.solve_nu(p_central,dr,rmax)
        M_r = sol.y[1,:]
        rad = sol.t

        M_r_interp = interp1d(rad,M_r,fill_value='extrapolate')
        nu_r_interp = interp1d(rad,nu,fill_value='extrapolate')

        def vol_integrand(r):
            return 4*np.pi*r**(2)*(1 - 2*R0*M_r_interp(r)/r)**(-1/2)*np.exp(nu_r_interp(r))
        def vol_avg_integrand(r):
            return q_interp(r)*vol_integrand(r)
        
        Vol = quad(vol_integrand,a=1e-3,b=r_stop)[0]
        Vol_avg_integral = quad(vol_avg_integrand,a=1e-3,b=r_stop)[0]
        return Vol_avg_integral/Vol
    
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
        radii = sol.t
        R = radii[-1]
        M = sol.y[1][-1]

        def integrand(r):
            m = sol.sol(r)[1]
            p = sol.sol(r)[0]
            return (m/r**2)*((1 + (self.beta_sol*r**3)*(p/m))/(1 - (2*R0/r)*m))

        nu_surf = 0.5*np.log(1 - (2*R0/R)*M)

        nu_central = nu_surf + (1/(6*e_c))*(1 + 3*(p_c/e_c))*np.log(1 - self.beta_sol*(2*R0)*(dr**2)*e_c) - \
                     R0*quad(integrand,dr,R)[0]

        nu = np.empty_like(radii)

        for i,r in enumerate(radii):
            if r == 0:
                nu[i] = nu_central
            else:
                nu[i] = nu_surf - R0*quad(integrand,r,R)[0]

        return sol, nu





