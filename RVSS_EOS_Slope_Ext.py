import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

fm3tocm3 = 10 ** 39
MeVfm3_dynecm2 = 1.60218 * 10 ** 33
MeVfm3_gcm3 = 1.78266 * 10 ** 12

n0 = 0.16
MN = 939.56
class RVSS_EOS_Slope_Ext:
    def __init__(self, mu_i, cs2_i = 0.95, mu_max = 1500, nmatch = 1.0):



        with open('Data/tables/eos.nb', 'r') as compose_bden_file:
            with open('Data/tables/eos.thermo', 'r') as compose_thermo_file:
                bden_lines = compose_bden_file.readlines()
                thermo_lines = compose_thermo_file.readlines()
                stop_index = 0

                bden_compose = []
                press_compose = []
                eden_compose = []
                chem_pot_compose = []

                self.m_n = 939

                for i,line in enumerate(bden_lines):
                    if i > 1:
                        line = line.replace('\n','')
                        line = line.replace('\t',' ')
                        line = line.split(' ')
                        line = [x for x in line if x]
                        if float(line[0]) < nmatch*n0:
                            bden_compose.append(float(line[0]))
                        else:
                            stop_index = i
                            break


                for i,line in enumerate(thermo_lines[0:stop_index-1]):
                    line = line.replace('\n', '')
                    line = line.replace('\t', ' ')
                    line = line.split(' ')
                    line = [x for x in line if x]

                    if i == 0:
                        pass
                    elif i == stop_index:
                        break
                    else:
                        press_compose.append(float(line[3]) * bden_compose[i - 1])
                        eden_compose.append((float(line[9]) + 1) * self.m_n * bden_compose[i - 1])
                        chem_pot_compose.append((float(line[5]) + 1) * self.m_n)
                press_compose = np.array(press_compose)
                eden_compose = np.array(eden_compose)
                bden_compose = np.array(bden_compose)
                chem_pot_compose = np.array(chem_pot_compose)

        cs2_m = (press_compose[-1] - press_compose[-2]) / (eden_compose[-1] - eden_compose[-2])
        mu_m = chem_pot_compose[-1]
        pr_m = press_compose[-1]

        alpha = (cs2_i - cs2_m)/(mu_i - mu_m)

        print(f'matching pressure: {pr_m:.3f}')
        print(f'mu match: {chem_pot_compose[-1]:.3f}')
        print(f'cs2 match: {cs2_m:.3f}')

        if mu_i < chem_pot_compose[-1]:
            print(f'mu_i should be higher than matching chemical potential'
                  f' {chem_pot_compose[-1]:.2f}')

        n_mesh = 100
        mu_mesh_regI = np.linspace(mu_m,mu_i,n_mesh)
        mu_mesh_regII = np.linspace(mu_i*1.01,mu_max,n_mesh)

        beta = 1/(cs2_m + alpha*mu_m)

        cs2_I = np.empty(n_mesh)
        cs2_II = np.empty(n_mesh)

        nb_hd = np.empty(2*n_mesh)

        P_hd = np.empty(2*n_mesh)

        eps_hd = np.empty(2*n_mesh)
        
        
        sp = 0.01
        mu_2 = mu_i + (1.0/(3.0*sp))

        for i in range(len(mu_mesh_regI)):
            mu = mu_mesh_regI[i]
            cs2_I[i] = alpha*(mu - mu_m) + cs2_m
        
        cs2_II[0] = 0.001
        for i in range(1,len(mu_mesh_regII)):
            mu = mu_mesh_regII[i]
            mu_m1 = mu_mesh_regII[i-1]
            if(mu < mu_2):
                cs2_II[i] = cs2_II[i-1] + sp*(mu - mu_m1)
            else:
                cs2_II[i] = 1.0/3.0
        
        mu_tot = np.concatenate([mu_mesh_regI,mu_mesh_regII])
        cs2_tot = np.concatenate([cs2_I,cs2_II])

        cs2_interp = interp1d(mu_tot,cs2_tot)
        for i,mu in enumerate(mu_tot):
            integral_nb = quad(lambda mu_p: 1/(mu_p*cs2_interp(mu_p)),mu_m,mu)
            nb_hd[i] = n0*np.exp(integral_nb[0])

        nb_interp = interp1d(mu_tot,nb_hd)

        for i,mu in enumerate(mu_tot):
            integral_pr = press_compose[-1] + quad(lambda mu_p: nb_interp(mu_p),mu_m,mu,limit=200)
            P_hd[i] = integral_pr[0]
            eps_hd[i] = mu*nb_hd[i] - P_hd[i]

        self.P_unified = np.concatenate([press_compose,P_hd[1:]])
        self.E_unified = np.concatenate([eden_compose,eps_hd[1:]])
        self.nb_unified = np.concatenate([bden_compose,nb_hd[1:]])
        self.mu_unified = np.concatenate([chem_pot_compose,mu_tot[1:]])
        self.eos_interp = PchipInterpolator(self.P_unified,self.E_unified)
        self.cs2_unified = 1/self.eos_interp.derivative()(self.P_unified)
        self.cs2_interp = interp1d(self.P_unified,self.cs2_unified,fill_value='extrapolate')
        self.cs2_nb = PchipInterpolator(self.nb_unified,self.cs2_unified)

        self.nb_interp = PchipInterpolator(self.P_unified,self.nb_unified)
        self.eden_nb = PchipInterpolator(self.nb_unified,self.E_unified)
        self.press_nb = PchipInterpolator(self.nb_unified,self.P_unified)
        
                
        self.alpha = alpha
        self.chem_pot_unified = (self.P_unified + self.E_unified)/(self.nb_unified)
        print('Done')
        print(f'alpha = {alpha}')


    # def export_tables_BNS(self,density_min,density_max,savepath,name,numpoints=1000):
    #     uc = ut.SI_UNITS/ut.PIZZA_UNITS
    #     densities = np.logspace(np.log10(density_min),np.log10(density_max),numpoints) # 1/fm^3

    #     MeVfm3_Pa = 1.6022 * 10 ** 32
    #     mev_fm3_kgm3_SI = ((ut.MEV_SI / (ut.C_SI ** 2)) / ut.FM_SI ** 3)

    #     rmd_SI = ((self.m_n*densities) * mev_fm3_kgm3_SI)
    #     rmd_PIZZA = rmd_SI*uc.density
    #     sed_PIZZA = (self.eden_nb(densities)/(self.m_n*densities) - 1)
    #     pres_PIZZA = self.press_nb(densities) * MeVfm3_Pa * uc.pressure
    #     cs_PIZZA = np.sqrt(self.cs2_nb(densities))*ut.C_SI*uc.velocity


    #     eos = EOS_Table(rmd_PIZZA,sed_PIZZA,pres_PIZZA,cs_PIZZA,mbar=self.m_n,isentropic=True)
    #     print('Making EOS adiabatic')
    #     eos = eos.resample_geom(numpoints)
    #     eos = eos.make_adiabatic()
    #     print('Attach Polytrope')
    #     eos = eos.make_poly_compatible(3)
    #     eos = eos.make_restmass_natural(3)
    #     eos = eos.attach_poly(rmd_PIZZA[0] / 100, 100)
    #     eos = eos.make_adiabatic()
    #     print('Done')


    #     eos.save_pizza(f'{savepath}/{name}.pizza')
    #     eos.save_lorene(f'{savepath}/{name}.lorene')




