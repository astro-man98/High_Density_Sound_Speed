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
class RVSS_EOS_Slope_PT:
    def __init__(self, slope_param, delta_n, nmatch = 1.0*n0,n_points_lin=250,n_points_hd=250,cs2_f=0.95,n_2 = 5.0*n0,cs2_qcd=1.0/3.0,n_max=50*n0):


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
                        if float(line[0]) < nmatch:
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
        
        cs2_compose = np.gradient(press_compose,eden_compose)

        cs2_m = cs2_compose[-1]
        mu_m = chem_pot_compose[-1]
        pr_m = press_compose[-1]

        delta_n_lin = (cs2_f - cs2_m)/(slope_param)
        n_f = nmatch + delta_n_lin


        if n_f > n_2:
            print('Error slope parameter out of range final n larger than n2')
        
        n_mesh_LD = np.linspace(nmatch,n_f,n_points_lin)
        n_mesh_ID = np.linspace(n_f,n_2,n_points_hd)
        n_mesh_PT = np.linspace(n_2,n_2 + delta_n,n_points_hd)
        n_mesh_QCD = np.linspace(n_2 + delta_n,n_max,n_points_hd)

        P_LD = np.empty_like(n_mesh_LD)
        P_ID = np.empty_like(n_mesh_ID)
        P_PT = np.empty_like(n_mesh_PT)
        P_QCD = np.empty_like(n_mesh_QCD)

        eps_LD = np.empty_like(n_mesh_ID)
        eps_ID = np.empty_like(n_mesh_ID)
        eps_PT = np.empty_like(n_mesh_PT)
        eps_QCD = np.empty_like(n_mesh_QCD)

        mu_LD = np.empty_like(n_mesh_LD)
        mu_ID = np.empty_like(n_mesh_ID)
        mu_PT = np.empty_like(n_mesh_PT)
        mu_QCD = np.empty_like(n_mesh_QCD)

        cs2_LD = np.empty_like(n_mesh_LD)
        cs2_ID = np.empty_like(n_mesh_ID)
        cs2_PT = np.empty_like(n_mesh_PT)
        cs2_QCD = np.empty_like(n_mesh_QCD)
        

        P_LD[0] = pr_m
        eps_LD[0] = eden_compose[-1]
        cs2_LD[0] = cs2_m
        mu_LD[0] = mu_m

        for i in range (1,len(n_mesh_LD)):
            cs2_LD[i] = ((n_mesh_LD[i] - n_mesh_LD[i-1])*slope_param) + cs2_LD[i-1]
            eps_LD[i] = eps_LD[i-1] + mu_LD[i-1]*(n_mesh_LD[i] - n_mesh_LD[i-1])
            P_LD[i] = P_LD[i-1] + cs2_LD[i-1]*mu_LD[i-1]*(n_mesh_LD[i] - n_mesh_LD[i-1])
            mu_LD[i] = (P_LD[i] + eps_LD[i])/n_mesh_LD[i]
        
        P_ID[0] = P_LD[-1]
        eps_ID[0] = eps_LD[-1]
        cs2_ID[0] = cs2_LD[-1]
        mu_ID[0] = mu_LD[-1]

        for i in range(1,len(n_mesh_ID)):
            cs2_ID[i] = cs2_ID[i-1]
            eps_ID[i] = eps_ID[i-1] + mu_ID[i-1]*(n_mesh_ID[i] - n_mesh_ID[i-1])
            P_ID[i] = P_ID[i-1] + cs2_ID[i-1]*mu_ID[i-1]*(n_mesh_ID[i] - n_mesh_ID[i-1])
            mu_ID[i] = (P_ID[i] + eps_ID[i])/n_mesh_ID[i]

        P_PT[0] = P_ID[-1]
        cs2_PT[0] = 0.0
        eps_PT[0] = eps_ID[-1]
        mu_PT[0] = mu_ID[-1]

        for i in range (1,len(n_mesh_PT)):
            cs2_PT[i] = 0.0
            eps_PT[i] = eps_PT[i-1] + mu_PT[i-1]*(n_mesh_PT[i] - n_mesh_PT[i-1])
            P_PT[i] = P_PT[i-1] +cs2_PT[i-1]*mu_PT[i-1]*(n_mesh_PT[i] - n_mesh_PT[i-1])
            mu_PT[i] = (P_PT[i] + eps_PT[i])/n_mesh_PT[i]
        
        P_QCD[0] = P_PT[-1]
        eps_QCD[0] = eps_PT[-1]
        cs2_QCD[0] = cs2_qcd
        mu_QCD[0] = (P_QCD[0] + eps_QCD[0])/n_mesh_QCD[0]

        for i in range (1,len(n_mesh_QCD)):
            cs2_QCD[i] = cs2_qcd
            eps_QCD[i] = eps_QCD[i-1] + mu_QCD[i-1]*(n_mesh_QCD[i] - n_mesh_QCD[i-1])
            P_QCD[i] = P_QCD[i-1] + cs2_qcd*mu_QCD[i-1]*(n_mesh_QCD[i] - n_mesh_QCD[i-1])
            mu_QCD[i] = (P_QCD[i] + eps_QCD[i])/n_mesh_QCD[i]

        self.mu_list = [chem_pot_compose,mu_LD[1:],mu_ID[1:],mu_PT[1:],mu_QCD[1:]]
        self.P_list = [press_compose,P_LD[1:],P_ID[1:],P_PT[1:],P_QCD[1:]]
        self.eps_list = [eden_compose,eps_LD[1:],eps_ID[1:],eps_PT[1:],eps_QCD[1:]]
        self.cs2_list = [cs2_compose,cs2_LD[1:],cs2_ID[1:],cs2_PT[1:],cs2_QCD[1:]]
        self.nb_list = [bden_compose,n_mesh_LD[1:],n_mesh_ID[1:],n_mesh_PT[1:],n_mesh_QCD[1:]]

        self.mu_tot = np.concatenate(self.mu_list).flatten()
        self.P_tot = np.concatenate(self.P_list).flatten()
        self.eps_tot = np.concatenate(self.eps_list).flatten()
        self.cs2_tot = np.concatenate(self.cs2_list).flatten()
        self.nb_tot = np.concatenate(self.nb_list).flatten()

        self.eos_interp = interp1d(self.P_tot,self.eps_tot,fill_value='extrapolate')
        self.dens_interp = interp1d(self.P_tot,self.nb_tot,fill_value='extrapolate')

        self.slope_p_min = (cs2_f - cs2_m)/(n_2 - nmatch)

        print(f'matching pressure: {pr_m:.3f}')
        print(f'mu match: {mu_m :.3f}')
        print(f'n_match: {nmatch:.3f}')
        print(f'cs2 match: {cs2_m:.3f}')

        print('Done!')


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



