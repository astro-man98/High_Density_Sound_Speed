import numpy as np
from cmath import isnan

R0 = 1.47 #(Half of Solar Schwarzchild Radius km)
PCUTOFF = 10**(-12)
hbar_c = 197.326



class tovSolverSingleReg:

    def __init__(self, epsilon, EOS, Density):
        # Initialize class variables
        self.epsilon = epsilon
        self.EOS = EOS
        self.Density = Density

        # Convert from nuclear units to astrophysical units (MeV/fm^3) -> (Msol c^2/km^3), 1/fm -> 1/km
        self.epsilon_sol = self.epsilon * 8.9495 * 10 ** (-7)
        self.beta_sol = 4 * np.pi * self.epsilon_sol

    # dP and dM given from TOV equations
    def diffP(self, p, mass, rad, EOS):
        return -((R0 * mass * EOS(p)) / (rad ** 2.0) * (1 + p / (EOS(p))) * (
                    1 + (self.beta_sol * rad ** 3 * p / mass)) * (1 - (2 * R0 * mass) / rad) ** (-1.0))

    def diffM(self, p, mass, rad, eDen):
        return self.beta_sol * (rad ** 2) * eDen

    def diffN(self, p, mass, rad, BaryonNum):
        if p < 0 or mass < 0: return 0
        else: return (4 * np.pi * rad ** 2 * BaryonNum(p)) / (np.sqrt(1 - (2 * R0 / rad) * mass))

    def TOV_Solver(self, pCent, dr1, raccel, dr2):
        '''
        Function that numerically solves the two fluid TOV equations and returns the properties of the Neutron Star
        pCent: Central unitless pressure
        dr1: initial distance step size
        raccel: radius at which to switch from step size dr1 to dr2
        dr2: optional second distance step size (set to dr1 for one step size)

        returns: radius, p(r), m(r), e(r), N(r)
        @param pCent,dr1,raccel,dr2
        '''
        EOS = self.EOS
        Density = self.Density
        dr = dr1
        i = 1

        p = [pCent]
        e = [EOS(pCent)]
        m = [0,(1.0/3.0)*self.beta_sol*(dr**3)*EOS(pCent)]
        r = [0]
        N = [0]
        a = (2.0 * R0 / 3.0) * self.beta_sol * (EOS(pCent))

        N.append(4 * np.pi * Density(pCent) * ((1 / (2 * (a) ** (3.0 / 2.0))) *
         (np.arcsin(np.sqrt(a) * dr) - np.sqrt(a) * dr * np.sqrt(1 - a * dr ** 2))))


        p.append(pCent + self.diffP(pCent,m[1],dr,EOS)*dr)
        e.append(EOS(p[1]))
        r.append(r[0] + dr)
        # Begin Solver Loop
        while complex(p[i]).real > PCUTOFF and complex(p[i]).imag == 0:
            if (r[i] >= raccel): dr = dr2
            # print(pBaryon[i],'      ',pDarkBaryon[i])

            # First RK Step (Baryonic Matter)

            p1 = p[i]
            m1 = m[i]

            kP1 = self.diffP(p1, m[i], r[i], EOS)
            km1 = self.diffM(p1, m[i], r[i], EOS(p1))
            kN1 = self.diffN(p1, m[i], r[i], Density)

            p2 = p[i] + kP1 * dr / 2
            m2 = m[i] + km1 * dr/2

            kP2 = self.diffP(p2, m2, r[i] + dr / 2, EOS)
            km2 = self.diffM(p2, m2, r[i] + dr / 2, EOS(p2))
            kN2 = self.diffN(p2, m2, r[i] + dr / 2, Density)

            p3 = p[i] + kP2 * dr / 2
            m3 = m[i] + km2*dr/2

            kP3 = self.diffP(p3, m3, r[i] + dr / 2, EOS)
            km3 = self.diffM(p3, m3, r[i] + dr / 2, EOS(p3))
            kN3 = self.diffN(p3, m3, r[i] + dr / 2, Density)

            p4 = p[i] + kP3 * dr
            m4 = m[i] + km3 * dr

            kP4 = self.diffP(p4, m4, r[i] + dr, EOS)
            km4 = self.diffM(p4, m4, r[i] + dr, EOS(p4))
            kN4 = self.diffN(p4, m4, r[i] + dr, Density)

            if(p[i] + ((kP1 + 2 * kP2 + 2 * kP3 + kP4) / 6) * dr < 0):
                break

            p.append(p[i] + ((kP1 + 2 * kP2 + 2 * kP3 + kP4) / 6) * dr)
            e.append(EOS(p[i + 1]))
            m.append(m[i] + ((km1 + 2*km2 + 2*km3 + km4)/6)*dr)
            N.append(N[i] + ((kN1 + 2 * kN2 + 2 * kN3 + kN4)/6) * dr)
            r.append(r[i] + dr)
            i += 1
        # Strips any complex values from pTot and mTot
        for j in range(len(p) - 1, -1, -1):
            if isinstance(p[j], complex):
                p.pop(j)
                e.pop(j)
                m.pop(j)
                N.pop(j)
                r.pop(j)

            if (p[j] < 0):
                p.pop(j)
                e.pop(j)
                m.pop(j)
                N.pop(j)
                r.pop(j)

        for j in range(len(m) - 1, -1, -1):
            if isnan(m[j]) or isinstance(m[j], complex) or m[j] < 0:
                p.pop(j)
                e.pop(j)
                m.pop(j)
                N.pop(j)
                r.pop(j)

        return r, p, m, e, N


class tovSolverTwoReg:
    def __init__(self,epsilon,NMEOS,DMEOS,NMDensity,DMDensity):
        #Initialize class variables
        self.epsilon = epsilon
        self.NMEOS = NMEOS
        self.DMEOS = DMEOS
        self.NMDensity = NMDensity
        self.DMDensity = DMDensity

        #Convert from nuclear units to astrophysical units (MeV/fm^3) -> (Msol c^2/km^3), 1/fm -> 1/km
        self.epsilon_sol = self.epsilon*8.9495*10**(-7)
        self.beta_sol = 4*np.pi*self.epsilon_sol
    
    #dP and dM given from TOV equations
    def diffP(self,p,pTot,mass,rad,EOS):
        return -(((R0*mass*EOS(p)))/((rad**2.0))*(1 + (p)/(EOS(p)))*(1 + (self.beta_sol*rad**(3)*(pTot)/(mass)))*(1 - (2*R0*mass)/(rad))**(-1.0))
    def diffM(self,p,mass,rad,eDen):
        return self.beta_sol*(rad**2)*eDen
    def diffN(self,p,mass,rad,BaryonNum):
        return (4*np.pi*rad**(2)*BaryonNum(p))/(np.sqrt(1 - (2*R0/rad)*mass))
    def diffN2(self,p,mass,rad,BaryonNum):
        try:
            diff = (4*np.pi*rad**(2)*BaryonNum)/(np.sqrt(1 - (2*R0/rad)*mass))
            return diff
        except RuntimeWarning:
            print('Runtime warning ')

    
 
    def TOV_Solver(self,pbCent,pdmCent,dr1,raccel,dr2):
        '''
        Function that numerically solves the two fluid TOV equations and returns the properties of the Neutron Star
        pbCent: Central unitless pressure of baryonic matter
        pdmCent: Central unitless pressure of dark matter
        dr1: initial distance step size
        raccel: radius at which to switch from step size dr1 to dr2
        dr2: optional second distance step size (set to dr1 for one step size)

        returns: radius,total mass, baryonic pressure/energy density profile, dark matter pressure/energy density profile,
        @param self,pbCent,pdmCent,dr1,raccel,dr2
        @returns r,mTot,pBaryon,pDarkBaryon,eDenBaryon,eDenDarkBaryon,nBaryon,nDarkBaryon
        '''
        nmEOS = self.NMEOS
        dmEOS = self.DMEOS
        nmDensity = self.NMDensity
        dmDensity = self.DMDensity
        dr = dr1
        i = 1

        baryonicMatter = False
        darkMatter = False

        if(pbCent != 0): baryonicMatter = True
        if(pdmCent != 0): darkMatter = True
        
        if(baryonicMatter):
            pBaryon = [pbCent]
            eDenBaryon = [nmEOS(pbCent)]
            mBaryon = [0,(1.0/3.0)*(self.beta_sol)*(dr**3)*nmEOS(pbCent)]
            nBaryon = [0]
        else:
            pBaryon = [0]
            eDenBaryon = [0]
            mBaryon = [0,0]
            nBaryon = [0]

        if darkMatter:
            pDarkBaryon = [pdmCent]
            eDenDarkBaryon = [dmEOS(pdmCent)]
            mDarkBaryon = [0,(1.0/3.0)*(self.beta_sol)*(dr**3)*dmEOS(pdmCent)]
            nDarkBaryon = [0]
        else:
            pDarkBaryon = [0]
            eDenDarkBaryon = [0]
            mDarkBaryon = [0,(1.0/3.0)*(self.beta_sol)*(dr**3)*dmEOS(pdmCent)]
            nDarkBaryon = [0]

        
        mTot = [0,mBaryon[1] + mDarkBaryon[1]]
        r = [0]


        if baryonicMatter:
            p1Baryon = pbCent + self.diffP(pbCent,pbCent + pdmCent,mTot[1],dr,nmEOS)*dr
            e1Baryon = nmEOS(p1Baryon)
        else:
            p1Baryon = 0
            e1Baryon = 0

        if(darkMatter):
            p1DarkBaryon = pdmCent + self.diffP(pdmCent,pbCent + pdmCent,mTot[1],dr,dmEOS)*dr
            e1DarkBaryon = dmEOS(p1DarkBaryon)
        else:
            p1DarkBaryon = 0
            e1DarkBaryon = 0

        a = (2.0*R0/(3.0))*self.beta_sol*(eDenBaryon[0] + eDenDarkBaryon[0])
        if(a < 0):
            print('Warning invalid value')
            print(a)
        if(baryonicMatter):nBaryon.append(4*np.pi*nmDensity(pbCent)*((1/(2*(a)**(3.0/2.0)))*(np.arcsin(np.sqrt(a)*dr) - np.sqrt(a)*dr*np.sqrt(1 - a*dr**(2)))))
        else: nBaryon.append(0)

        if(darkMatter): nDarkBaryon.append(4*np.pi*(dmDensity(pdmCent))*pDarkBaryon[0]*(1/(2*(a)**(3.0/2.0)))*(np.arcsin(np.sqrt(a)*dr) - np.sqrt(a)*dr*np.sqrt(1 - a*dr**(2))))
        else: nDarkBaryon.append(0)

        pBaryon.append(p1Baryon)
        eDenBaryon.append(e1Baryon)

        pDarkBaryon.append(p1DarkBaryon)
        eDenDarkBaryon.append(e1DarkBaryon)

        r.append(dr)

        pTot = [pBaryon[0] + pDarkBaryon[0],pBaryon[1] + pDarkBaryon[1]]
        eDenTot =  [eDenBaryon[0] + eDenDarkBaryon[0],eDenBaryon[1] + eDenDarkBaryon[1]]
        

        # Begin Solver Loop
        while complex(pBaryon[i] + pDarkBaryon[i]).real > 0 and complex(pBaryon[i] + pDarkBaryon[i]).imag == 0 and (darkMatter or baryonicMatter):
            if(r[i] >= raccel):dr = dr2
            if(r[i] >= 20): break
            #print(pBaryon[i],'      ',pDarkBaryon[i])

            #First RK Step (Baryonic Matter)
            if baryonicMatter: p1B = pBaryon[i]
            else: p1B = 0
            if p1B < 0: 
                baryonicMatter = False
                p1B = 0
            

            if darkMatter: p1D = pDarkBaryon[i]
            else: p1D = 0
            if p1D < 0: 
                p1D = 0
                darkMatter = False
            
            
            if baryonicMatter:
                kP1B = self.diffP(p1B,p1B + p1D,mTot[i],r[i],nmEOS)
                km1B = self.diffM(p1B,mTot[i],r[i],nmEOS(p1B))
                kN1B = self.diffN(p1B,mTot[i],r[i],nmDensity)
            else:
                kP1B = 0
                km1B = 0
                kN1B = 0

            #First Step (Dark Matter)
            if(darkMatter):
                kP1D = self.diffP(p1D,p1B + p1D,mTot[i],r[i],dmEOS)
                km1D = self.diffM(p1D,mTot[i],r[i],dmEOS(p1D))
                kN1D = self.diffN2(p1D,mTot[i],r[i],dmDensity(p1D))
            else:
                kP1D = 0
                km1D = 0
                kN1D = 0
            
            #second RK (Normal Matter)
            if(baryonicMatter):p2B = pBaryon[i] + kP1B*dr/2
            else: p2B = 0
            if p2B < 0: 
                p2B = 0
                baryonicMatter = False
            

            if darkMatter: p2D = pDarkBaryon[i] + kP1D*dr/2
            else: p2D = 0
            if p2D < 0: 
                p2D = 0
                darkMatter = False
            
        
            if baryonicMatter:
                kP2B = self.diffP(p2B,p2B + p2D,mTot[i] + (km1B + km1D)*dr/2,r[i] + dr/2,nmEOS)
                km2B = self.diffM(p2B,mTot[i] + (km1B + km1D)*dr/2,r[i]+ dr/2,nmEOS(p2B))
                kN2B = self.diffN(p2B,mTot[i] + (km1B + km1D)*dr/2,r[i]+ dr/2,nmDensity)
            else:
                kP2B = 0
                km2B = 0
                kN2B = 0

            #second RK (Dark Matter)
            if(darkMatter):
                kP2D = self.diffP(p2D,p2D + p2B,mTot[i] + (km1B + km1D)*dr/2,r[i] + dr/2,dmEOS)
                km2D = self.diffM(p2D,mTot[i] + (km1B + km1D)*dr/2,r[i] + dr/2,dmEOS(p2D))
                kN2D = self.diffN2(p2D,mTot[i] + (km1B + km1D)*dr/2,r[i] + dr/2,dmDensity(p2D))
            else:
                kP2D = 0
                km2D = 0
                kN2D = 0

            #Third RK (Normal Matter)
            if baryonicMatter:p3B = pBaryon[i] + kP2B*dr/2
            else: p3B = 0
            if p3B < 0 : 
                p3B = 0
                baryonicMatter = False
            

            if darkMatter: p3D = pDarkBaryon[i] + kP2D*dr/2
            else: p3D = 0
            if p3D < 0: 
                p3D = 0
                darkMatter = False
            
           
            if baryonicMatter:
                kP3B = self.diffP(p3B,p3B + p3D,mTot[i] + (km2B + km2D)*dr/2,r[i] + dr/2,nmEOS)
                km3B = self.diffM(p3B,mTot[i] + (km2B + km2D)*dr/2,r[i] + dr/2,nmEOS(p3B))
                kN3B = self.diffN(p3B,mTot[i] + (km2B + km2D)*dr/2,r[i] + dr/2,nmDensity)
            else:
                kP3B = 0
                km3B = 0
                kN3B = 0

            #Third RK (Dark Matter)

            if(darkMatter):
                kP3D = self.diffP(p3D,p3D + p3B,mTot[i] + (km2B + km2D)*dr/2,r[i],dmEOS)
                km3D = self.diffM(p3D,mTot[i] + (km2B + km2D)*dr/2,r[i],dmEOS(p3D))
                kN3D = self.diffN2(p3D,mTot[i] + (km2B + km2D)*dr/2,r[i],dmDensity(p3D))
            else:
                kP3D = 0
                km3D = 0
                kN3D = 0

            #Fourth RK step for baryonic matter
            if baryonicMatter: p4B = pBaryon[i] + kP3B*dr
            else: p4B = 0
            if p4B < 0: 
                p4B = 0
                baryonicMatter = False

            if darkMatter: p4D = pDarkBaryon[i] + kP3D*dr
            else: p4D = 0
            if p4D < 0:
                 p4D = 0
                 darkMatter = False

            if baryonicMatter:
                kP4B = self.diffP(p4B,p4B + p4D,mTot[i] + (km3B + km3D)*dr,r[i] + dr,nmEOS)
                km4B = self.diffM(p4B,mTot[i] + (km3B + km3D)*dr,r[i]+dr,nmEOS(p4B))
                kN4B = self.diffN(p4B,mTot[i] + (km3B + km3D)*dr,r[i]+dr,nmDensity)
            else:
                kP4B = 0
                km4B = 0
                kN4B = 0

            if(darkMatter):
                kP4D = self.diffP(p4D,p4D + p4B,mTot[i] + (km3B + km3D)*dr,r[i],dmEOS)
                km4D = self.diffM(p4D,mTot[i] + (km3B + km3D)*dr,r[i],dmEOS(p4D))
                kN4D = self.diffN2(p4D,mTot[i] + (km3B + km3D)*dr,r[i],dmDensity(p4D))
            else:
                kP4D = 0
                km4D = 0
                kN4D = 0

            pBaryon.append(pBaryon[i] + ((kP1B + 2*kP2B + 2*kP3B + kP4B)/(6))*dr)
            pDarkBaryon.append(pDarkBaryon[i] + ((kP1D + 2*kP2D + 2*kP3D + kP4D)/(6))*dr)

            if(pBaryon[i+1] < 0 or complex(pBaryon[i+1].imag != 0) or baryonicMatter == False):
                pBaryon[i+1] = 0
                baryonicMatter = False
                nBaryon.append(nBaryon[i])
                mBaryon.append(mBaryon[i])
                eDenBaryon.append(0)
            else:
                nBaryon.append(nBaryon[i] + ((kN1B + 2*kN2B + 2*kN3B + kN4B)/(6))*dr)
                mBaryon.append(mBaryon[i] + ((km1B+ 2*km2B + 2*km3B + km4B)/(6))*dr)
                eDenBaryon.append(nmEOS(pBaryon[i+1]))
            
            if (pDarkBaryon[i+1] < 0 or complex(pDarkBaryon[i+1]).imag != 0 or darkMatter == False):
                pDarkBaryon[i+1] = 0
                eDenDarkBaryon.append(0)
                darkMatter = False
                nDarkBaryon.append(nDarkBaryon[i])
                mDarkBaryon.append(mDarkBaryon[i])
            else:
                nDarkBaryon.append(nDarkBaryon[i] + ((kN1D + 2*kN2D + 2*kN3D + kN4D)/(6))*dr)
                mDarkBaryon.append(mDarkBaryon[i] + ((km1D+ 2*km2D + 2*km3D + km4D)/(6))*dr)
                eDenDarkBaryon.append(dmEOS(pDarkBaryon[i+1]))

            mTot.append(mBaryon[i+1] + mDarkBaryon[i+1])
            eDenTot.append(eDenBaryon[i+1] + eDenDarkBaryon[i+1])
            pTot.append(pBaryon[i+1] + pDarkBaryon[i+1])
        
            r.append(r[i] + dr)
            i += 1
        #Strips any complex values from pTot and mTot
        for j in range(len(pTot)-1,-1,-1):
            if isinstance(pTot[j],complex):
                pDarkBaryon.pop(j)
                pBaryon.pop(j)
                mDarkBaryon.pop(j)
                mBaryon.pop(j)
                nDarkBaryon.pop(j)
                nBaryon.pop(j)
                eDenBaryon.pop(j)
                eDenDarkBaryon.pop(j)

                mTot.pop(j) 
                pTot.pop(j)
                r.pop(j)

            if(pTot[j] < 0):
                pDarkBaryon.pop(j)
                pBaryon.pop(j)
                mDarkBaryon.pop(j)
                mBaryon.pop(j)
                nDarkBaryon.pop(j)
                nBaryon.pop(j)
                eDenBaryon.pop(j)
                eDenDarkBaryon.pop(j)
                mTot.pop(j)
                eDenTot.pop(j)
                pTot.pop(j)
                r.pop(j)

        for j in range(len(mTot)-1,-1,-1):
            if isnan(mTot[j]) or isinstance(mTot[j],complex) or mTot[j] < 0:
                pDarkBaryon.pop(j)
                pBaryon.pop(j)
                mDarkBaryon.pop(j)
                mBaryon.pop(j)
                nDarkBaryon.pop(j)
                nBaryon.pop(j)
                eDenBaryon.pop(j)
                eDenDarkBaryon.pop(j)
                mTot.pop(j)
                eDenTot.pop(j)
                pTot.pop(j)
                r.pop(j)

        return r,mTot,pBaryon,pDarkBaryon,eDenBaryon,eDenDarkBaryon,nBaryon,nDarkBaryon
