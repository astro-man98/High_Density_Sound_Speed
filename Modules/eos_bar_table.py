"""
This module provides a class EOS_Table to manage and modify tabulated
equations of state. There are methods to save to and functions to load from
different formats, including RNS, LORENE, and PIZZA. Further, this defines a
new hdf5 based file format which allows to store all information.
The EOSs consist at least of rest mass density, pressure, and specific energy,
and optionally can have soundspeed, electron fraction, and temperature.
The EOS have metadata, namely name, textual description, whether the EOS is
isentropic, and the formal mass unit to convert between mass density and
number density.
For isentropic EOSs, soundspeed can be recomputed. There are methods to
make EOS exactly isentropic, cut density ranges, resample the table, join
several EOSs, and sample analytic EOSs. Further, by making the
assumption of polytropic behaviour below the lowest tabulated density,
a matching polytrope can be found, the EOS can be made compatible with a
given polytrope, and a natural definition of rest mass density can be found
which equals total energy density at zero density.
The units used internally are by convention geometric units (G=c=1) with
length unit 1m, i.e. Pizza code units.
"""

import sys
import os
import tables as h5
from Modules.unitconv import *
from math import *
from numpy import *
import scipy as spy
import scipy.interpolate as sip
import scipy.integrate as sint
from Modules.eos_bar_analytic import *

COLWIDTH = 10

def cubic_spline(x, y, order=3):
  return sip.InterpolatedUnivariateSpline(x,y,k=order)
#

def cubic_spline_loglog(x, y, order=3):
  s = cubic_spline(log(x),log(y),order)
  return lambda x: exp(s(log(x)))
#

def is_strictly_increasing(x):
  d = x[1:] - x[:-1]
  return all(d>0)
#

def geomspace(x0,x1,n):
  return exp(linspace(log(x0), log(x1), n))
#

class EOS_Table(object):
  """Class representing a tabulated EOS

  This holds tables for the following quantities:
    rmd:        Rest mass density
    sed:        Specific energy, excluding rest mass
    p:          Pressure

  Derived quantities:
    bnd = rmd / mbar   Baryon number density
    ted = rmd*(1+sed)  Total energy density
    hm1 = sed + p/rmd
    h   = hm1 + 1      Specific enthalpy
    gm1 = g0*exp{int_P0^P dP' 1/(ted(P')+P')} - 1
          this is useful to compute hydrostatic equilibrium

  Optionally,
    csnd:       Soundspeed (NOT squared)
    efr:        Electron fraction
    temp:       Temperature

  General infomation on the EOS
    isentropic: If EOS is isentropic.
    mbar:       Conversion factor to compute baryon number density bnd = rmd/mbar
                If set to None, unified atomic mass unit will be used.

    name:       Short name for EOS
    comment:    Description of EOS

  By convention, you are forbidden to modify any member directly, to ensure internal
  consistency. The philosophy is to create new EOS objects to change something.
  Accordingly, the methods return new EOS objects.

  EOS name and comment can always be modified, since they cannot
  cause any inconsistency.

  All quantities are also available as funtion of rmd, e.g. p_from_rmd.
  The functions are computed by cubic interpolation from the table.
  WARNING: If you modify tables, the functions remain unchanged.

  The units are generally geometric units in which G=c=1 and length unit = 1m
  This implies soundspeed is in units of lightspeed, and that energy and mass
  density are equivalent. The temperature is given in MeV.
  """
  units = PIZZA_UNITS
  def _resist(self, msg):
    raise RuntimeError("ERROR".ljust(COLWIDTH)+msg)
  #
  def _protest(self, msg):
      print(("WARNING".ljust(COLWIDTH) + msg))
  #
  def _info(self, msg):
    print(("INFO".ljust(COLWIDTH) + msg))
  #
  def _sed_adiabatic(self, rmd, p, sed0):
    """Compute specific energy from pressure and rest mass density, assuming constant entropy.

    Since tabulated values do not (and may not) start at 0, the specific energy sed0
    at the lowest tabulated density must be given.
    """
    ig    = p / (rmd**2)
    sed   = sint.cumulative_trapezoid(ig, rmd)
    sed   = hstack([[0], sed])
    sed  += sed0
    return sed
  #
  def _csnd_adiabatic(self, rmd, sed, p):
    """Compute soundspeed by numerical differentiation, assuming adiabatic EOS"""
    ed        = (1.0 + sed) * rmd
    p_from_ed = cubic_spline(ed, p, self.order)
    cs2       = p_from_ed(ed, nu=1)
    if (any(cs2 < 0)):
      self._resist('Soundspeed computation failed: d(pressure)/d(energy density) < 0')
    #
    return sqrt(cs2)
  #
  def _compute_gm1(self, ed, P, gm1_0):
    a     = 1.0/(ed + P)
    z     = sint.cumulative_trapezoid(a, P)
    z     = hstack([[0],z])
    gm1   = gm1_0 + (1.0+gm1_0)*expm1(z)
    return gm1
  #
  def _sanity_checks(self):
    """Any EOS has to satisfy some guarantees which are enforced here"""
    res = len(self.rmd)
    if ((res != len(self.sed)) or (res != len(self.p))):
      self._resist('Different table length for rmd,sed,p')
    if (any(self.rmd <= 0)):
      self._resist('Rest mass density not strictly positive')
    if (not is_strictly_increasing(self.rmd)):
      self._resist('Rest mass density not strictly increasing')
    if (any(self.p <= 0)):
      self._resist('Detected pressure <= 0')
    if (self.has_csnd):
      if (res != len(self.csnd)):
        self._resist('Different table length for rmd, csnd')
      if (any(self.csnd < 0.0)):
        self._protest('Detected sound speed < 0')
      if (any(self.csnd > 1.0)):
        self._protest('Detected sound speed > c')
    #
    if (self.has_efr):
      if (res != len(self.efr)):
        self._resist('Different table length for rmd, efr')
      if (any(self.efr > 1)):
        self._resist('Detected electron fraction > 1')
      if (any(self.efr < 0)):
        self._resist('Detected electron fraction < 0')
      #
    #
    if (self.has_temp):
      if (any(self.temp < 0)):
        self._protest('Detected temperature < 0')
      if (len(self.temp) != res):
        self._resist('Different table length for rmd, temp')
      #
    #
  #
  def _oddity_checks(self):
    """This warns in case of odd, but allowed behavior"""
    if (any(self.sed <=0)):
      self._protest('Detected negative specific energy')
    if (not is_strictly_increasing(self.p)):
      self._protest("Pressure not strictly increasing")
    if (any(self.hm1 <=0)):
      self._protest('Detected enthalpy < 1')
    if (not is_strictly_increasing(self.sed)):
      self._protest('Specific energy not strictly increasing')
    if (not is_strictly_increasing(self.hm1)):
      self._protest('Specific enthalpy not strictly increasing')
    if (not is_strictly_increasing(self.gm1)):
      self._protest('Hydrostatic potential not strictly increasing')
  #
  def resample_geom(self, npoints, rmd_min=None, rmd_max=None):
    """Returns EOS resampled at geometrically spaced restmass densities

    If lower and/or upper range is not specified, table limits are used
    """
    d0      = (float(rmd_min) if (rmd_min is not None) else self.rmd_min)
    d1      = (float(rmd_max) if (rmd_max is not None) else self.rmd_max)
    rmd_reg = geomspace(d0, d1, npoints)
    return self.resample(rmd_reg)
  #
  def resample(self, rmd_new):
    """Returns EOS resampled at given densities"""
    return sample_eos(self, rmd_new, self.mbar)
  #
  def compute_soundspeed(self):
    """Recompute soundspeed assuming adiabatic EOS.

    Returns: same EOS with soundspeed attached.
    """
    if (not self.isentropic):
      self._resist("Need isentropic EOS to recompute soundspeed")
    #
    csnd_new  = self._csnd_adiabatic(self.rmd, self.sed, self.p)
    return EOS_Table(self.rmd, self.sed, self.p, isentropic=True,
                      csnd=csnd_new, efr=self.efr, temp=self.temp,
                      mbar=self.mbar, order=self.order)
  #
  def make_adiabatic(self, remove_unphys_csnd=False):
    """Recompute specific energy and soundspeed assuming adiabaticity.

    The specific energy at the lowest tabulated value is assumed to be
    correct. The soundspeed is recomputed. If it exceeds the speed of
    light and remove_unphys_csnd is True, the EOS is restricted to the
    range where it is still causal, else an exception is raised.

    Returns: Isentropic EOS with adjusted specific energy.
    """
    sed_new   = self._sed_adiabatic(self.rmd, self.p, self.sed[0])
    csnd_new  = self._csnd_adiabatic(self.rmd, sed_new, self.p)

    sl        = slice(None, None)
    if remove_unphys_csnd:
      wrong   = csnd_new>1.0
      if any(wrong):
        print("Warning: removing parts with superluminal soundspeed")
        sl    = slice(0,where(wrong)[0][0])
      #
    #
    def cut(a):
      if a is None: return a
      return a[sl]
    #

    return EOS_Table(cut(self.rmd), cut(sed_new), cut(self.p),
                     isentropic=True, mbar=self.mbar,
                     csnd=cut(csnd_new), efr=cut(self.efr),
                     temp=cut(self.temp), order=self.order)
  #
  def natural_restmass_def(self, n_poly):
    """Find natural rest mass density definition

    Rest mass density is just some factor times baryon number density.
    The most natural normalization is given by demanding that
    in the zero density limit, total energy density = rest mass density
    Since the table does not extend down to zero, one has to make an
    assumtion on the analytic form of the EOS below the lowest tabulated
    density. The lower this density is, the smaller the influence of this
    assumption on the result. Here, we assume a polytropic EOS with
    index n_poly at low densities.

    Returns: scaling factor rmd_new_def / rmd_old_def
    """

    rmd_min       = self.rmd[0]
    tmd_min       = rmd_min * (1.0 + self.sed[0])
    p_min         = self.p[0]
    mcf           = (tmd_min - n_poly*p_min) / rmd_min
    return mcf
  #
  def change_restmass_def(self, fconv):
    """Change definition of restmass, i.e. mbar factor.

    Specific energy is adjusted such that total energy density is not changed,
    specific enthalpy is adjusted accordingly. Other quantities are not changed.
    """

    rmd_new       = self.rmd * fconv
    mbar_new      = self.mbar * fconv
    sed_new       = (1.0 / fconv - 1.0) + self.sed / fconv

    self._info("Correcting rest mass density")
    print((" "*COLWIDTH + "rmd_new / rmd_old = %.14e" % fconv))

    return EOS_Table(rmd_new, sed_new, self.p, isentropic=self.isentropic,
                      csnd=self.csnd, efr=self.efr, temp=self.temp,
                      mbar=mbar_new, order=self.order)
  #
  def make_restmass_natural(self, n_poly):
    """Returns EOS with natural restmass density definition.

    Assumes polytrope with index n_poly at low densities, see natural_restmass_def().
    """
    return self.change_restmass_def(self.natural_restmass_def(n_poly))
  #
  def change_ndens_def(self, mbar_new):
    """Returns EOS with number density computed using particle mass mbar_new"""
    return EOS_Table(self.rmd, self.sed, self.p, isentropic=self.isentropic,
                      csnd=self.csnd, efr=self.efr, temp=self.temp,
                      mbar=mbar_new, order=self.order)
  #
  def make_poly_compatible(self, poly_n):
    """Adjust energy density to match polytrope at lowest density.

    First, a polytrope with given index poly_n is constructed such that
    the pressure at the lowest tabulated density matches.
    Then a constant is added to the specific energy such that it
    matches the polytrope at the lowest tabulated density.
    For isentropic EOSs with soundspeed, the latter is recomputed,
    otherwise it is discarded"""
    #
    rmd_p   = self.rmd[0] * ((self.rmd[0]/self.p[0])**poly_n)
    sed0    = poly_n * ((self.rmd[0] / rmd_p)**(1.0/poly_n))
    dsed    = sed0 - self.sed[0]
    self._info("Changing energy to match polytrope\n"
               "  poly_n = %.14e rmd_p = %.14e kg/m^3 \n"
               "  sed_new - sed_old    = %.14e \n"
               "  sed_new/sed_old -1   = %.14e \n"
               % (poly_n, rmd_p * self.units.density, dsed, dsed / self.sed[0]))

    sed_new = self.sed + dsed

    csnd_new = None
    if (self.has_csnd):
      if (self.isentropic):
        csnd_new  = self._csnd_adiabatic(self.rmd, sed_new, self.p)
        d = abs(csnd_new/self.csnd - 1).max()
        self._info("Recomputing soundspeed\n"
                   "  max(|cs_new/cs_old - 1|) = %.14e" % d)
      else:
        self._info("Cannot recompute soundspeed, setting to unknown")
      #
    #
    return EOS_Table(self.rmd, sed_new, self.p, isentropic=self.isentropic,
                      csnd=csnd_new, efr=self.efr, temp=self.temp,
                      mbar=self.mbar, order=self.order)
  #
  def matching_polytrope(self):
    """Find matching polytrope at lowest tabulated density.

    Polytropic constant and index are chosen such that pressure and total energy density
    match the lowest tabulated density.
    If the electron fraction is known, the polytrope is augmented with a constant electron
    fraction of the lowest density table entry.
    If a temperature is known, the polytrope is augmented with one set to zero."""

    n_p       = self.rmd[0] * self.sed[0] / self.p[0]
    rmd_p     = self.rmd[0] * ((n_p / self.sed[0])**n_p)
    poly      = EOS_Poly(rmd_p, n_p)
    if (self.has_temp):
      poly.temp_from_rmd = vectorize(lambda x: 0)
    #
    if (self.has_efr):
      poly.efr_from_rmd  = vectorize(lambda x: self.efr[0])
    #
    return poly
  #
  def attach_poly(self, rmd_min, npoints):
    """Extend tabulated values to lower densities by using a matching polytrope."""

    poly      = self.matching_polytrope()
    rmd_new   = hstack((geomspace(rmd_min, self.rmd_min, npoints)[:-1],
                       self.rmd))

    return join_eos([poly,self],[self.rmd[0]], rmd_new, mbar=self.mbar,
            order=self.order)
  #
  def remove_unphys_points(self):
    """Remove points for which p and/or sed are not increasing"""
    def copy(a):
      if a is None:
        return None
      else:
        return a.copy()
    def restrict(a, mask):
      if a is None:
        return None
      else:
        return a[mask]

    rmd_new  = copy(self.rmd)
    sed_new  = copy(self.sed)
    p_new    = copy(self.p)
    csnd_new = copy(self.csnd)
    efr_new  = copy(self.efr)
    temp_new = copy(self.temp)

    while True:
      mask = ones_like(p_new, dtype=bool)
      mask[1:] = mask[1:] & (diff(sed_new) > 0)
      mask[1:] = mask[1:] & (diff(p_new) > 0)
      if all(mask):
        break
      rmd_new  = restrict(rmd_new,  mask)
      sed_new  = restrict(sed_new,  mask)
      p_new    = restrict(p_new,    mask)
      csnd_new = restrict(csnd_new, mask)
      efr_new  = restrict(efr_new,  mask)
      temp_new = restrict(temp_new, mask)

    return EOS_Table(rmd_new, sed_new, p_new, isentropic=self.isentropic,
        mbar=self.mbar, csnd=csnd_new, efr=efr_new, temp=temp_new,
        order=self.order)
  #
  @property
  def has_temp(self):
    return (self.temp is not None)
  #
  @property
  def has_efr(self):
    return (self.efr is not None)
  #
  @property
  def has_csnd(self):
    return (self.csnd is not None)
  #
  @property
  def rmd_min(self):
    return (self.rmd[0])
  #
  @property
  def rmd_max(self):
    return (self.rmd[-1])
  #
  def __init__(self, rmd, sed, p, csnd=None, efr=None, temp=None, name='noname',
    comment='', isentropic=False, mbar=None, order=3):
    """Construct EOS from tabulated values

    rmd:        Rest mass density
    sed:        Specific energy, excluding rest mass
    p:          Pressure
    isentropic: Set to True if EOS is isentropic.
    mbar:       Conversion factor to compute baryon number density bnd = rmd/mbar
                If set to None, unified atomic mass unit will be used.

    Optional:

    csnd:       Soundspeed (NOT squared)
    efr:        Electron fraction
    temp:       Temperature
    name:       Short name for EOS
    comment:    Description of EOS
    order:      Spline order (default 3)

    The "hydrostatic potential" gm1 is always computed.
    It is defined up to a constant, we set gm1=hm1 at the lower end of
    the table. Assuming isentropic behaviour at lower densities, gm1=0
    at zero density (if sed=0).
    """

    self.name           = str(name)
    self.comment        = str(comment)
    self.isentropic     = bool(isentropic)
    self.rmd            = array(rmd)
    self.sed            = array(sed)
    self.p              = array(p)
    if (mbar is None):
      self._protest('Formal baryon mass not specified, using atomic mass unit')
      mbar = UAMU_SI / self.units.mass
    #
    self.mbar           = float(mbar)
    self.efr            = (array(efr) if (efr is not None) else None)
    self.temp           = (array(temp) if (temp is not None) else None)
    self.csnd           = (array(csnd) if (csnd is not None) else None)

    self.ted            = (1.0 + self.sed) * self.rmd
    self.hm1            = self.sed + self.p / self.rmd
    self.h              = 1.0 + self.hm1
    self.bnd            = self.rmd / self.mbar
    self.gm1            = self._compute_gm1(self.ted, self.p, self.hm1[0])
    self.order          = order

    self._sanity_checks()
    self._oddity_checks()

    self.p_from_rmd     = cubic_spline_loglog(self.rmd, self.p, order)
    self.sed_from_rmd   = cubic_spline(self.rmd, self.sed, order)
    self.hm1_from_rmd   = cubic_spline(self.rmd, self.hm1, order)
    if is_strictly_increasing(self.hm1):
      self.rmd_from_hm1 = cubic_spline(self.hm1, self.rmd, order)
    else:
      self.rmd_from_hm1 = None

    if (self.efr is not None):
      self.efr_from_rmd   = cubic_spline(self.rmd, self.efr, order)
    #
    if (self.temp is not None):
      self.temp_from_rmd  = cubic_spline(self.rmd, self.temp, order)
    #
    if (self.csnd is not None):
      self.csnd_from_rmd  = cubic_spline(self.rmd, self.csnd, order)
    #
  #
  def save_pizza(self, path):
    """Save EOS in tovstar/Pizza format."""
    if (self.csnd is None):
      self._resist('Need soundspeed to save in pizza format')
    #
    u = self.units
    with open(path, 'w') as f:
      f.write("name = %s\n" % self.name)
      f.write("type = tabulated\n")
      f.write("isentropic = %d\n" % self.isentropic)
      f.write("has_temp   = %d\n" % self.has_temp)
      f.write("has_efrac  = %d\n" % self.has_efr)

      fmt = "%.15e  %.15e  %.15e  %.15e  %.15e"
      for i in arange(0,len(self.rmd)):
        f.write(fmt % (self.rmd[i]*u.density, self.sed[i],
                       self.p[i]*u.pressure, self.csnd[i] ** 2, self.gm1[i]))
        if (self.has_temp):
          f.write("  %.15e" % self.temp[i])
        #
        if (self.has_efr):
          f.write("  %.15e" % self.efr[i])
        #
        f.write("\n")
      #
    #
  #
  def save_filga(self, path):
    """Save EOS in Fillipos' format"""
    if (not self.has_efr):
      self._resist('Need electron fraction to save in Fillipos format')
    #
    fm_CGS        = 1e-13 #femtometer
    uc            = self.units / CGS_UNITS
    bnd_FU        = self.bnd / uc.volume * (fm_CGS**3)
    ed_FU         = self.ted * uc.density
    p_FU          = self.p * uc.pressure
    with open(path, 'w') as f:
      for bnd,ed,p,efr in zip(bnd_FU, ed_FU, p_FU, self.efr):
        f.write("%.15e  %.15e  %.15e  %.15e \n" % (bnd, ed, p, efr))
      #
    #
  #
  def save(self, path):
    """Save EOS in hdf5 file. """
    u       = self.units
    h5file  = h5.open_file(path, mode = "w", title = "Tabulated barotropic EOS")
    group   = h5file.create_group(h5file.root, 'eos_table', 'EOS Tables')
    a_rmd   = h5file.create_array(group, 'rmd', self.rmd * u.density,
                  "Rest mass density [kg m^-3]")
    a_sed   = h5file.create_array(group, 'sed', self.sed,
                                  "Specific energy [dimensionless]")
    a_press = h5file.create_array(group, 'press', self.p * u.pressure,
                                  "Pressure [Pa]")
    if (self.has_csnd):
      a_csnd   = h5file.create_array(group, 'csnd', self.csnd * u.velocity,
                                  "Soundspeed [m/s]")
    #
    if (self.has_efr):
      a_efr    = h5file.create_array(group, 'efr', self.efr, "Electron fraction")
    #
    if (self.has_temp):
      a_temp   = h5file.create_array(group, 'temp', self.temp, "Temperature [MeV]")
    #
    h5file.root._v_attrs.eos_name   = self.name
    h5file.root._v_attrs.comment    = self.comment
    h5file.root._v_attrs.isentropic = self.isentropic
    h5file.root._v_attrs.mbar       = self.mbar

    h5file.close()
  #
  def save_lorene(self, path):
    """Save EOS in the LORENE format"""
    uc = PIZZA_UNITS / CGS_UNITS
    uc_SI = PIZZA_UNITS/SI_UNITS

    mbar_SI       = (self.mbar*MEV_SI)/(C_SI**2)
#939.0 * MEV_SI / (C_SI**2)
    fm_SI         = 1e-15
    rmd_SI        = self.rmd * uc_SI.density
    bnd_LU        = (rmd_SI/mbar_SI)*fm_SI**3
    tmd_LU        = self.ted * uc.density
    p_LU          = self.p * uc.pressure
    with open(path, 'w') as f:
      f.write("#\n#\n#\n#\n#\n%d\n#\n#\n#\n" % len(self.rmd))
      for i,bnd,tmd,p in zip(list(range(len(self.rmd))), bnd_LU, tmd_LU, p_LU):
        f.write("%d  %.15e  %.15e  %.15e \n" % (i, bnd, tmd, p))
      #
    #
  #
  def save_rns(self, path):
    """Save EOS in the RNS format.
    """
    mbar_SI       = self.mbar * self.units.mass
    #1.66e-24 * CGS_UNITS.mass
    uc            = self.units / CGS_UNITS
    nd_SI         = self.rmd * self.units.density / mbar_SI
    nd_CGS        = nd_SI * CGS_UNITS.volume
    tmd_CGS       = self.ted * uc.density
    p_CGS         = self.p * uc.pressure
    hodd          = log( (1.0+self.gm1) / (1.0+self.gm1[0])) * ((C_SI/CGS_UNITS.velocity)**2)
    hodd[0]       = 1 # Exact value = 0, but RNS seems to require that.
    p_CGS[0]      = 1

    with open(path, 'w') as f:
      f.write("%d \n" % (len(tmd_CGS)))
      for ed,p,h,n in zip(tmd_CGS, p_CGS, hodd, nd_CGS):
        f.write("%.15e  %.15e  %.15e  %.15e \n" % (ed, p, h, n))
      #
    #
  #
  def save_tovl(self, path):
    """Save EOS in the TOVL format.
    """
    uc = self.units / CGS_UNITS

    mbar_SI       = self.mbar * self.units.mass
    cm_SI         = 1e-2
    rmd_SI        = self.rmd * self.units.density
    bnd_TU        = rmd_SI / mbar_SI * (cm_SI**3)
    tmd_TU        = self.ted * uc.density
    p_TU          = self.p * uc.pressure
    nlines        = len(bnd_TU)
    with open(path, "w") as f:
      f.write("#\n#\n#\n")
      f.write("%d <-- Number of lines\n#\n" % nlines)
      f.write("#          nb [cm-3]       energy dens [g/cm3] pressure [erg/cm2]\n#\n")
      for ix, nb, rho, p in zip(range(nlines), bnd_TU, tmd_TU, p_TU):
        f.write("%d %.16e %.16e %.16e\n" % (ix, nb, rho, p))
      #
  #
  def __str__(self):
    """Textual representation for print etc"""
    fb      = {True:'yes', False:'no'}
    drange  = (self.rmd[0] * self.units.density, self.rmd[-1] * self.units.density)
    s   = "Name                  %s\n" % self.name
    s  += "Comments              %s\n" % self.comment
    s  += "Restmass range        (%.6e, %.6e) kg/m^3\n" % drange
    if (self.has_csnd):
      s  += "Max soundspeed        %.6e c\n" % (self.csnd.max())
    #
    s  += "Resolution            %d points\n" % len(self.rmd)
    s  += "Is isentropic         %s\n" % fb[self.isentropic]
    s  += "Has soundpeed         %s\n" % fb[self.has_csnd]
    s  += "Has temperature       %s\n" % fb[self.has_temp]
    s  += "Has electron fraction %s\n" % fb[self.has_efr]
    mbmev = (self.mbar * self.units.mass * (C_SI**2) / MEV_SI)
    s  += "Baryonic mass unit    %.14e MeV/c^2" % mbmev
    return s
  #
#

def load_eos(path, order=3):
  """Load EOS_Table object from hdf5 file"""
  u = SI_UNITS/PIZZA_UNITS
  with h5.open_file(path, mode = 'r') as h5file:
    name        = h5file.root._v_attrs.eos_name
    comment     = h5file.root._v_attrs.comment
    isentropic  = h5file.root._v_attrs.isentropic
    mbar        = h5file.root._v_attrs.mbar
    gtab        = h5file.root.eos_table
    rmd         = array(gtab.rmd.read()) * u.density
    sed         = array(gtab.sed.read())
    p           = array(gtab.press.read()) * u.pressure
    csnd        = ((array(gtab.csnd.read()) * u.velocity) if ('csnd' in gtab) else None)
    temp        = (array(gtab.temp.read()) if ('temp' in gtab) else None)
    efr         = (array(gtab.efr.read()) if ('efr' in gtab) else None)
  #
  return EOS_Table(rmd, sed, p, isentropic=isentropic,
                      csnd=csnd, efr=efr, temp=temp, name=name, comment=comment, mbar=mbar, order=order)
#


def load_eos_filga(path, umass, name='generic', order=3):
  """Load eos from table in Filippos' format

  Collumn 0: baryon number density [fm^-3]
  Collumn 1: total energy density [c^2 g cm^-3]
  Collumn 2: pressure [dyne/cm^2]
  Collumn 3: electron fraction
  Collumn 4: soundspeed (optional)
  Collumn 5: temperature (optional)

  Converts baryon number density to a formal mass density using a baryon mass umass [MeV].

  Returns EOS_Table
  """
  fm_SI         = 1e-15 #femtometer
  m_b_SI        = umass * MEV_SI /(C_SI**2) #formal baryon mass unit
#931.47222777689672
  pu            = PIZZA_UNITS
  uc            = CGS_UNITS / pu
  mbar          = m_b_SI / pu.mass

  tab           = loadtxt(path, unpack=True)
  rmd_SI        = tab[0] * m_b_SI / (fm_SI**3)
  rmd_GU        = rmd_SI / pu.density
  tmd_GU        = tab[1] * uc.density
  P_GU          = tab[2] * uc.pressure
  efrac         = tab[3]
  sed           = tmd_GU / rmd_GU - 1.0
  csnd          = sqrt(tab[4]) if (len(tab)>=5) else None
  temp          = tab[5] if (len(tab)>=6) else None
  #nd/fm^-3,p/cgs,sed/1,efr,csnd/c,temp/MeV

  return EOS_Table(rmd_GU, sed, P_GU, efr=efrac, csnd=csnd, temp=temp, name=name,
                   comment='Loaded from '+path, mbar=mbar, order=order)
#

def load_eos_lorene(path, umass, name='generic', order=3):
  """Load eos from table in Lorene format

  Collumn 0: baryon number density [fm^-3]
  Collumn 1: total energy density [c^2 g cm^-3]
  Collumn 2: pressure [dyne/cm^2]

  Converts baryon number density to a formal mass density using a baryon mass umass [MeV].

  Returns EOS_Table
  """
  fm_SI         = 1e-15 #femtometer
  m_b_SI        = umass * MEV_SI /(C_SI**2) #formal baryon mass unit
#931.47222777689672
  pu            = PIZZA_UNITS
  uc            = CGS_UNITS / pu
  mbar          = m_b_SI / pu.mass

  tab           = loadtxt(path, unpack=True)
  rmd_SI        = tab[0] * m_b_SI / (fm_SI**3)
  rmd_GU        = rmd_SI / pu.density
  tmd_GU        = tab[1] * uc.density
  P_GU          = tab[2] * uc.pressure
  sed           = tmd_GU / rmd_GU - 1.0


  return EOS_Table(rmd_GU, sed, P_GU, name=name,
                   comment='Loaded from '+path, mbar=mbar, order=order)
#

def load_eos_lorene_std(path, umass, name='generic', order=3):
  """Load eos from table in Lorene format

  Collumn 1: baryon number density [fm^-3]
  Collumn 2: total energy density [c^2 g cm^-3]
  Collumn 3: pressure [dyne/cm^2]

  Converts baryon number density to a formal mass density using a baryon mass umass [MeV].

  Returns EOS_Table
  """
  fm_SI         = 1e-15 #femtometer
  m_b_SI        = umass * MEV_SI /(C_SI**2) #formal baryon mass unit
#931.47222777689672
  pu            = PIZZA_UNITS
  uc            = CGS_UNITS / pu
  mbar          = m_b_SI / pu.mass

  tab           = loadtxt(path, unpack=True, skiprows=9)
  rmd_SI        = tab[1] * m_b_SI / (fm_SI**3)
  rmd_GU        = rmd_SI / pu.density
  tmd_GU        = tab[2] * uc.density
  P_GU          = tab[3] * uc.pressure
  sed           = tmd_GU / rmd_GU - 1.0


  return EOS_Table(rmd_GU, sed, P_GU, name=name,
                   comment='Loaded from '+path, mbar=mbar, order=order)
#

def sample_eos(eos, rmd_new, mbar=None, order=3):
  """Create tabulated EOS by sampling analytic EOS at densities rmd_new

  This only requires an object with functions p_from_rmd, sed_from_rmd,
  and optionally csnd_from_rmd, efr_from_rmd, temp_from_rmd

  The EOS_Table objects provide all the required functions by cubic interpolation,
  and can thus be resampled by this function.
  """
  p_new     = eos.p_from_rmd(rmd_new)
  sed_new   = eos.sed_from_rmd(rmd_new)
  niente    = lambda x : None
  csnd_new  = getattr(eos, 'csnd_from_rmd', niente)(rmd_new)
  efr_new   = getattr(eos, 'efr_from_rmd', niente)(rmd_new)
  temp_new  = getattr(eos, 'temp_from_rmd', niente)(rmd_new)

  return EOS_Table(rmd_new, sed_new, p_new, isentropic=eos.isentropic,
                    csnd=csnd_new, efr=efr_new, temp=temp_new, mbar=mbar, order=order)
#

def join_eos(eoss, rmd_bnd, rmd_new, mbar=None, order=3):
  """Create EOS_Table by joining two or more analytic EOSs

  eoss      List of EOSs to join
  rmd_bnd   Densities at which to change EOS.
  rmd_new   Densities for resulting table.
  mbar      Formal baryonic mass to use in new EOS.
  """

  if (len(eoss) != len(rmd_bnd)+1):
    raise ValueError('Number of EOSs has to be number of boundaries + 1')
  if (len(eoss)<2):
    raise ValueError('You need at least 2 EOSs to join')
  #
  mi =  [rmd_new <= rmd_bnd[0]]
  mi += [logical_and(rmd_new > rl, rmd_new <= rr) for rl,rr in zip(rmd_bnd[:-1],rmd_bnd[1:])]
  mi += [rmd_new > rmd_bnd[-1]]

  rmdi = [rmd_new[m] for m in mi]

  def stitch(mn):
    fi = [getattr(e,mn,None) for e in eoss]
    if any([(f is None) for f in fi]): return None
    return hstack([f(rmd) for f,rmd in zip(fi,rmdi)])
  #

  p_new     = stitch('p_from_rmd')
  sed_new   = stitch('sed_from_rmd')
  if ((p_new is None) or (sed_new is None)):
    raise RuntimeError('Invalid EOS object')
  #

  csnd_new  = stitch('csnd_from_rmd')
  temp_new  = stitch('temp_from_rmd')
  efr_new   = stitch('efr_from_rmd')
  isent_new = all([e.isentropic for e in eoss])

  return EOS_Table(rmd_new, sed_new, p_new, isentropic=isent_new,
                    csnd=csnd_new, efr=efr_new, temp=temp_new, mbar=mbar, order=order)
#

