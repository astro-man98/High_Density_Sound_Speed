"""This module provides classes representing analytic equations of state.
They provide the following functions, which can operate on scalars or numpy
arrays, returning a result of the same type:

p_from_rmd:     compute pressure from rest mass density (rmd)
sed_from_rmd:   computre specific energy (w.o. rest mass energy) from rmd
hm1_from_rmd:   compute h-1 from rmd, where h=1+sed+p/rmd is the specific enthalpy
csnd_from_rmd:  compute soundspeed from rmd

Further they have a flag whether they are isentropic. The units are by convention
geometric units (G=c=1) with length unit 1m, i.e. Pizza code units

Currently implemented are polytropic and piecewise polytropic EOS.
"""

import sys
import os
from Modules.unitconv import *
from math import *
from numpy import *

class EOS_Poly(object):
  """Class representing a Polytropic EOS"""
  isentropic  = True
  units       = PIZZA_UNITS
  def __init__(self, rmd_p, n_p):
    self.ds         = float(rmd_p)
    self.n          = float(n_p)
    self.ga         = 1.0 + 1.0/n_p
  #
  def p_from_rmd(self, rmd):
    """Compute pressure from rest mass density"""
    return self.ds * ((rmd/self.ds)**self.ga)
  #
  def sed_from_rmd(self, rmd):
    """Compute specific energy from rest mass density"""
    return self.n * ((rmd/self.ds)**(1.0/self.n))
  #
  def hm1_from_rmd(self, rmd):
    """Compute h-1 from rest mass density"""
    return self.ga * self.sed_from_rmd(rmd)
  #
  def csnd_from_rmd(self, rmd):
    """Compute soundspeed from rest mass density"""
    hm1 = self.hm1_from_rmd(rmd)
    return sqrt(hm1 / (self.n * (hm1 + 1.0)))
  #
  def __str__(self):
    s =  "Polytropic EOS\n"
    s += "  n_poly     = %.15e \n" % self.n
    s += "  gamma_poly = %.15e \n" % self.ga
    s += "  rmd_poly   = %.15e kg/m^3 \n" % (self.ds * self.units.density)
    return s
  #
#

def generalize_func(f):
  """Convert a function operating on scalars into a function also operating on arrays

  If the resulting function is called with an array, an array is returned,
  if called with a number, a number (not a trivial array like for vectorize).
  """
  v = vectorize(f)
  return lambda x: v(x) if isinstance(x, ndarray) else f(x)
#

class EOS_Piecewise_Poly(object):
  """Class to represent a piecewise polytropic EOS

  The pressure is computed from polytropic EOSs depending on the rest mass density.
  The pressure between the different segments is continuous.
  The specific energy is computed by demanding that the EOS is adiabatic.
  """
  isentropic  = True
  units       = PIZZA_UNITS
  def __init__(self, rmd_poly0, ga_pieces, rmd_bnd):
    """Constructor.

    rmd_poly0: polytropic density scale for the first segment
    ga_pieces: list of the polytropic indices for the different segments
    rmd_bnd:   list of the rest mass densities where to join the segments

    Note the EOS functions accepting arrays as well are added to the instance
    on the fly in the constructor, e.g. p_from_rmd() is derived from _p_from_rmd().
    """
    rmdp_pieces = [rmd_poly0]
    n_pieces    = [1.0/(ga-1.0) for ga in ga_pieces]
    et_bnd      = [nip1/ni for ni,nip1 in zip(n_pieces[:-1],n_pieces[1:])]
    for et,rmdb in zip(et_bnd, rmd_bnd):
      rmd_next = (rmdp_pieces[-1]**et) * (rmdb**(1.0-et))
      rmdp_pieces.append(rmd_next)
    #
    self.poly_pieces    = [EOS_Poly(rmdp, np) for rmdp,np in zip(rmdp_pieces, n_pieces)]
    self.dsed_bnd       = [pl.sed_from_rmd(rmdb)-pr.sed_from_rmd(rmdb) for pl,pr,rmdb
                           in zip(self.poly_pieces[:-1], self.poly_pieces[1:], rmd_bnd)]
    self.rmd_bnd        = rmd_bnd

    #Create general functions from scalar functions
    self.p_from_rmd     = generalize_func(self._p_from_rmd)
    self.sed_from_rmd   = generalize_func(self._sed_from_rmd)
    self.hm1_from_rmd   = generalize_func(self._hm1_from_rmd)
    self.csnd_from_rmd  = generalize_func(self._csnd_from_rmd)
  #
  def which_poly(self, rmd):
    """Return polytrope corresponding to the given rest mass density"""
    for rmdb,p in zip(self.rmd_bnd, self.poly_pieces[:-1]):
      if (rmd<rmdb): return p
    return self.poly_pieces[-1]
  #
  def sed_corr(self, rmd):
    """The specific energy offset with respect to the polytrope segment at rmd"""
    d   = [dsed for rmdb,dsed in zip(self.rmd_bnd, self.dsed_bnd) if rmd>rmdb]
    return sum(d)
  #
  def _p_from_rmd(self, rmd):
    """pressure from rest mass density (as scalar function)"""
    return self.which_poly(rmd).p_from_rmd(rmd)
  #
  def _sed_from_rmd(self, rmd):
    """specific energy rest mass density (as scalar function)"""
    sed = self.which_poly(rmd).sed_from_rmd(rmd)
    return sed + self.sed_corr(rmd)
  #
  def _hm1_from_rmd(self, rmd):
    """h-1 from rest mass density (as scalar function)"""
    hm1 = self.which_poly(rmd).hm1_from_rmd(rmd)
    return hm1 + self.sed_corr(rmd)
  #
  def _csnd_from_rmd(self, rmd):
    """soundspeed from  rest mass density (as scalar function)"""
    csnd  = self.which_poly(rmd).csnd_from_rmd(rmd)
    hp    = 1.0 + self.which_poly(rmd).hm1_from_rmd(rmd)
    cs2   = (csnd**2) / (1.0 + self.sed_corr(rmd)/hp)
    return sqrt(cs2)
  #
  def __str__(self):
    s = "Piecewise polytropic EOS\n"
    sbd = ["%.15e" % (b * self.units.density) for b in self.rmd_bnd]
    sbd = "[%s]" % (', '.join(sbd))
    s+= "  Segment boundaries: %s kg/m^3 \n" % sbd
    s+= "  Segment  n_poly                gamma_poly            rmd_poly/ (kg/m^3)\n"
    for n,p in enumerate(self.poly_pieces):
      s += "  %2d       %.15e %.15e %.15e\n" % (n, p.n, p.ga, p.ds * p.units.density)
    #
    return s
  #
#

def load_pizza(fname):
  """Load an analytic EOS in PIZZA format"""
  udens = PIZZA_UNITS.density
  with open(fname, "r") as fobj:
    header = {}
    values = []
    for line in fobj:
      if "=" in line:
        L = line.split("=")
        header[L[0].strip().lower()] = L[1].strip().lower()
      else:
        values.append([float(x) for x in line.split()])
    try:
      eos_type = header["type"]
      if eos_type == "polytrope":
        poly_n   = float(header["poly_n"])
        poly_rmd = float(header["poly_rmd"])/udens
        return EOS_Poly(poly_rmd, poly_n)
      elif eos_type == "pwpoly":
        rmd_poly0 = float(header["poly_rmd"])/udens
        ga_pieces = [v[1] for v in values]
        rmd_bnd = [v[0]/udens for v in values[1:]]
        return EOS_Piecewise_Poly(rmd_poly0, ga_pieces, rmd_bnd)
      else:
        raise IOError("Unkown eos_type: \"{}\"".format(eos_type))
    except KeyError:
      raise IOError("Could not parse \"{}\"".format(fname))
