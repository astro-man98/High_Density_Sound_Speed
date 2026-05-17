#!/usr/bin/env python
#
# EOS_Tables: manages EOS tables for THC
# Loosely based on Filippo Galeazzi's Matlab scripts
# Copyright (C) 2016, David Radice <dradice@caltech.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

def myinterp1d(x, y, **kwargs):
    """
    Custom 1d interpolator
    """
    return interp1d(x, y, kind='cubic', bounds_error=True, **kwargs)

def find_temp_given_ent(temp1d, ye1d, ent2d, entropy):
    """
    Find the temperature such that S(T, Ye) = S0 for each ye

    * temp1d  : 1d grid of temperatures
    * ye1d    : 1d grid of Ye
    * ent2d   : S[iye,itemp]
    * entropy : wanted entropy
    """
    tout = np.zeros_like(ye1d)
    for iye in range(ye1d.shape[0]):
        f = myinterp1d(temp1d, (ent2d[iye,:] - entropy)**2)
        res = minimize_scalar(f, bounds=(temp1d[0], temp1d[-1]), method='bounded',
            options = {'xatol': 1e-2, 'maxiter': 100})
        tout[iye] = res.x
    return tout

def find_beta_eq(ye, mu_e, mu_n, mu_p):
    """
    Find the neutrino-less beta equilibrium ye for each point
    in a 1D table tab(ye).

    Beta equilibrium is found from the condition

    mu_n = mu_p + mu_e

    * ye   : electron fraction
    * mu_e : relativistic electron chemical potential
    * mu_n : relativistic electron chemical potential
    * mu_p : relativistic electron chemical potential
    """
    mu_nu = mu_p + mu_e - mu_n

    # These cases have beta-equilibrium out of the table
    if np.all(mu_nu > 0):
        return ye[0]
    if np.all(mu_nu < 0):
        return ye[-1]

    f = myinterp1d(ye, mu_nu**2)
    res = minimize_scalar(f, bounds=(ye[0], ye[-1]), method='bounded',
            options = {'xatol': 1e-6, 'maxiter': 100})
    return res.x
