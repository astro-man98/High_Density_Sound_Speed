from matplotlib import pyplot as plt
import numpy as np
from PyCompOSE.compose.eos import Metadata,Table
import os
import glob
from RVSS_EOS_Slope import RVSS_EOS_Slope as rvss_eos
from RMFT.rmft_eos import RMFT_Finite_T
from scipy.interpolate import PchipInterpolator,interp1d
from Modules.TOV_Solver_RK45_Tides import TovSolverTwoFluidRK45 as tov
import kuibit.simdir as SD
import h5py as h5


class coreEOS:
    def __init__(self,eos_path,name,m1,m2,q):
        self.name = name
        self.m1 = m1
        self.m2 = m2
        self.q = q
        self.path = eos_path if os.path.isdir(eos_path) else None

    