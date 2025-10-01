__version__     = "0.9.0"
__author__      = "Hugh McDougall"
__email__       = "hughmcdougallemaiL@gmail.com"
__uri__         = "https://github.com/HughMcDougall/LITMUS/"
__license__     = "Free to use"
__description__ = "JAX-based lag recovery program for AGN reverberation mapping"


import litmus_rm._utils
import litmus_rm._types
import litmus_rm.lin_scatter
import litmus_rm.lightcurve
import litmus_rm.gp_working
import litmus_rm.models
import litmus_rm.ICCF_working
import litmus_rm.fitting_methods
import litmus_rm.mocks
import litmus_rm.logging

from litmus_rm.lightcurve import *
from litmus_rm.litmusclass import LITMUS
