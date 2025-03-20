__version__     = "0.9.0"
__author__      = "Hugh McDougall"
__email__       = "hughmcdougallemaiL@gmail.com"
__uri__         = "https://github.com/HughMcDougall/LITMUS/"
__license__     = "Free to use"
__description__ = "JAX-based lag recovery program for AGN reverberation mapping"



import litmus._utils
import litmus._types
import litmus.lin_scatter
import litmus.lightcurve
import litmus.gp_working
import litmus.models
import litmus.ICCF_working
import litmus.fitting_methods
import litmus._ss.clustering
import litmus.mocks
import litmus.logging

from litmus.lightcurve import *
from litmus.litmusclass import LITMUS
