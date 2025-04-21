"""
A single ref package to create and pull in types for type hinting
"""

from typing import Annotated, Literal, TypeVar, Union, Iterable, Callable, Any

from types import MethodType, FunctionType

from matplotlib.figure import Figure

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from numpy.typing import NDArray

from numpy import generic

DType = TypeVar("DType", bound=generic)
"""Any Dtype"""
ArrayN = Annotated[NDArray[DType], Literal["N"]]
"""1D array corresponding to N sample sites, e.g. an array of log densities"""
ArrayNxN = Annotated[NDArray[DType], Literal["N", "N"]]
"""2D array corresponding to NxN sample sites, e.g. a GP covariance matrix"""
ArrayM = Annotated[NDArray[DType], Literal["M"]]
"""1D array corresponding to M parameters, e.g. a grad"""
ArrayMxM = Annotated[NDArray[DType], Literal["M", "M"]]
"""2D array corresponding to MxM parameters, e.g. a hessian"""
ArrayNxMxM = Annotated[NDArray[DType], Literal["M", "N", "N"]]
"""3D array corresponding to N sheets of MxM arrays for N data points and M parameters, e.g. a plate of hessians"""