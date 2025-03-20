"""
handy utils for type hinting
"""
from typing import Annotated, Literal, TypeVar, Union, Iterable

from types import MethodType

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from numpy.typing import NDArray

from numpy import generic

DType = TypeVar("DType", bound=generic)
ArrayN = Annotated[NDArray[DType], Literal["N"]]
ArrayNxN = Annotated[NDArray[DType], Literal["N", "N"]]
ArrayM = Annotated[NDArray[DType], Literal["M"]]
ArrayMxM = Annotated[NDArray[DType], Literal["M", "M"]]
ArrayNxMxM = Annotated[NDArray[DType], Literal["M", "N", "N"]]
