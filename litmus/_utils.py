"""
_utils.py
Handy internal utilities for brevity and convenience.
Nothing in here is accesible in the public _init_ file
"""

# ============================================
# IMPORTS
import sys

import numpy as np
import jax.numpy as jnp
import jax

from contextlib import contextmanager
from IPython import get_ipython
from IPython.utils import io
import sys, os
from copy import copy

import litmus._types as _types

from scipy.special import jnp_zeros

# ============================================
# PRINTING UTILITIES
# ============================================

'''
@contextmanager
def suppress_stdout():
    # Duplicate the original stdout file descriptor to restore later
    original_stdout_fd = os.dup(sys.stdout.fileno())

    # Open devnull file and redirect stdout to it
    with open(os.devnull, 'w') as devnull:
        os.dup2(devnull.fileno(), sys.stdout.fileno())
        try:
            yield
        finally:
            # Restore original stdout from the duplicated file descriptor
            os.dup2(original_stdout_fd, sys.stdout.fileno())
            # Close the duplicated file descriptor
            os.close(original_stdout_fd)
'''


@contextmanager
def suppress_stdout():
    ipython = get_ipython()

    if ipython and hasattr(ipython, 'kernel'):  # Likely in a Jupyter notebook
        with io.capture_output() as captured:
            yield
    else:
        # Standard Python environment
        original_stdout_fd = os.dup(sys.stdout.fileno())
        with open(os.devnull, 'w') as devnull:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            try:
                yield
            finally:
                os.dup2(original_stdout_fd, sys.stdout.fileno())
                os.close(original_stdout_fd)

# ============================================
# DICTIONARY UTILITIES
# ============================================

def isiter(x: any) -> bool:
    """
    Checks to see if an object is itterable
    """
    if type(x) == dict:
        return len(x[list(x.keys())[0]]) > 1
    try:
        iter(x)
    except:
        return (False)
    else:
        return (True)


def isiter_dict(DICT: dict) -> bool:
    """
    like isiter but for a dictionary. Checks only the first element in DICT.keys
    """

    key = list(DICT.keys())[0]
    if isiter(DICT[key]):
        return True
    else:
        return False


def dict_dim(DICT: dict) -> (int, int):
    """
    Checks the first element of a dictionary and returns its length
    """

    if isiter_dict(DICT):
        firstkey = list(DICT.keys())[0]
        return (len(list(DICT.keys())), len(DICT[firstkey]))
    else:
        return (len(list(DICT.keys())), 1)


# -------------

def dict_pack(DICT: dict, keys=None, recursive=True, H=None, d0={}) -> np.array:
    """
    Packs a dictionary into an array format
    :param DICT: the dict to unpack
    :param keys: the order in which to index the keyed elements. If none, will use DICT.keys(). Can be partial
    :param recursive: whether to recurse into arrays
    :param H: Matrix to scale parameters by
    :param d0: Value to offset by before packing
    :return: (nkeys x len_array) np.arrayobject

    X = H (d-d0)
    """

    nokeys = True if keys is None else 0
    keys = keys if keys is not None else DICT.keys()

    if d0 is {}: d0 = {key:0 for key in keys}

    for key in keys:
        if key in DICT.keys() and key not in d0.keys(): d0 |= {key: 0.0}

    if recursive and type(list(DICT.values())[0]) == dict:
        out = np.array(
            [dict_pack(DICT[key] - d0[key], keys=keys if not nokeys else None, recursive=recursive) for key in keys])
    else:
        if isiter(DICT[list(keys)[0]]):
            out = np.array([[DICT[key][i] - d0[key] for i in range(dict_dim(DICT)[1])] for key in keys])
        else:
            out = np.array([DICT[key] - d0[key] for key in keys])

    return (out)


def dict_unpack(X: np.array, keys: [str], recursive=True, Hinv=None, x0=None) -> np.array:
    """
    Unpacks an array into a dict
    :param X: Array to unpack
    :param keys: keys to unpack with
    :return:

    Hinv(X) + x0
    """
    if Hinv is not None: assert Hinv.shape[0] == len(keys), "Size of H must be equal to number of keys in dict_unpack"

    if recursive and isiter(X[0]):
        out = {key: dict_unpack(X[i], keys, recursive) for i, key in enumerate(list(keys))}
    else:
        X = X.copy()
        if Hinv is not None:
            X = np.dot(Hinv, X)
        if x0 is not None:
            X += x0
        out = {key: X[i] for i, key in enumerate(list(keys))}

    return (out)


def dict_sortby(A: dict, B: dict, match_only=True) -> dict:
    """
    Sorts dict A to match keys of dict B.

    :param A: Dict to be sorted
    :param B: Dict whose keys are will provide the ordering
    :param match_only: If true, returns only for keys common to both A and B. Else, append un-sorted entries to end
    :return: {key: A[key] for key in B if key in A}
    """
    out = {key: A[key] for key in B if key in A}
    if not match_only:
        out |= {key: A[key] for key in A if key not in B}
    return (out)


def dict_extend(A: dict, B: dict = None) -> dict:
    """
    Extends all single-length entries of a dict to match the length of a non-singular element
    :param A: Dictionary whose elements are to be extended
    :param B: (optional) the array to extend by, equivalent to dict_extend(A|B)
    :return: Dict A with any singleton elements extended to the longest entry in A or B
    """

    out = A.copy()
    if B is not None: out |= B

    to_extend = [key for key in out if not isiter(out[key])]
    to_leave = [key for key in out if isiter(out[key])]

    if len(to_extend) == 0: return out
    if len(to_leave) == 0: return out

    N = len(out[to_leave[0]])
    for key in to_leave[1:]:
        assert len(out[key]) == N, "Tried to dict_extend() a dictionary with inhomogeneous lengths"

    for key in to_extend:
        out[key] = np.array([A[key]] * N)

    return (out)


def dict_combine(X: [dict]) -> {str: [float]}:
    """
    Combines an array, list etc. of dictionaries into a dictionary of arrays

    :param X: 1D Iterable of dicts
    :return: Dict of 1D iterables
    """

    N = len(X)
    keys = X[0].keys()

    out = {key: np.zeros(N) for key in keys}
    for n in range(N):
        for key in keys:
            out[key][n] = X[n][key]
    return (out)


def dict_divide(X: dict) -> [dict]:
    """
    Splits dict of arrays into array of dicts. Opposite of dict_combine

    :param X: Dict of 1D iterables
    :return: 1D Iterable of dicts
    """

    keys = list(X.keys())
    N = len(X[keys[0]])

    out = [{key: X[key][i] for key in X} for i in range(N)]

    return (out)


def dict_split(X: dict, keys: [str]) -> (dict, dict):
    """
    Splits a dict in two based on keys

    :param X: Dict to be split into A,B
    :param keys: Keys to be present in A, but not in B
    :return: tuple of dicts (A,B)
    """
    assert type(X) is dict, "input to dict_split() must be of type dict"
    assert isiter(keys) and type(keys[0])==str, "in dict_split() keys must be list of strings"
    A = {key: X[key] for key in keys}
    B = {key: X[key] for key in X.keys() if key not in keys}
    return (A, B)


# ============================================
# FUNCTION UTILITIES
# ============================================
def pack_function(func, packed_keys: ['str'], fixed_values: dict = {}, invert: bool = False, jit: bool = False,
                  H: np.array = None, d0: dict = {}) -> _types.FunctionType:
    """
    Re-arranges a function that takes dict arguments to tak array-like arguments instead, so as to be autograd friendly
    Takes a function f(D:dict, *arg, **kwargs) and returns f(X, D2, *args, **kwargs), D2 is all elements of D not
    listed in 'packed_keys' or fixed_values.

    :param func: Function to be unpacked
    :param packed_keys: Keys in 'D' to be packed in an array
    :param fixed_values: Elements of 'D' to be fixed
    :param invert:  If true, will 'flip' the function upside down
    :param jit: If true, will 'jit' the function
    :param H: (optional) scaling matrix to reparameterize H with
    :param d0: (optional) If given, will center the reparameterized  function at x0
    """

    if H is not None:
        assert H.shape[0] == len(packed_keys), "Scaling matrix H must be same length as packed_keys"
    else:
        H = jnp.eye(len(packed_keys))
    d0 = {key: 0.0 for key in packed_keys} | d0
    x0 = dict_pack(d0, packed_keys)

    # --------

    sign = -1 if invert else 1

    # --------
    def new_func(X, unpacked_params={}, *args, **kwargs):
        X = jnp.dot(H, X - x0)
        packed_dict = {key: x for key, x in zip(packed_keys, X)}
        packed_dict |= unpacked_params
        packed_dict |= fixed_values

        out = func(packed_dict, *args, **kwargs)
        return (sign * out)

    # --------
    if jit: new_func = jax.jit(new_func)

    return (new_func)


# ============================================
# RANDOMIZATION UTILITIES
# ============================================
def randint():
    """
    Quick utility to generate a random integer
    """
    return (np.random.randint(0, sys.maxsize // 1024))

