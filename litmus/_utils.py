'''
_utils.py
Handy internal utilities for brevity and convenience.
Nothing in here is accesible in the public _init_ file
'''

# ============================================
# IMPORTS
import sys

import numpy as np
import jax.numpy as jnp
import jax

from contextlib import contextmanager
import sys, os
from copy import copy

from scipy.special import jnp_zeros

# ============================================
# PRINTING UTILITIES
# ============================================

'''
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr


@contextmanager
class suppress_stdout:
    def __enter__(self):
        self._original_stdout = copy(sys.stdout)
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
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


# TODO - Remove this redundant code when swapping to optimistix
'''
class SuppressStdout:
    def __init__(self):
        # Duplicate the original stdout file descriptor
        self.original_stdout_fd = os.dup(sys.stdout.fileno())
        self.devnull = open(os.devnull, 'w')
        self.is_suppressed = False

    def on(self):
        if not self.is_suppressed:
            # Open devnull and redirect stdout to it
            os.dup2(self.devnull.fileno(), sys.stdout.fileno())
            self.is_suppressed = True

    def off(self):
        if self.is_suppressed:
            # Restore original stdout from the duplicated file descriptor
            os.dup2(self.original_stdout_fd, sys.stdout.fileno())
            self.devnull.close()
            self.is_suppressed = False

sso = SuppressStdout()
'''


# ============================================
# DICTIONARY UTILITIES
# ============================================

def isiter(x: any) -> bool:
    '''
    Checks to see if an object is itterable
    '''
    if type(x) == dict:
        return len(x[list(x.keys())[0]]) > 1
    try:
        iter(x)
    except:
        return (False)
    else:
        return (True)


def isiter_dict(DICT: dict) -> bool:
    '''
    like isiter but for a dictionary. Checks only the first element in DICT.keys
    '''

    key = list(DICT.keys())[0]
    if isiter(DICT[key]):
        return True
    else:
        return False


def dict_dim(DICT: dict) -> (int, int):
    '''
    Checks the first element of a dictionary and returns its length
    '''

    if isiter_dict(DICT):
        firstkey = list(DICT.keys())[0]
        return (len(list(DICT.keys())), len(DICT[firstkey]))
    else:
        return (len(list(DICT.keys())), 1)


# -------------

def dict_pack(DICT: dict, keys=None, recursive=True, H=None, d0={}) -> np.array:
    '''
    Packs a dictionary into an array format
    :param DICT: the dict to unpack
    :param keys: the order in which to index the keyed elements. If none, will use DICT.keys(). Can be partial
    :param recursive: whether to recurse into arrays
    :param H: Matrix to scale parameters by
    :param d0: Value to offset by before packing
    :return: (nkeys x len_array) np.arrayobject

    X = H (d-d0)
    '''

    nokeys = True if keys is None else 0
    keys = keys if keys is not None else DICT.keys()

    for key in keys:
        if key in DICT.keys() and key not in d0.keys(): d0 |= {key: 0.0}

    if recursive and type(list(DICT.values())[0]) == dict:
        out = np.array(
            [dict_pack(DICT[key] - d0[key], keys=keys if not nokeys else None, recursive=recursive) for key in keys])
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
    Sorts dict A to match keys of dict B. If match_only, returns only for keys common to both.
    Else, append un-sorted entries to end
    """
    out = {key: A[key] for key in B if key in A}
    if not match_only:
        out |= {key: A[key] for key in A if key not in B}
    return (out)


def dict_extend(A: dict, B: dict = None) -> dict:
    '''
    Extends all single-length entries of a dict to match the length of a non-singular element
    :param A: Dictionary whose elements are to be extended
    :param B: (optional) the array to extend by, equivalent to dict_extend(A|B)
    :return:
    '''

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
    '''
    Combines an array, list etc of dictionary into a dictionary of arrays
    '''

    N = len(X)
    keys = X[0].keys()

    out = {key: np.zeros(N) for key in keys}
    for n in range(N):
        for key in keys:
            out[key][n] = X[n][key]
    return (out)


def dict_divide(X: dict) -> [dict]:
    '''
    Splits dict of arrays into array of dicts
    '''

    keys = list(X.keys())
    N = len(X[keys[0]])

    out = [{key: X[key][i] for key in X} for i in range(N)]

    return (out)


def dict_split(X: dict, keys: [str]) -> (dict, dict):
    assert type(X) is dict, "input to dict_split() must be of type dict"
    assert isiter(keys) and type(keys[0])==str, "in dict_split() keys must be list of strings"
    A = {key: X[key] for key in keys}
    B = {key: X[key] for key in X.keys() if key not in keys}
    return (A, B)


# ============================================
# FUNCTION UTILITIES
# ============================================
def pack_function(func, packed_keys: ['str'], fixed_values: dict = {}, invert: bool = False, jit: bool = False,
                  H: np.array = None, d0: dict = {}):
    '''
    Re-arranges a function that takes dict arguments to tak array-like arguments instead, so as to be autograd friendly
    Takes a function f(D:dict, *arg, **kwargs) and returns f(X, D2, *args, **kwargs), D2 is all elements of D not
    listed in 'packed_keys' or fixed_values.

    :param func: Function to be unpacked
    :param packed_keys: Keys in 'D' to be packed in an array
    :param fixed_values: Elements of 'D' to be fixed
    :param invert:  If true, will 'flip' the function upside down
    :param jit: If true, will 'jit' the function
    :param H: (optional) scaling matrix to reparameterize H with
    :param x0: (optional) If given, will center the reparameterized  function at x0
    '''

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
    return (np.random.randint(0, sys.maxsize // 1024))


# ===================================
# Testing
if __name__ == "__main__":
    a, b = 1, [1, 2, 3]
    DICT_NOITER = {'a': 3}
    DICT_ITER = {'a': [1, 2, 3], 'b': [4, 5, 6]}

    print(a, "\tItterable?\t", isiter(a))
    print(b, "\tItterable?\t", isiter(b))

    print("-" * 24)
    print(DICT_NOITER, "\tItterable?\t", isiter_dict(DICT_NOITER))
    print(DICT_ITER, "\tItterable?\t", isiter_dict(DICT_ITER))

    print("-" * 24)
    print(DICT_ITER, "\tUnpacks to\t", dict_pack(DICT_ITER))

    print("-" * 24)
    print(dict_pack(DICT_ITER), "\tPacks to\t", dict_unpack(dict_pack(DICT_ITER), keys=DICT_ITER.keys()))

    print("-" * 24)
    print("Extending array", DICT_ITER | {'b': [4, 5, 6]}, "\tGives \t", dict_extend(DICT_NOITER, {'b': [4, 5, 6]}))


    # -------------------------------------------------------
    def f(D: dict, m=1.0, c=2.0):
        x, y, z = [D[key] for key in list('xyz')]
        out = m * (2 * x + 3 * y + 4 * z + c)
        return (out)


    fu = pack_function(f, packed_keys=['x'], fixed_values={'z': 0.0})
    fu([0.0], {'y': 1.0}, m=1.0, c=2.0)

    fugrad = jax.grad(fu, argnums=0)

    # -------------------------------------------------------
    combined_dict = {'a': [0, 1, 2, 3],
                     'B': [10, 11, 12, 13]}
    divided_dict = dict_divide(combined_dict)
    combined_dict_2 = dict_combine(divided_dict)
