'''
_utils.py
Handy internal utilities for brevity and convenience
'''

# ============================================
# IMPORTS
import sys

import numpy as np
import jax

# ============================================
# DICTIONARY UTILITIES
# ============================================

def isiter(x: any) -> bool:
    '''
    Checks to see if an object is itterable
    '''
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
    if isiter(DICT[key]) and len(DICT[key])>1:
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

def dict_unpack(DICT: dict, keys=None) -> np.array:
    '''
    Unpacks a dictionary into an array format
    :param DICT: the dict to unpack
    :param keys: the order in which to index the keyed elements. If none, will use DICT.keys(). Can be partial
    :return: (nkeys x len_array) np.arrayobject
    '''

    keys = keys if keys is not None else DICT.keys()

    out = np.array([DICT[key] for key in keys])

    return (out)


def dict_pack(X: np.array, keys: [str]) -> np.array:
    '''
    Unpacks a dictionary into an array format
    :param DICT: the dict to pack
    :param keys: the order and labels with which to index the keyed elements
    :return: dict object {key: array} or {key: float}
    '''

    out = {key: X[i] for i, key in enumerate(list(keys))}

    return (out)


def dict_sortby(A: dict, B: dict, match_only=True) -> dict:
    '''
    Sorts dict A to match keys of dict B. If match_only, returns only for keys common to both.
    Else, append un-sorted entries to end
    '''
    out = {key: A[key] for key in B if key in A}
    if not match_only:
        out |= {key: A[key] for key in A if key not in B}
    return (out)


def dict_extend(A: dict, B: dict = None) -> dict:
    '''
    :param A:
    :param B:
    :return:
    '''

    out = {} | A
    if B is not None: out |= B

    to_extend = [key for key in out if not isiter(out[key])]
    to_leave = [key for key in out if isiter(out[key])]

    N = len(out[to_leave[0]])
    for key in to_leave[1:]:
        assert len(out[key]) == N, "Tried to dict_extend() a dictionary with inhomogeneous lengths"

    for key in to_extend:
        out[key] = np.array([A[key]] * N)

    return (out)

# ============================================
# FUNCTION UTILITIES
# ============================================
def pack_function(func, packed_keys, fixed_values = {}):
    '''
    Re-arranges a function that takes dict arguments to tak array-like arguments instead, so as to be autograd friendly
    Takes a function f(D:dict, *arg, **kwargs) and returns f(X, D2, *args, **kwargs), D2 is all elements of D not
    listed in 'packed_keys' or fixed_values.

    :param func: Function to be unpacked
    :param packed_keys: Keys in 'D' to be packed in an array
    :param fixed_values: Elements of 'D' to be fixed
    '''
    def new_func(X, unpacked_params = {}, *args, **kwargs):
        packed_dict = {key: x for key, x in zip(packed_keys, X)}
        packed_dict|=unpacked_params
        packed_dict|= fixed_values

        out = func(packed_dict, *args, **kwargs)
        return (out)

    return(new_func)

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
    print(DICT_ITER, "\tUnpacks to\t", dict_unpack(DICT_ITER))

    print("-" * 24)
    print(dict_unpack(DICT_ITER), "\tPacks to\t", dict_pack(dict_unpack(DICT_ITER), keys=DICT_ITER.keys()))

    print("-" * 24)
    print("Extending array", DICT_ITER | {'b': [4, 5, 6]}, "\tGives \t", dict_extend(DICT_NOITER, {'b': [4, 5, 6]}))


    #-------------------------------------------------------
    def f(D: dict, m=1.0, c=2.0):
        x,y,z = [D[key] for key in list('xyz')]
        out = m*(2*x+3*y+4*z+c)
        return (out)
    fu = pack_function(f, packed_keys=['x'], unpacked_keys = ['y'], fixed_values = {'z': 0.0})
    fu([0.0], {'y': 1.0}, m=1.0, c=2.0)

    fugrad=jax.grad(fu, argnums = 0)




