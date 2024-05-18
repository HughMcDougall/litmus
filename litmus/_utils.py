'''
_utils.py
Handy internal utilities for brevity and convenience
'''

# ============================================
# IMPORTS
import numpy as np


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
    return (isiter(DICT[key]))


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
