
# ===================================
# Testing
if __name__ == "__main__":

    from litmus._utils import *

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
