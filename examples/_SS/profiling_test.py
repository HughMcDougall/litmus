

import cProfile

pr = cProfile.Profile()
pr.enable()


def f(x):
    return(x**2)

for i in range(int(1E5)):
    a = f(2)

pr.disable()
# after your program ends
pr.print_stats(sort="calls")
