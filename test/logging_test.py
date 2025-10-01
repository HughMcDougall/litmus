from litmus_rm.logging import *

if __name__ == '__main__':
    print("Making Logger")
    mylog = logger(verbose=True, debug=True)
    print(mylog.verbose)
    mylog.msg_err("Test 1", end="\n")
    mylog.msg_err("Test 2")
