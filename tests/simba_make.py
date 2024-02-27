'''
An example of a simba runfile runtest
'''

import SIMBA

#----------------------
if __name__=="__main__":
    A=[10,20,30]
    B=[40,50,60]

    args={"A":A, "B":B}

    SIMBA.make(args=args)

    SIMBA.start(0,comment="testcomment")
    SIMBA.finish(0,comment="testcomment2")
    SIMBA.reset(0,comment="testcomment2")
