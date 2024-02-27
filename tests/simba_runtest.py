import SIMBA
from argparse import ArgumentParser

def job(args):
    file = open("./out-%s.dat" %args['i'],"w")
    file.write(str(args["A"])+"\t"+str(args["B"]))



def main():
    ap = ArgumentParser(description='This is a test argparse file')
    ap.add_argument('-i', '--i', metavar='i', type=int,    help='job index',         default=0)
    args = ap.parse_args()

    i=args.i
    SIMBA.start(i)
    func_args =     SIMBA.get_args(i)
    job(func_args)
    SIMBA.finish(i)
    

if __name__ == "__main__":
    main()
