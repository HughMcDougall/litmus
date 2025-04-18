'''
logging.py

Collection of logging functions in a handy inheritable class, to make it easier to edit later in one place
'''

import sys


class logger:
    """
    Object class that contains methods for printing to debug, verbose and error streams.
    Somewhat a placeholder atm
    """

    def __init__(self, out_stream=sys.stdout, err_stream=sys.stderr, verbose: bool = True, debug: bool = False):
        # ----------------------------

        self.out_stream = out_stream
        self.err_stream = err_stream
        self.verbose = verbose
        self.debug = debug

    # ----------------------
    # Error message printing
    def msg_err(self, *x: str, end='\n', delim=' '):
        """
        Messages for when something has broken or been called incorrectly
        """
        if True:
            for a in x:
                print(a, file=self.err_stream, end=delim)

        print(end, end='')
        return

    def msg_run(self, *x: str, end='\n', delim=' '):
        """
        Standard messages about when things are running
        """
        if self.verbose:
            for a in x:
                print(a, file=self.out_stream, end=delim)

        print(end, end='')
        return

    def msg_verbose(self, *x: str, end='\n', delim=' '):
        """
        Explicit messages to help debug when things are behaving strangely
        """
        if self.debug:
            for a in x:
                print(a, file=self.out_stream, end=delim)

        print(end, end='')
        return
