"""
logging.py

Collection of logging functions in a handy inheritable class, to make it easier to edit later in one place
"""

import sys
from typing import Union


class logger:
    """
    Object class that contains methods for printing to debug, verbose and error streams.
    Internal behaviour may change in future releases
    """

    def __init__(self, out_stream=sys.stdout, err_stream=sys.stderr,
                 verbose: Union[bool, int] = True,
                 debug: Union[bool, int] = False,
                 warn: Union[bool, int] = 0):
        """
        :param out_stream: Stream to print run and debug messages to
        :param err_stream: Stream to print error messages to
        :param verbose: Whether to print msg_run statements / lvl of reporting
        :param debug: Whether to print msg_debug statements / lvl of reporting
        :param warn: Whether to print msg_debug statements / lvl of warnings
        """
        # ----------------------------

        self.out_stream = out_stream
        self.err_stream = err_stream
        self.verbose = verbose
        self.debug = debug
        self.warn = warn

    # ----------------------
    # Error message printing
    def msg_err(self, *x: str, end: str = '\n', delim: str = ' ', lvl: int = 0):
        """
        Messages for when something has broken or been called incorrectly
        :param x: string(s) to be printed
        :param end: string to be printed at end of the msg
        :param delim: string to be printed at delimiter between messages
        :param lvl: logging level. int >=0
        """
        lvl = max(lvl, 0)
        if self.warn >= lvl:
            for i, a in enumerate(x):
                print(a, file=self.err_stream, end=delim if i != len(x) else "")
            print('', end=end)
        return

    def msg_run(self, *x: str, end: str = '\n', delim: str = ' ', lvl: int = 1):
        """
        Standard messages about when things are running
        :param x: string(s) to be printed
        :param end: string to be printed at end of the msg
        :param delim: string to be printed at delimiter between messages
        :param lvl: logging level. int >=1
        """
        lvl = max(lvl, 1)
        if self.verbose >= lvl:
            for i, a in enumerate(x):
                print(a, file=self.out_stream, end=delim if i != len(x) else "")
            print('', end=end)
        return

    def msg_debug(self, *x: str, end: str = '\n', delim: str = ' ', lvl: int = 1):
        """
        Explicit messages to help debug when things are behaving strangely
        :param x: string(s) to be printed
        :param end: string to be printed at end of the msg
        :param delim: string to be printed at delimiter between messages
        :param lvl: logging level. int >=1
        """
        lvl = max(lvl, 1)
        if self.debug >= lvl:
            for i, a in enumerate(x):
                print(a, file=self.out_stream, end=delim if i != len(x) else "")

            print('', end=end)
        return
