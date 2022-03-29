from pandda_gemmi.analyse_interface import *


class Partial(PartialInterface):
    def __init__(self,
                 func: T,
                 *args: P.args,
                 **kwargs: P.kwargs,
                 ):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, ) -> V:
        return self.func(*self.args, **self.kwargs)
