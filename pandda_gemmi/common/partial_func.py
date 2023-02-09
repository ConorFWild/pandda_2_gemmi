from pandda_gemmi.analyse_interface import *


class Partial(PartialInterface[P, V]):
    def __init__(self,
                 func: Callable[P, V],

                 ):
        self.func = func
        self.args = []
        self.kwargs = {}

    def paramaterise(self,
                     *args: P.args,
                     **kwargs: P.kwargs,
                     ):
        self.args = self.args + [_arg for _arg in args]
        self.kwargs = self.kwargs.update(kwargs)
        # return self.func(*self.args, **self.kwargs)
        return self

    def __call__(self, ) -> V:
        return self.func(*self.args, **self.kwargs)
