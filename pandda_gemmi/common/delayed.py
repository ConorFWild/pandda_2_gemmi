from types import MethodType
from typing import Any

import dataclasses


@dataclasses.dataclass()
class DelayedFuncReady:
    func: MethodType
    args: Any

    def __call__(self) -> Any:
        return self.func(*self.args)


@dataclasses.dataclass()
class DelayedFuncWaiting:
    func: MethodType

    def __call__(self, *args: Any) -> Any:
        return DelayedFuncReady(self.func, args)


def delayed(func: MethodType):
    return DelayedFuncWaiting(func)
