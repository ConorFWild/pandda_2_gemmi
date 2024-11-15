import time

import ray

from ..interfaces import *


@ray.remote
class RayWrapper(Generic[P, V]):

    def run(self, func: Callable[P, V], *args: P.args, **kwargs: P.kwargs) -> V:
        return func(*args, **kwargs)


@ray.remote
def ray_wrapper(func: Callable[P, V], *args: P.args, **kwargs: P.kwargs) -> V:
    return func(*args, **kwargs)


@ray.remote
def ray_batch_wrapper(funcs, args, kwargs):
    result = [f(*args, **kwargs) for f, args, kwargs in zip(funcs, args, kwargs)]
    return result


class ProcessLocalRay(ProcessorInterface):

    def __init__(self, local_cpus):
        self.local_cpus = local_cpus
        ray.init(num_cpus=local_cpus)
        self.tag: Literal["not_async"] = "not_async"

    def put(self, object):
        return ray.put(object)

    def get(self, ref):
        return ray.get(ref)

    def __call__(self, funcs: Iterable[PartialInterface[P, V]]) -> List[V]:
        assert ray.is_initialized() == True
        tasks = [ray_wrapper.remote(f.func, *f.args, **f.kwargs) for f in funcs]
        results = ray.get(tasks)
        return results

    def process_local_ray(self, funcs):
        assert ray.is_initialized() == True
        tasks = [f.func.remote(*f.args, **f.kwargs) for f in funcs]
        results = ray.get(tasks)
        return results

    def process_dict(self, funcs):
        assert ray.is_initialized() == True
        tasks = [ray_wrapper.remote(f.func, *f.args, **f.kwargs) for f in funcs.values()]
        results = ray.get(tasks)

        return {key: result for key, result in zip(funcs, results)}

    def reset(self):
        ray.shutdown()
        ray.init(num_cpus=self.local_cpus)

    def shutdown(self):
        ray.shutdown()