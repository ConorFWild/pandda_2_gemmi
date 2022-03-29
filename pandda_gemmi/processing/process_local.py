import ray

from pandda_gemmi.analyse_interface import *



@ray.remote
class RayWrapper(RayCompatibleInterface, Generic[P, V]):
    def __init__(self, func: Callable[P, V]):
        self.partial = func

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> V:
        return self.func(*args, **kwargs)


class ProcessLocalRay(ProcessorInterface):

    def __call__(self, funcs: List[PartialInterface[P, V]]) -> List[V]:
        assert ray.is_initialized() == True
        actors = [f.func.remote() for f in funcs]
        tasks = [a.__call__.remote(*f.args, **f.kwargs) for a, f in zip(actors, funcs)]
        results = ray.get(tasks)
        return results

    def process_local_ray(self, funcs):
        assert ray.is_initialized() == True
        tasks = [f.func.remote(*f.args, **f.kwargs) for f in funcs]
        results = ray.get(tasks)
        return results
