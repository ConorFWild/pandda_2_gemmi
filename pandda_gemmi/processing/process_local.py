import ray

from pandda_gemmi.analyse_interface import *


@ray.remote
class RayWrapper(Generic[P, V]):

    def run(self, func: PartialInterface[P, V], *args, **kwargs) -> V:
        return func(*args, **kwargs)


class ProcessLocalRay(ProcessorInterface):

    def __call__(self, funcs: Iterable[PartialInterface[P, V]]) -> List[V]:
        assert ray.is_initialized() == True
        actors = [RayWrapper.remote() for f in funcs]
        # This isn't properly typeable because the wrapper dynamically costructs the run method on the actor
        print([f for f in funcs])
        print([f.args for f in funcs])
        print(f.kwargs for f in funcs)
        tasks = [a.run.remote(f.func, *f.args, **f.kwargs) for a, f in zip(actors, funcs)]  # type: ignore
        print(tasks)
        results = ray.get(tasks)
        return results

    def process_local_ray(self, funcs):
        assert ray.is_initialized() == True
        tasks = [f.func.remote(*f.args, **f.kwargs) for f in funcs]
        results = ray.get(tasks)
        return results
