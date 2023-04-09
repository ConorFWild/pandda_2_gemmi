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
def ray_batch_wrapper(funcs):
    return [f() for f in funcs]


class ProcessLocalRay(ProcessorInterface):

    def __init__(self, local_cpus):
        self.local_cpus = local_cpus
        ray.init(num_cpus=local_cpus)
        self.tag: Literal["not_async"] = "not_async"

    def put(self, object):
        return ray.put(object)

    def __call__(self, funcs: Iterable[PartialInterface[P, V]]) -> List[V]:
        assert ray.is_initialized() == True
        # actors = [RayWrapper.remote() for f in funcs]
        # This isn't properly typeable because the wrapper dynamically costructs the run method on the actor
        # print([f for f in funcs])
        # print([f.args for f in funcs])
        # print(f.kwargs for f in funcs)
        # tasks = [a.run.remote(f.func, *f.args, **f.kwargs) for a, f in zip(actors, funcs)]  # type: ignore
        tasks = [ray_wrapper.remote(f.func, *f.args, **f.kwargs) for f in funcs]
        # print(tasks)
        results = ray.get(tasks)
        # gc.collect()
        return results

    def process_local_ray(self, funcs):
        assert ray.is_initialized() == True
        tasks = [f.func.remote(*f.args, **f.kwargs) for f in funcs]
        results = ray.get(tasks)
        return results




    # def process_dict(self, funcs):
    #     assert ray.is_initialized() == True
    #     tasks = [ray_wrapper.remote(f.func, *f.args, **f.kwargs) for f in funcs.values()]
    #     # print(tasks)
    #
    #     results = ray.get(tasks)
    #
    #     return {key: result for key, result in zip(funcs, results)}

    def process_dict(self, funcs, ):
        assert ray.is_initialized() == True
        key_list = list(funcs.keys())
        func_list = list(funcs.values())
        num_keys = len(key_list)

        batch_size = int(len(funcs) / self.local_cpus) + 1

        tasks = []
        for j in range(self.local_cpus):
            tasks.append(
                ray_batch_wrapper.remote(func_list[j*batch_size: min(num_keys, (j+1)*batch_size)])
            )

        # tasks = [ray_wrapper.remote(f.func, *f.args, **f.kwargs) for f in funcs.values()]
        # print(tasks)

        results = ray.get(tasks)
        result_dict = {}
        j = 0
        for result in results:
            for r in result:
                result_dict[key_list[j]] = r

        return result_dict
        # return {key: result for key, result in zip(funcs, results)}

    def reset(self):
        ray.shutdown()
        ray.init(num_cpus=self.local_cpus)
