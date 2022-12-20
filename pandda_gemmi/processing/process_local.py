import time

import numpy
import ray
import multiprocessing as mp
from joblib import Parallel, delayed

from pandda_gemmi.analyse_interface import *


class ProcessLocalSerial(ProcessorInterface):
    def __call__(self, funcs: Iterable[Callable[P, V]]) -> List[V]:
        results = []
        for func in funcs:
            results.append(func())

        return results


@ray.remote
class RayWrapper(Generic[P, V]):

    def run(self, func: Callable[P, V], *args: P.args, **kwargs: P.kwargs) -> V:
        return func(*args, **kwargs)


@ray.remote
def ray_wrapper(func: Callable[P, V], *args: P.args, **kwargs: P.kwargs) -> V:
    return func(*args, **kwargs)


class ProcessLocalRay(ProcessorInterface):

    def __init__(self, local_cpus):
        ray.init(num_cpus=local_cpus)


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
        return results

    def process_local_ray(self, funcs):
        assert ray.is_initialized() == True
        tasks = [f.func.remote(*f.args, **f.kwargs) for f in funcs]
        results = ray.get(tasks)
        return results


def run_multiprocessing(func: PartialInterface[P, V]) -> V:
    return func()


class ProcessLocalSpawn(ProcessorInterface):

    def __init__(self, n_jobs: int):
        self.n_jobs = n_jobs

    def __call__(self, funcs: Iterable[PartialInterface[P, V]]) -> List[V]:
        try:
            mp.set_start_method("spawn")
        except Exception as e:
            print(e)

        with mp.Pool(self.n_jobs) as pool:
            # results = pool.map(
            #     run_multiprocessing,
            #     funcs,
            # )
            # results = pool.map(
            #     run_multiprocessing,
            #     funcs,
            # )

            start_time = time.time()
            results = []
            for func in funcs:
                results.append(pool.apply(func.func, *func.args, **func.kwargs))

            task_status = [result.ready() for result in results]
            num_previously_completed = 0
            num_completed = 0
            num_tasks = len(task_status)
            while not all(task_status):
                current_time = time.time()

                num_completed = len([x for x in task_status if x])
                if num_completed != 0:
                    average_time_per_task = (current_time - start_time) / num_completed
                else:
                    average_time_per_task = "-Unknown-"

                if num_completed > num_previously_completed:
                    print(f"\tCompleted {num_completed} out of {num_tasks} tasks. Average time per task: {average_time_per_task}.")

                num_previously_completed = num_completed
                time.sleep(15)
                task_status = [result.ready() for result in results]


        return results


class ProcessLocalThreading(ProcessorInterface):
    def __init__(self, n_jobs: int):
        self.n_jobs = n_jobs

    def __call__(self, funcs: Iterable[PartialInterface[P, V]]) -> List[V]:
        results = Parallel(
            n_jobs=self.n_jobs,
            prefer="threads",
        )(
            delayed(run_multiprocessing)(func)
            for func
            in funcs
        )

        return results
