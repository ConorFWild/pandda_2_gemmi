import gc
import time

import numpy
import ray
import multiprocessing as mp
from joblib import Parallel, delayed

from pandda_gemmi.analyse_interface import *
from pandda_gemmi.processing.process_global import SGEResultStatus, SGEFuture


class ProcessLocalSerial(ProcessorInterface):
    def __init__(self,):
        self.tag: Literal["serial"] = "serial"

    def __call__(self, funcs: Iterable[Callable[P, V]]) -> List[V]:
        results = []
        for func in funcs:
            results.append(func())

        return results

    def process_dict(self, funcs):
        return {key: func() for key, func in funcs.items()}


@ray.remote
class RayWrapper(Generic[P, V]):

    def run(self, func: Callable[P, V], *args: P.args, **kwargs: P.kwargs) -> V:
        return func(*args, **kwargs)


@ray.remote
def ray_wrapper(func: Callable[P, V], *args: P.args, **kwargs: P.kwargs) -> V:
    return func(*args, **kwargs)


class ProcessLocalRay(ProcessorInterface):

    def __init__(self, local_cpus):
        self.local_cpus = local_cpus
        ray.init(num_cpus=local_cpus)
        self.tag: Literal["not_async"] = "not_async"

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

    def process_dict(self, funcs):
        assert ray.is_initialized() == True
        tasks = [ray_wrapper.remote(f.func, *f.args, **f.kwargs) for f in funcs.values()]
        # print(tasks)
        results = ray.get(tasks)

        return {key: result for key, result in zip(funcs, results)}

    def reset(self):
        ray.shutdown()
        ray.init(num_cpus=self.local_cpus)


def run_multiprocessing(func: PartialInterface[P, V]) -> V:
    return func()


# class ProcessLocalSpawn(ProcessorInterface):
#
#     def __init__(self, n_jobs: int):
#         self.n_jobs = n_jobs
#         self.pool = mp.Pool(self.n_jobs)
#         self.tag: Literal["not_async"] = "not_async"
#
#     def __call__(self, funcs: Iterable[PartialInterface[P, V]]) -> List[V]:
#         try:
#             mp.set_start_method("spawn")
#         except Exception as e:
#             print(e)
#
#             # results = pool.map(
#             #     run_multiprocessing,
#             #     funcs,
#             # )
#             # results = pool.map(
#             #     run_multiprocessing,
#             #     funcs,
#             # )
#
#         start_time = time.time()
#         result_futuress = []
#         for func in funcs:
#             result_futuress.append(self.pool.apply_async(func.func, args=func.args, kwds=func.kwargs))
#
#         task_status = [result.ready() for result in result_futuress]
#         num_previously_completed = 0
#         num_completed = 0
#         num_tasks = len(task_status)
#         while not all(task_status):
#             current_time = time.time()
#
#             num_completed = len([x for x in task_status if x])
#             if num_completed != 0:
#                 average_time_per_task = round((current_time - start_time) / num_completed, 1)
#             else:
#                 average_time_per_task = "-Unknown-"
#
#             if num_completed > num_previously_completed:
#                 print(f"\tCompleted {num_completed} out of {num_tasks} tasks. Average time per task: {average_time_per_task}.")
#
#             num_previously_completed = num_completed
#             time.sleep(15)
#             task_status = [result.ready() for result in result_futuress]
#
#         current_time = time.time()
#         num_completed = len([x for x in task_status if x])
#         if num_completed != 0:
#             average_time_per_task = round((current_time - start_time) / num_completed, 1)
#         else:
#             average_time_per_task = "-Unknown-"
#
#         print(f"\tFinished tasks! Completed {num_completed} out of {num_tasks} tasks. Average time per task:"
#               f" {average_time_per_task}.")
#
#         results = [result.get() for result in result_futuress]
#
#         return results
#
#     def __getstate__(self):
#         return (self.n_jobs, self.tag)
#
#     def __setstate__(self, state):
#         self.n_jobs = state[0]
#         self.tag = state[1]
#         self.pool = mp.Pool(self.n_jobs)

class ProcessLocalSpawn(ProcessorInterface):

    def __init__(self, n_jobs: int):
        self.n_jobs = n_jobs
        self.pool = Parallel(n_jobs=n_jobs)
        self.tag: Literal["not_async"] = "not_async"

    def __call__(self, funcs: Iterable[PartialInterface[P, V]]) -> List[V]:
        # try:
        #     mp.set_start_method("spawn")
        # except Exception as e:
        #     print(e)
        #
        #     # results = pool.map(
        #     #     run_multiprocessing,
        #     #     funcs,
        #     # )
        #     # results = pool.map(
        #     #     run_multiprocessing,
        #     #     funcs,
        #     # )
        #
        # start_time = time.time()
        # result_futuress = []
        # for func in funcs:
        #     result_futuress.append(self.pool.apply_async(func.func, args=func.args, kwds=func.kwargs))
        #
        # task_status = [result.ready() for result in result_futuress]
        # num_previously_completed = 0
        # num_completed = 0
        # num_tasks = len(task_status)
        # while not all(task_status):
        #     current_time = time.time()
        #
        #     num_completed = len([x for x in task_status if x])
        #     if num_completed != 0:
        #         average_time_per_task = round((current_time - start_time) / num_completed, 1)
        #     else:
        #         average_time_per_task = "-Unknown-"
        #
        #     if num_completed > num_previously_completed:
        #         print(f"\tCompleted {num_completed} out of {num_tasks} tasks. Average time per task: {average_time_per_task}.")
        #
        #     num_previously_completed = num_completed
        #     time.sleep(15)
        #     task_status = [result.ready() for result in result_futuress]
        #
        # current_time = time.time()
        # num_completed = len([x for x in task_status if x])
        # if num_completed != 0:
        #     average_time_per_task = round((current_time - start_time) / num_completed, 1)
        # else:
        #     average_time_per_task = "-Unknown-"
        #
        # print(f"\tFinished tasks! Completed {num_completed} out of {num_tasks} tasks. Average time per task:"
        #       f" {average_time_per_task}.")
        #
        # results = [result.get() for result in result_futuress]

        results = self.pool(delayed(func.func)(*func.args, **func.kwargs) for func in funcs)

        return results

    def process_dict(self, funcs):
        results = self.pool(delayed(func.func)(*func.args, **func.kwargs) for func in funcs)

        return {key: result for key, result in zip(funcs, results)}

    def __getstate__(self):
        return (self.n_jobs, self.tag)

    def __setstate__(self, state):
        self.n_jobs = state[0]
        self.tag = state[1]
        self.pool = Parallel(n_jobs=self.n_jobs)


class ProcessLocalThreading(ProcessorInterface):
    def __init__(self, n_jobs: int):
        self.n_jobs = n_jobs
        self.tag: Literal["not_async"] = "not_async"

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
