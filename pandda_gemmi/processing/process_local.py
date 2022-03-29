import ray

from pandda_gemmi.analyse_interface import *


class ProcessLocalRay(ProcessorInterface):

    def process_local_ray(self, funcs):
        assert ray.is_initialized() == True
        tasks = [f.func.remote(*f.args, **f.kwargs) for f in funcs]
        results = ray.get(tasks)
        return results