import dask
from dask.distributed import Client, progress
from dask_jobqueue import HTCondorCluster, PBSCluster, SGECluster, SLURMCluster

from pandda_gemmi.analyse_interface import *


def run_multiprocessing(func: PartialInterface[P, V]) -> V:
    return func()


class DaskDistributedProcessor(ProcessorInterface):

    def __init__(self,
                 scheduler="SGE",
                 num_workers=10,
                 queue=None,
                 project=None,
                 cores_per_worker=12,
                 distributed_mem_per_core=10,
                 resource_spec="",
                 job_extra=("",),
                 walltime="150:00:00",
                 watcher=True,
                 ):

        dask.config.set({'distributed.worker.daemon': False})

        schedulers = ["HTCONDOR", "PBS", "SGE", "SLURM"]
        if scheduler not in schedulers:
            raise Exception(f"Supported schedulers are: {schedulers}")

        if scheduler == "HTCONDOR":
            job_extra = [(f"GetEnv", "True"), ]
            cluster = HTCondorCluster(
                # queue=queue,
                # project=project,
                cores=cores_per_worker,
                memory=f"{distributed_mem_per_core * cores_per_worker}G",
                # resource_spec=resource_spec,
                # walltime=walltime,
                disk="10G",
                processes=1,
                nanny=watcher,
                job_extra=job_extra,
            )

        elif scheduler == "PBS":
            cluster = PBSCluster(
                queue=queue,
                project=project,
                cores=cores_per_worker,
                memory=f"{distributed_mem_per_core * cores_per_worker}G",
                resource_spec=resource_spec,
                walltime=walltime,
                processes=1,
                nanny=watcher,
            )

        elif scheduler == "SGE":
            extra = [f"-pe smp {cores_per_worker}", "-V"]
            cluster = SGECluster(
                queue=queue,
                project=project,
                cores=cores_per_worker,
                memory=f"{distributed_mem_per_core * cores_per_worker}G",
                resource_spec=resource_spec,
                walltime=walltime,
                processes=1,
                nanny=watcher,
                job_extra=extra,
            )

        elif scheduler == "SLURM":
            cluster = SLURMCluster(
                queue=queue,
                project=project,
                cores=cores_per_worker,
                memory=f"{distributed_mem_per_core * cores_per_worker}GB",
                walltime=walltime,
                processes=1,
                nanny=watcher,
                job_extra=job_extra
            )

        else:
            raise Exception("Something has gone wrong process_global_dask")

        # Scale the cluster up to the number of workers
        cluster.scale(jobs=num_workers)

        # Launch the client
        self.client = Client(cluster)

    def __call__(self, funcs: Iterable[PartialInterface[P, V]]) -> List[V]:

        # func_futures = self.client.scatter(funcs)
        # result_futures = self.client.map(run_multiprocessing, func_futures)
        result_futures = []
        for func in funcs:
            arg_futures = [self.client.scatter(arg) for arg in func.args]
            kwarg_futures = {kwrd: self.client.scatter(kwarg) for kwrd, kwarg in func.kwargs.items()}
            # result_futures.append(self.client.submit(func.func, *func.args, **func.kwargs))
            result_futures.append(self.client.submit(func.func, *arg_futures, **kwarg_futures))

        # progress(result_futures)

        results = self.client.gather(result_futures)

        return results
