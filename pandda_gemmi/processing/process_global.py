import re
import subprocess
import pickle
import secrets
import time

import dask
from dask.distributed import Client, progress
from dask_jobqueue import HTCondorCluster, PBSCluster, SGECluster, SLURMCluster

from pandda_gemmi.analyse_interface import *


def run_multiprocessing(func: PartialInterface[P, V]) -> V:
    return func()


job_script_template = (
    "#!/bin/sh\n"
    "python {run_process_shell_path} {func_path} {output_path} {arg_paths} {kwarg_paths}\n"
)

submit_command = "qsub -V -pe smp {cores} -l m_mem_free={mem_per_core}G -q medium.q -o {out_path} -e {err_path}.q {" \
                 "job_script_path}"


class QSubScheduler:
    def __init__(self,
                 queue,
                 project,
                 cores,
                 mem_per_core,
                 walltime,
                 job_extra,
                 tmp_dir):
        self.queue = queue
        self.project = project
        self.cores = cores
        self.mem_per_core = mem_per_core
        self.walltime = walltime
        self.job_extra = job_extra
        self.tmp_dir: Path = tmp_dir

    def generate_io_path(self, ):
        code = secrets.token_hex(16)
        path = self.tmp_dir / code
        return path, code

    def save(self, obj):
        code = secrets.token_hex(16)
        path = self.tmp_dir / code
        with open(path, "wb") as f:
            pickle.dump(obj, f)

        return path

    def chmod(self, path):
        p = subprocess.Popen(
            f"chmod 777 {path}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = p.communicate()
        print(stdout)
        print(stderr)

    def write_script(self, job_script, job_script_path):
        with open(job_script_path, "w") as f:
            f.write(job_script)

        self.chmod(job_script_path)

    def submit(self, func_path, *arg_paths, **kwarg_paths):
        # Write job to file

        job_script_path, code = self.generate_io_path()

        arg_paths_string = ""
        for arg_path in arg_paths:
            arg_paths_string += f" {arg_path}"

        kwarg_paths_string = ""
        for kwrd, kwarg_path in kwarg_paths.items():
            kwarg_paths_string += f" --{kwrd}={kwarg_path}"

        output_path, code = self.generate_io_path()

        job_script = job_script_template.format(
            func_path,
            output_path,
            arg_paths,
            kwarg_paths,
        )
        print(job_script)

        self.write_script(job_script, job_script_path)

        # Get submit command
        out_path = self.tmp_dir / f"{code}.out"
        err_path = self.tmp_dir / f"{code}.err"

        _submit_command = submit_command.format(
            cores=self.cores,
            mem_per_core=self.mem_per_core,
            job_script_path=job_script_path,
            out_path=out_path,
            err_path=err_path
        )
        print(_submit_command)

        # Submit
        p = subprocess.Popen(
            _submit_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = p.communicate()
        print(stdout)
        print(stderr)

        job_id = re.match("Your job ([0-9]+) ", str(stdout)).groups()

        return output_path

    def load(self, path):
        with open(path, "rb") as f:
            obj = pickle.load(f)

        return obj

    def gather(self, output_paths):
        task_status = [output_path.exists() for output_path in output_paths]
        while not all(task_status):
            task_status = [output_path.exists() for output_path in output_paths]
            completed = [x for x in task_status if x]

            print(f"\tCompleted {len(completed)} out of {len(task_status)} tasks...")
            time.sleep(60)

        results = [self.load(output_path) for output_path in output_paths]

        return results

class DistributedProcessor(ProcessorInterface):
    def __init__(self,
                 tmp_dir,
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
        schedulers = ["HTCONDOR", "PBS", "SGE", "SLURM"]
        if scheduler not in schedulers:
            raise Exception(f"Supported schedulers are: {schedulers}")

        if scheduler == "HTCONDOR":
            job_extra = [(f"GetEnv", "True"), ]
            raise NotImplementedError("PBS cluster is not implemented")

        elif scheduler == "PBS":
            raise NotImplementedError("HTCONDOR cluster is not implemented")

        elif scheduler == "SGE":
            extra = [f"-pe smp {cores_per_worker}", "-V"]
            cluster = QSubScheduler(
                queue=queue,
                project=project,
                cores=cores_per_worker,
                mem_per_core=f"{distributed_mem_per_core}G",
                walltime=walltime,
                job_extra=extra,
                tmp_dir=tmp_dir
            )

        elif scheduler == "SLURM":
            raise NotImplementedError("SLURM cluster is not implemented")

        else:
            raise Exception(f"Scheduler {scheduler} is not one of the supported schedulers: {schedulers}")

        self.client = cluster

    def __call__(self, funcs: Iterable[PartialInterface[P, V]]) -> List[V]:
        result_futures = []
        for func in funcs:
            func_path = self.client.save(func.func)
            arg_pickle_paths = [self.client.save(arg) for arg in func.args]
            kwarg_pickle_paths = {kwrd: self.client.save(kwarg) for kwrd, kwarg in func.kwargs.items()}
            # result_futures.append(self.client.submit(func.func, *func.args, **func.kwargs))
            result_futures.append(self.client.submit(func_path, *arg_pickle_paths, **kwarg_pickle_paths))

        # progress(result_futures)

        results = self.client.gather(result_futures)

        return results


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
