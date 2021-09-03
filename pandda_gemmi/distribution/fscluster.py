from time import sleep
from enum import Enum, auto
from typing import *
from pathlib import Path
import subprocess
import re
import pickle
import secrets
import os


def shell(command: str):
    p = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout, stderr = p.communicate()

    return stdout, stderr


def chmod(path: Path):
    command = f"chmod 777 {path}"
    shell(command)


def write(script, file):
    with open(file, "w") as f:
        f.write(script)

    chmod(file)


class Run:
    def __init__(self,
                 f,
                 output_file,
                 # target_file,
                 ):
        self.f = f
        self.output_file = output_file
        # self.target_file = target_file

    def __call__(self):
        result = self.f()

        with open(self.output_file, "wb") as f:
            pickle.dump(result, f)

        # with open(self.target_file, "w") as f:
        #     f.write("done")


class SGE:
    ...


class SLURM:
    JOB_SCRIPT = (
        "#!/bin/bash\n"
        "#SBATCH --job-name={job_name}\n"
        "#SBATCH --partition={partition}\n"
        "#SBATCH --ntasks=1\n"
        "#SBATCH --cpus-per-task={cpus}\n"
        "#SBATCH --mem-per-cpu={mem_per_cpu}G\n"
        "#SBATCH --output=slurm-%j.out\n"
        "#SBATCH --error=slurm-%j.err\n"
        "#SBATCH --exclusive      \n"

        "{executable_file}\n"
    )

    def __init__(self):
        ...

    def submit(self):
        ...

    def running(self, key):
        ...


class HTCONDOR:
    JOB_SCRIPT = (
        "#################### \n"
        "# \n"
        "# Example 1                                   \n"
        "# Simple HTCondor submit description file \n"
        "#                          \n"
        "####################    \n"

        "Executable   = {executable_file} \n"
        "Log          = {log_file} \n"
        "Output = {output_file} \n"
        "Error = {error_file} \n"

        "RequestCpus = {request_cpus}\n"
        "request_memory = {request_memory} GB \n"

        "GetEnv = True\n"

        "Queue"
    )

    def __init__(self,
                 output_dir,
                 distributed_cores_per_worker,
                 distributed_mem_per_core,
                 ):
        self.output_dir = output_dir
        self.distributed_cores_per_worker = distributed_cores_per_worker
        self.distributed_mem_per_core = distributed_mem_per_core

    def submit(self, func, debug=True):
        # Assign the key to the future
        key = str(secrets.token_hex(16))
        if debug:
            print(f"\tKey is: {key}")

        # Wrap func to pickle it's result
        input_file = self.output_dir / f"{key}.in.pickle"
        output_file = self.output_dir / f"{key}.out.pickle"
        func_ob = Run(func, output_file)
        with open(input_file, "wb") as f:
            pickle.dump(func_ob, f)

        if debug:
            print(f"\tInput file is: {input_file}")
            print(f"\tOutput file is: {output_file}")

        # Script to run on worker: needs to be saved by this process/removed by future
        run_script = f"#!/bin/bash\npython {Path(__file__).parent}/run.py {input_file}"
        run_script_file = self.output_dir / f"{key}.run.sh"
        write(run_script, run_script_file)
        if debug:
            print(f"\tRun script is: {run_script}")
            print(f"\tRun script file is: {run_script_file}")

        # describe job to scheduler: needs to be saved/removed by this process
        job_script = self.JOB_SCRIPT.format(
            executable_file=run_script_file,
            log_file=self.output_dir / f"{key}.log",
            output_file=self.output_dir / f"{key}.out",
            error_file=self.output_dir / f"{key}.err",
            request_cpus=self.distributed_cores_per_worker,
            request_memory=self.distributed_mem_per_core * self.distributed_cores_per_worker,
        )
        job_script_file = self.output_dir / f"{key}.job"
        write(job_script, job_script_file)

        if debug:
            print(f"\tJob script is: {job_script}")
            print(f"\tJob script file file is: {job_script_file}")

        # code to submit job to sceduler
        submit_script = f"condor_submit {job_script_file}"

        if debug:
            print(f"\tSubmit script is: {submit_script}")

        # run the submitscript locally
        shell(submit_script)

        # Construct the future
        future = FSFuture(
            key=key,
            run_script_file=run_script_file,
            input_file=input_file,
            target_file=output_file,
        )

        return future

    def running(self, key, debug=True):
        stdout, stderr = shell("condor_q --json")

        if debug:
            print(str(stdout))
            print(str(stderr))

        if re.search(key, str(stdout)):
            return True
        else:
            return False


class FutureStatus(Enum):
    RUNNING = auto()
    FAILED = auto()
    DONE = auto()


class Scheduler(Enum):
    SGE = SGE
    SLURM = SLURM
    HTCONDOR = HTCONDOR


class FSFuture:
    def __init__(self,
                 key: str,
                 run_script_file: Path,
                 input_file: Path,
                 target_file: Path,
                 ):
        self.key = key
        self.run_script_file = run_script_file
        self.input_file = input_file
        self.target_file = target_file

    def status(self, scheduler):
        running = scheduler.running(self.key)
        finished = self.target_file.exists()

        if running and not finished:
            return FutureStatus.RUNNING
        elif finished and not running:
            return FutureStatus.DONE
        elif not finished and not running:
            return FutureStatus.FAILED
        else:
            raise Exception(f"Unknown future status for job")

    def result(self):
        with open(self.target_file, "rb") as f:
            result = pickle.load(f)

        self.clean()

        return result

    def clean(self):
        os.remove(self.input_file)
        os.remove(self.target_file)


class FSCluster:

    def __init__(self,
                 scheduler: Literal["SGE", "HTCONDOR", "SLURM"],
                 num_workers=10,
                 queue=None,
                 project=None,
                 cores_per_worker=12,
                 distributed_mem_per_core=10,
                 resource_spec="",
                 job_extra=("",),
                 walltime="1:00:00",
                 watcher=True,
                 output_dir=Path("/tmp"),
                 ):

        if scheduler == "SGE":
            self.scheduler = SGE(

            )

        elif scheduler == "HTCONDOR":
            self.scheduler = HTCONDOR(
                output_dir=output_dir,
                distributed_cores_per_worker=cores_per_worker,
                distributed_mem_per_core=distributed_mem_per_core,
            )

        elif scheduler == "SLURM":
            self.scheduler = SLURM(
                queue=queue,
                project=project,
                cores=cores_per_worker,
                memory=f"{distributed_mem_per_core * cores_per_worker}GB",
                walltime=walltime,
                processes=1,
            )

        else:
            raise Exception(f"Scheduler: {scheduler} is not recognised!")

    def __call__(self, funcs):
        futures = [self.submit(f) for f in funcs]

        print([f.status(self.scheduler) for f in futures])

        while any(f.status(self.scheduler) == FutureStatus.RUNNING for f in futures):
            running = [f for f in futures if f.status(self.scheduler) == FutureStatus.RUNNING]
            failed = [f for f in futures if f.status(self.scheduler) == FutureStatus.FAILED]
            complete = [f for f in futures if f.status(self.scheduler) == FutureStatus.DONE]

            print(f"\t{len(running)} out of {len(futures)} running. {len(failed)} failed. {len(complete)} completee")
            sleep(1)


        statuses = [f.status(self.scheduler) for f in futures]
        print("###########################")
        print(f"Statuses are: {statuses}")
        print("###########################")


        results = [f.result() for f in futures]
        for result in results:
            print(result)

        return results

    def submit(self, f) -> FSFuture:
        return self.scheduler.submit(f)
