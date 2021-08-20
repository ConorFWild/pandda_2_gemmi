import os
import fire
from pathlib import Path
from functools import partial
import subprocess


def generate_pandda_command(data_dir: str, out_dir: str) -> str:
    command = f"pandda.analyse data_dirs=\'{data_dir}/*\' out_dir=\'{out_dir}\' pdb_style=\'*.pdb\' mtz_style=\'*.mtz\' cpus=12 max_new_datasets=9999"
    return command


def run(command: str):
    print(f"Running command: {command}")
    p = subprocess.Popen(command,
                         shell=True,
                         )
    p.communicate()


def run_pandda(data_dir: str, out_dir: str):
    command = generate_pandda_command(data_dir, out_dir)
    run(command)


def run_panddas(data_dirs: str, out_dirs: str, distributed: bool = True):
    data_dirs = Path(data_dirs)
    out_dirs = Path(out_dirs)

    if not out_dirs.exists():
        os.mkdir(out_dirs)

    commands = [
        generate_pandda_command(
            str(data_dir.resolve()),
            str((out_dirs / data_dir.name).resolve()),
        )
        for data_dir
        in data_dirs.glob("*")]
    for command in commands:
        print(f"\t{command}")

    if distributed:
        num_workers = 10
        queue = None
        project = None
        cores_per_worker = 6
        distributed_mem_per_core = 40
        resource_spec = ""
        walltime = "120:00:00"

        import dask
        from dask.distributed import Client
        from dask_jobqueue import HTCondorCluster

        dask.config.set({'distributed.worker.daemon': False})

        job_extra = [
            (f"GetEnv", "True"),
            ("RequestCpus", f"{cores_per_worker}"),
            ("RequestMemory", f"{cores_per_worker*distributed_mem_per_core}"),
            ("RequestDisk", f"100G"),
        ]
        cluster = HTCondorCluster(
            cores=cores_per_worker,
            memory=f"{distributed_mem_per_core * cores_per_worker}G",
            disk="100G",
            processes=1,
            nanny=False,
            job_extra=job_extra,
            log_directory=str(out_dirs),
        )

        # Scale the cluster up to the number of workers
        cluster.scale(jobs=num_workers)

        # Launch the client
        client = Client(cluster)

        funcs = [
            partial(
                run,
                command,
            )
            for command
            in commands
        ]

        processes = [client.submit(func) for func in funcs]
        results = client.gather(processes)


    else:
        for command in commands:
            run(command)


if __name__ == "__main__":
    fire.Fire(run_panddas)
