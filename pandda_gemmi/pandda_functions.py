import joblib


def process_local_joblib(n_jobs, verbosity, funcs):
    mapper = joblib.Parallel(n_jobs=n_jobs,
                             verbose=verbosity,
                             backend="loky",
                             )

    results = mapper(joblib.delayed(func)() for func in funcs)

    return results