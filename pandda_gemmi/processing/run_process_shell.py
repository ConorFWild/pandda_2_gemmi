import pickle

import fire


def path_to_obj(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    return obj


def run(func_path, output_path, *arg_paths, **kwarg_paths):
    print(f"Processed arguments...")
    print(f"Function path is: {func_path}")
    print(f"Argument paths are: ")
    for arg_path in arg_paths:
        print(f"\t{arg_path}")
    print(f"Key word argument paths are:")
    for kwrd, kwarg_path in kwarg_paths.items():
        print(f"\t{kwrd} : {kwarg_path}")

    func = path_to_obj(func_path)
    args = []
    for arg_path in arg_paths:
        args.append(path_to_obj(arg_path))
    kwargs = {}
    for kwrd, kwarg_path in kwarg_paths.items():
        kwargs[kwrd] = path_to_obj(kwarg_path)

    print(f"Loaded input. Running function!")

    result = func(*args, **kwargs)

    print(f"Ran function: Pickeling results...")

    with open(output_path, "wb") as f:
        pickle.dump(result, f)

    print(f"Pickeled results! Returning!")


if __name__ == "__main__":
    print(f"Running qsub'd process...")
    fire.Fire(run)
