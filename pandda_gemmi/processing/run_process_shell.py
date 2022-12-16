import pickle

import fire


def path_to_obj(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    return obj


def run(func_path, output_path, *arg_paths, **kwarg_paths):
    func = path_to_obj(func_path)
    args = [path_to_obj(arg_path) for arg_path in arg_paths]
    kwargs = {kwrd: path_to_obj(kwarg_path) for kwrd, kwarg_path in kwarg_paths.items()}

    result = func(*args, **kwargs)

    with open(output_path, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    fire.Fire(run)
