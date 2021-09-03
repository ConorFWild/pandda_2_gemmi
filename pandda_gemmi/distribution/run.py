import pickle

import fire


def main(path: str):
    with open(path, "rb") as f:
        func = pickle.load(f)

    func()


if __name__ == "__main__":
    fire.Fire(main)
