import pickle

import fire

from pandda_gemmi.autobuild import autobuild_rhofit


def main(
        dataset_pickle_path,
        event_pickle_path,
        pandda_fs_pickle_path
):
    dataset = pickle.load(dataset_pickle_path)
    event = pickle.load(event_pickle_path)
    pandda_fs = pickle.load(pandda_fs_pickle_path)

    autobuild_rhofit(dataset, event, pandda_fs_pickle_path)


if __name__ == "__main__":
    fire.Fire(main)
