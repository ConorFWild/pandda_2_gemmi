import pickle

import fire

from pandda_gemmi.autobuild import autobuild_rhofit


def main(
        dataset_pickle_path,
        event_pickle_path,
        pandda_fs_pickle_path
):
    with open(dataset_pickle_path, "rb") as f:
        dataset = pickle.load(f)
    with open(event_pickle_path, "rb") as f:
        event = pickle.load(f)
    with open(pandda_fs_pickle_path, "rb") as f:
        pandda_fs = pickle.load(f)

    print("####################################################")
    print("Dataset info:")
    print(pandda_fs.processed_datasets[event.event_id.dtag])
    print("####################################################")

    autobuild_rhofit(dataset, event, pandda_fs, cif_strategy="elbow")


if __name__ == "__main__":
    fire.Fire(main)
