import re
import pickle

import fire

from pandda_gemmi.analyse_interface import *
from pandda_gemmi.autobuild import autobuild_rhofit, GetAutobuildResultRhofit


def unpickle(path: Path):
    with open(path, "rb") as f:
        ob = pickle.load(f)

    return ob


class DtagAutobuildTestDataDir:
    def __init__(self, path: Path):
        dtag_pickle_path = path / "test_dtag.pickle"
        events_pickle_path = path / "events.pickle"
        dataset_pickle_path = path / "dataset_processed_dataset.pickle"
        dataset_xmap_pickle_path = path / "dataset_xmap.pickle"

        self.dtag = unpickle(dtag_pickle_path)
        self.dataset = unpickle(dataset_pickle_path)
        self.events = unpickle(events_pickle_path)
        self.dataset_xmap = unpickle(dataset_xmap_pickle_path)

        self.models: Dict[int, ModelInterface] = {}
        for model_path in path.glob("model_*.pickle"):
            match = re.match(
                "model_([0-9]+).pickle",
                str(model_path)
            )
            if match:
                model_num = int(
                    match.groups()[0]
                )
                model: ModelInterface = unpickle(model_path)
                self.models[model_num] = model


def _main(
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


class ModelResult:
    ...

class DatasetModelsScoreResult:
    model_results: Dict[int, ModelResult]
    selected_model: int

def score_dataset_models(
            pandda_fs_model: PanDDAFSModelInterface,
            alignments: AlignmentsInterface,
            grid: GridInterface,
            reference: ReferenceInterface,
            dtag: DtagInterface,
            dataset_dir: DtagAutobuildTestDataDir,
        ) -> DatasetModelsScoreResult:
    ...


def main(autobuild_test_data_dir: str):
    autobuild_test_data_dir = Path(autobuild_test_data_dir)
    print("Loading data...")
    pandda_fs_model_pickle_path = autobuild_test_data_dir / "pandda_fs_model.pickle"
    alignments_pickle_path = autobuild_test_data_dir / "alignments.pickle"
    grid_pickle_path = autobuild_test_data_dir / "grid.pickle"
    reference_pickle_path = autobuild_test_data_dir / "reference.pickle"

    pandda_fs_model: PanDDAFSModelInterface = unpickle(pandda_fs_model_pickle_path)
    alignments: AlignmentsInterface = unpickle(alignments_pickle_path)
    grid: GridInterface = unpickle(grid_pickle_path)
    reference: ReferenceInterface = unpickle(reference_pickle_path)

    dataset_dirs: Dict[DtagInterface, DtagAutobuildTestDataDir] = {}
    for dataset_dir in autobuild_test_data_dir.glob("*"):
        if not dataset_dir.is_dir():
            continue

        dtag_test_data_dir = DtagAutobuildTestDataDir(dataset_dir)

        dataset_dirs[dtag_test_data_dir.dtag] = dtag_test_data_dir

    print("Running tests...")

    for dtag, dataset_dir in dataset_dirs.items():
        print(f"\tTesting dtag: {dtag}")
        dataset_models_score_result: DatasetModelsScoreResult = score_dataset_models(
            pandda_fs_model,
            alignments,
            grid,
            reference,
            dtag,
            dataset_dir,
        )


if __name__ == "__main__":
    fire.Fire(main)
