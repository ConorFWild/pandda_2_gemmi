try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ImportError:
    print('No sklearn-express available!')

from pandda_gemmi.interfaces import *
from pandda_gemmi.fs import PanDDAFS
from pandda_gemmi.dataset import XRayDataset
from pandda_gemmi.comparators import (
    FilterRFree,
    FilterRange,
    FilterExcludeFromAnalysis,
    FilterOnlyDatasets,
    FilterResolutionLowerLimit,
    FilterNoLigandData
)
from pandda_gemmi import serialize


class GetDatasetsToProcess:
    def __init__(self, filters=None):
        self.filters = filters

    def __call__(self,
                 # *args, **kwargs
                 datasets: Dict[str, DatasetInterface],
                 fs: PanDDAFSInterface
                 ):
        datasets_not_to_process = {}
        remaining_datasets = {_dtag: _dataset for _dtag, _dataset in datasets.items()}
        for _filter in self.filters:
            remaining_datasets = _filter(remaining_datasets)
            for dtag in datasets:
                if (dtag not in datasets_not_to_process) and (dtag not in remaining_datasets):
                    datasets_not_to_process[dtag] = _filter.description()

        sorted_remaining_datasets = {
            _k: remaining_datasets[_k]
            for _k
            in sorted(remaining_datasets)
        }
        sorted_datasets_not_to_process = {
            _k: datasets_not_to_process[_k]
            for _k
            in sorted(datasets_not_to_process)
        }
        return sorted_remaining_datasets, sorted_datasets_not_to_process


def prerun(args, console, processor):
    # Get the model of the input and output of the program on the file systems
    console.start_fs_model()
    fs: PanDDAFSInterface = PanDDAFS(Path(args.data_dirs), Path(args.out_dir), args.pdb_regex, args.mtz_regex)
    console.summarise_fs_model(fs)

    # Load the structures and reflections from the datasets found in the file system, and create references to these
    # dataset objects and the arrays of their structures in the multiprocessing cache
    console.start_load_datasets()
    datasets: Dict[str, DatasetInterface] = {
        dataset_dir.dtag: XRayDataset.from_paths(
            dataset_dir.input_pdb_file,
            dataset_dir.input_mtz_file,
            dataset_dir.input_ligands,
            name=dataset_dir.dtag
        )
        for dataset_dir
        in fs.input.dataset_dirs.values()
    }

    # Summarise the datasets loaded from the file system and serialize the information on the input into a human
    # readable yaml file
    console.summarise_datasets(datasets, fs)
    serialize.input_data(
        fs, datasets, fs.output.path / "input.yaml"
    )

    # Get the datasets to process
    dataset_filters = [
        FilterRFree(args.max_rfree),
        FilterResolutionLowerLimit(args.high_res_lower_limit),
        FilterRange(args.dataset_range),
        FilterExcludeFromAnalysis(args.exclude_from_z_map_analysis),
        FilterOnlyDatasets(args.only_datasets)
    ]
    if args.use_ligand_data:
        dataset_filters.append(FilterNoLigandData())

    datasets_to_process, datasets_not_to_process = GetDatasetsToProcess(dataset_filters)(datasets, fs)
    console.summarize_datasets_to_process(datasets_to_process, datasets_not_to_process)

    return
