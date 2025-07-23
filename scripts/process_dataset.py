import time

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ImportError:
    print('No sklearn-express available!')

from pandda_gemmi.interfaces import *
from pandda_gemmi.dataset import StructureArray
from pandda_gemmi.processor import ProcessLocalRay
from pandda_gemmi.pandda_logging import PanDDAConsole
from pandda_gemmi.pandda.prerun import prerun
from pandda_gemmi.pandda.get_scoring_models import get_scoring_models
from pandda_gemmi.args import PanDDAProcessDatasetArgs
from pandda_gemmi.pandda.process_model import ProcessModel
from pandda_gemmi.pandda.process_dataset import process_dataset

if __name__ == '__main__':
    # Parse Command Line Arguments
    args = PanDDAProcessDatasetArgs.from_command_line()

    # Create the console to print output throughout the programs run
    console = PanDDAConsole()

    # Get the processor to handle the dispatch of functions to multiple cores and the cache of parallel
    # processed objects
    # TODO: uses ray not mulyiprocessing_spawn
    console.start_initialise_multiprocessor()
    processor: ProcessorInterface = ProcessLocalRay(args.local_cpus)
    console.print_initialized_local_processor(args)

    # Do the prerun
    fs, datasets, datasets_to_process = prerun(args, console, processor)

    # Get the event and build scores
    score_event, score_build, event_model_config, event_score_quantiles = get_scoring_models(args)
    score_build_ref = processor.put(score_build)

    # Create processor references to datasets and structure arrays
    dataset_refs = {_dtag: processor.put(datasets[_dtag]) for _dtag in datasets}
    structure_array_refs = {_dtag: processor.put(StructureArray.from_structure(datasets[_dtag].structure)) for _dtag in
                            datasets}

    # Get the method for processing the statistical models
    process_model = ProcessModel(
        minimum_event_score=event_model_config['minimum_event_score'],
        use_ligand_data=args.use_ligand_data,
        debug=args.debug,
    )

    # Process the PanDDA
    j = 0
    if args.dtag in datasets_to_process:
        j = [_dtag for _dtag in datasets_to_process].index(args.dtag)
    else:
        exit()
    time_begin_process_datasets = time.time()
    process_dataset(
        args.dtag,
        args,
        fs,
        datasets,
        console,
        j,
        datasets_to_process,
        time_begin_process_datasets,
        process_model,
        score_event,
        processor,
        dataset_refs,
        structure_array_refs,
        score_build_ref
    )
