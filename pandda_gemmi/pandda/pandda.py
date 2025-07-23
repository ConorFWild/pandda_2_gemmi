import time

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ImportError:
    print('No sklearn-express available!')

from pandda_gemmi.interfaces import *
from pandda_gemmi.args import PanDDAArgs
from pandda_gemmi.dataset import StructureArray
from pandda_gemmi.processor import ProcessLocalRay
from pandda_gemmi.pandda_logging import PanDDAConsole
from pandda_gemmi.pandda.process_dataset import process_dataset
from pandda_gemmi.pandda.process_model import ProcessModel
from pandda_gemmi.pandda.prerun import prerun
from pandda_gemmi.pandda.get_scoring_models import get_scoring_models
from pandda_gemmi.pandda.postrun import postrun


def pandda(args: PanDDAArgs):
    # Record time at which PanDDA processing begins
    time_pandda_begin = time.time()

    # Create the console to print output throughout the programs run
    console = PanDDAConsole()

    # Print the PanDDA initialization message and the command line arguments
    console.start_pandda()
    console.start_parse_command_line_args()
    console.summarise_arguments(args)

    # Get the processor to handle the dispatch of functions to multiple cores and the cache of parallel
    # processed objects
    # TODO: uses ray not mulyiprocessing_spawn
    console.start_initialise_multiprocessor()
    processor: ProcessorInterface = ProcessLocalRay(args.local_cpus)
    console.print_initialized_local_processor(args)

    fs, datasets, datasets_to_process = prerun(args, console, processor)

    # Get the event and build scores
    score_event, score_build, event_model_config, event_score_quantiles = get_scoring_models(args)
    score_build_ref = processor.put(score_build)

    # Get the method for processing the statistical models
    process_model = ProcessModel(minimum_event_score=event_model_config['minimum_event_score'],
                                 use_ligand_data=args.use_ligand_data, debug=args.debug)

    # Create processor references to datasets and structure arrays
    dataset_refs = {_dtag: processor.put(datasets[_dtag]) for _dtag in datasets}
    structure_array_refs = {_dtag: processor.put(StructureArray.from_structure(datasets[_dtag].structure)) for _dtag in
                            datasets}

    # Process each dataset by identifying potential comparator datasets, constructing proposed statistical models,
    # calculating alignments of comparator datasets, locally aligning electron density, filtering statistical models
    # to the plausible set, evaluating those models for events, selecting a model to take forward based on those events
    # and outputing event maps, z maps and mean maps for that model
    pandda_events = {}
    autobuilds = {}

    time_begin_process_datasets = time.time()
    console.start_process_shells()
    for j, dtag in enumerate(datasets_to_process):
        new_pandda_events, new_autobuilds = process_dataset(
            dtag,
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
        pandda_events.update(new_pandda_events)
        autobuilds.update(new_autobuilds)

    time_finish_process_datasets = time.time()
    postrun(
        args,
        fs,
        console,
        datasets,
        pandda_events,
        autobuilds,
        datasets_to_process,
        event_score_quantiles,
        time_pandda_begin
    )
