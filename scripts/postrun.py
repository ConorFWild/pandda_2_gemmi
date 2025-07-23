import time

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ImportError:
    print('No sklearn-express available!')

from pandda_gemmi.interfaces import *
from pandda_gemmi.processor import ProcessLocalRay
from pandda_gemmi.pandda_logging import PanDDAConsole
from pandda_gemmi.pandda.prerun import prerun
from pandda_gemmi.pandda.get_scoring_models import get_scoring_models
from pandda_gemmi.args import PanDDAArgs
from pandda_gemmi.pandda.process_dataset import read_dataset
from pandda_gemmi.pandda.postrun import postrun

if __name__ == '__main__':
    # Parse Command Line Arguments
    args = PanDDAArgs.from_command_line()

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

    # Load known events and autobuilds
    if args.debug:
        print('Processed Datasets')
        print(fs.output.processed_datasets)
    pandda_events = {}
    autobuilds = {}
    for _dtag in datasets:
        new_events, new_autobuilds = read_dataset(fs, _dtag)
        pandda_events.update(new_events)
        autobuilds.update(new_autobuilds)

    # Process the PanDDA
    time_pandda_begin = time.time()
    postrun(
        args,
        fs,
        console,
        datasets,
        pandda_events,
        autobuilds,
        datasets_to_process,
        event_score_quantiles,
        time_pandda_begin,
    )
