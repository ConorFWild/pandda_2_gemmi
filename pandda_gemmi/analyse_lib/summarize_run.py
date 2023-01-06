import time

from pandda_gemmi.analyse_interface import *
from pandda_gemmi import constants
from pandda_gemmi.logs import (
    save_json_log,
)

def summarize_run(pandda_args, console, pandda_log, time_start):
    time_finish = time.time()
    pandda_log[constants.LOG_TIME] = time_finish - time_start
    # Output json log
    console.start_log_save()
    # with STDOUTManager('Saving json log with detailed information on run...', f'\tDone!'):
    # if pandda_args.debug >= Debug.PRINT_SUMMARIES:
        # printer.pprint(pandda_log)
    save_json_log(
        pandda_log,
        pandda_args.out_dir / constants.PANDDA_LOG_FILE,
    )
    console.summarise_log_save(pandda_args.out_dir / constants.PANDDA_LOG_FILE)
    # print(f"PanDDA ran in: {time_finish - time_start}")
    console.summarise_run(time_finish - time_start)