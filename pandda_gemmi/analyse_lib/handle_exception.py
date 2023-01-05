import traceback

from pandda_gemmi import constants
from pandda_gemmi.logs import (
    save_json_log,
)


def handle_exception(pandda_args, console, e, pandda_log,):
    console.print_exception()
    console.save(pandda_args.out_dir / constants.PANDDA_TEXT_LOG_FILE)

    pandda_log[constants.LOG_TRACE] = traceback.format_exc()
    pandda_log[constants.LOG_EXCEPTION] = str(e)

    print(f"Saving PanDDA log to: {pandda_args.out_dir / constants.PANDDA_LOG_FILE}")

    save_json_log(
        pandda_log,
        pandda_args.out_dir / constants.PANDDA_LOG_FILE,
    )