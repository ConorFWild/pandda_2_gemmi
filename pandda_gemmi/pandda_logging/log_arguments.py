from dataclasses import fields, asdict

from pandda_gemmi.args import PanDDAArgs


def log_arguments(pandda_args: PanDDAArgs):
    # key = 'Initial arguments to PanDDA are'
    log = {}

    args_dict = asdict(pandda_args)
    for field in fields(pandda_args):
        log[field.name] = str(args_dict[field.name])

    return log
