import dataclasses
import shutil
import pprint
from pandda_gemmi.args import PanDDAArgs
from pandda_gemmi.analyse_interface import *

{
    (("autobuild", True), ("autobuild_strategy", "Rhofit")): ["rhofit", "ana_pdbmaps"],
    (("rank_method", "autobuild"),): ["--autobuild", ]
}


@dataclasses.dataclass()
class Option:
    name: str
    option_value: any
    actual_value: any


@dataclasses.dataclass()
class Dependency:
    name: str
    status: bool


@dataclasses.dataclass()
class OptionSet:
    options: List[Option]
    dependencies: List[Dependency]


def check_dependencies(args: PanDDAArgs):
    printer = pprint.PrettyPrinter(indent=4)
    status = {}

    # Dependencies
    dep_ana_pdbmaps = Dependency("ana_pdbmaps", bool(shutil.which("ana_pdbmaps")))
    dep_rhofit = Dependency("rhofit", bool(shutil.which("rhofit")))
    dep_autobuild = Dependency("autobuild", bool(args.autobuild))
    dep_pandda_rhofit = Dependency("pandda_rhofit.sh", bool(shutil.which("pandda_rhofit.sh")))

    # Options
    opt_autobuild = Option("autobuild", True, args.autobuild)
    opt_autobuild_ranking = Option("rank_method", "autobuild", args.rank_method)

    # Option sets
    option_sets = [
        OptionSet(
            [opt_autobuild_ranking, ],
            [dep_autobuild, ]
        ),
        OptionSet(
            [opt_autobuild, ],
            [dep_rhofit, dep_ana_pdbmaps, dep_ana_pdbmaps]
        )
    ]

    # Check
    failed_option_sets = []
    for option_set in option_sets:
        # Check if all the options appropriate to induce the dependencies
        if all([option.option_value == option.actual_value for option in option_set.options]):
            if not all([dep for dep in option_set.dependencies]):
                failed_option_sets.append(option_set)

    return failed_option_sets

    # # check autobuilding args
    # if args.autobuild:
    #     # Check autobuild_strategy
    #     if args.autobuild_strategy == 'rhofit':
    #         if shutil.which("ana_pdbmaps"):
    #             status['ana_pdbmaps'] = 'Found!'
    #         else:
    #             status['ana_pdbmaps'] = 'Not Found!'
    #
    #         if shutil.which("rhofit"):
    #             status['rhofit'] = 'Found!'
    #         else:
    #             status['rhofit'] = 'Not Found!'
    #
    #         if shutil.which("pandda_rhofit.sh"):
    #             status['pandda_rhofit.sh'] = 'Found!'
    #         else:
    #             status['pandda_rhofit.sh'] = 'Not Found!'
    #
    #     if args.cif_strategy == 'elbow':
    #         if shutil.which("phenix.elbow"):
    #             status['phenix.elbow'] = 'Found!'
    #         else:
    #             status['phenix.elbow'] = 'Not Found!'
    #     if args.cif_strategy == 'grade':
    #         if shutil.which("grade"):
    #             status['grade'] = 'Found!'
    #         else:
    #             status['grade'] = 'Not Found!'
    #     if args.cif_strategy == 'grade2':
    #         if shutil.which("grade2"):
    #             status['grade2'] = 'Found!'
    #         else:
    #             status['grade2'] = 'Not Found!'
    #
    # # Check ranking args
    # if args.rank_method == 'autobuild':
    #     if args.autobuild:
    #         status['--autobuild'] = 'Found!'
    #     else:
    #         status['--autobuild'] = 'Not Found!'

    # for dependency, state in status.items():
    #     if state != 'Found!':
    #         print('An exception was encountered in resolving the dependencies. Please check the following status to '
    #               'ensure that required options are enabled and that required programs are in your path.')
    #         printer.pprint(status)
    #         print('PanDDA will now exit!')
    #         exit()

    return status
