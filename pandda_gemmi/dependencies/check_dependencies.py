import shutil
import pprint
from pandda_gemmi.args import PanDDAArgs


def check_dependencies(args: PanDDAArgs):
    printer = pprint.PrettyPrinter(indent=4)
    status = {}

    # check autobuilding args
    if args.autobuild:
        # Check autobuild_strategy
        if args.autobuild_strategy == 'rhofit':
            if shutil.which("ana_pdbmaps"):
                status['ana_pdbmaps'] = 'Found!'
            else:
                status['ana_pdbmaps'] = 'Not Found!'

            if shutil.which("rhofit"):
                status['rhofit'] = 'Found!'
            else:
                status['rhofit'] = 'Not Found!'

            if shutil.which("pandda_rhofit.sh"):
                status['pandda_rhofit.sh'] = 'Found!'
            else:
                status['pandda_rhofit.sh'] = 'Not Found!'

        if args.cif_strategy == 'elbow':
            if shutil.which("phenix.elbow"):
                status['phenix.elbow'] = 'Found!'
            else:
                status['phenix.elbow'] = 'Not Found!'
        if args.cif_strategy == 'grade':
            if shutil.which("grade"):
                status['grade'] = 'Found!'
            else:
                status['grade'] = 'Not Found!'
        if args.cif_strategy == 'grade2':
            if shutil.which("grade2"):
                status['grade2'] = 'Found!'
            else:
                status['grade2'] = 'Not Found!'

    # Check ranking args
    if args.rank_method == 'autobuild':
        if args.autobuild:
            status['--autobuild'] = 'Found!'
        else:
            status['--autobuild'] = 'Not Found!'

    for dependency, state in status.items():
        if state != 'Found!':
            print('An exception was encountered in resolving the dependencies. Please check the following status to '
                  'ensure that required options are enabled and that required programs are in your path.')
            printer.pprint(status)
            print('PanDDA will now exit!')
            exit()

    return status
