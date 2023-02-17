import pickle

from pandda_gemmi.common import update_log
from pandda_gemmi.analyse_interface import *
from pandda_gemmi import constants
from pandda_gemmi.edalignment import GetAlignments, Alignments

def get_alignments(pandda_args, console, pandda_log, pandda_fs_model: PanDDAFSModelInterface, datasets, reference):
    console.start_alignments()

    alignment_files_exist = [alignment_file.path.exists() for dtag, alignment_file in pandda_fs_model.alignment_files.items()]
    if (len(alignment_files_exist) > 0) & all(alignment_files_exist):
        alignments = Alignments({dtag: alignment_file.load() for dtag, alignment_file in pandda_fs_model.alignment_files.items()})

    else:
        # with STDOUTManager('Getting local alignments of the electron density to the reference...', f'\tDone!'):
        alignments: AlignmentsInterface = GetAlignments()(
            reference,
            datasets,
        )

        if pandda_args.debug >= Debug.AVERAGE_MAPS:
            with open(pandda_fs_model.pandda_dir / "alignments.pickle", "wb") as f:
                pickle.dump(alignments, f)

        update_log(pandda_log, pandda_args.out_dir / constants.PANDDA_LOG_FILE)

        console.summarise_local_alignment()

        for dtag, alignment in alignments.alignments.items():
            pandda_fs_model.alignment_files[dtag].save(alignment)

    return alignments