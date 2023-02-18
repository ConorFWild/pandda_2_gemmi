import pickle

from pandda_gemmi.analyse_interface import *
from pandda_gemmi.dataset import GetReferenceDataset


def get_reference(pandda_args,
                  console, pandda_log,
                  pandda_fs_model: PanDDAFSModelInterface, datasets_wilson, datasets_statistics):
    console.start_reference_selection()
    if pandda_fs_model.reference_file.path.exists():
        reference = pandda_fs_model.reference_file.load()

    else:
        # Select refernce
        reference: ReferenceInterface = GetReferenceDataset()(
            datasets_wilson,
            datasets_statistics,
        )
        pandda_log["Reference Dtag"] = str(reference.dtag)
        console.summarise_reference(reference)
        pandda_fs_model.reference_file.save(reference)
        if pandda_args.debug >= Debug.PRINT_SUMMARIES:
            print(reference.dtag)

        if pandda_args.debug >= Debug.AVERAGE_MAPS:
            with open(pandda_fs_model.pandda_dir / "reference.pickle", "wb") as f:
                pickle.dump(reference, f)

    return reference
