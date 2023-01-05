import os
import dataclasses

from pandda_gemmi.analyse_interface import *


@dataclasses.dataclass()
class AutobuildResult:
    status: bool
    paths: List[str]
    scores: Dict[str, float]
    cif_path: str
    selected_fragment_path: Optional[str]
    command: str

    def log(self):
        return self.command


def autobuild_inbuilt(
        event_id: EventIDInterface, dataset: DatasetInterface,
        event: EventInterface,
        pandda_fs: PanDDAFSModelInterface,
        event_scoring_result: EventScoringResultInterface,
        debug: Debug,
):
    # Select highest scoring build
    structure = event_scoring_result.get_selected_structure()

    # Make autobuild dir if it is not there
    autobuild_dir = pandda_fs.processed_datasets.processed_datasets[event_id.dtag].path / "autobuilds"
    if not autobuild_dir:
        os.mkdir(autobuild_dir)

    # Save autobuild to dir
    if structure:
        structure_path = autobuild_dir / f"{event_id.dtag.dtag}_{event_id.event_idx.event_idx}.pdb"
        structure.write_pdb(str(structure_path))

        # Construct result
        return AutobuildResult(
            True,
            [str(structure_path), ],
            {event_id: event_scoring_result.get_selected_structure_score(), },
            "",
            str(structure_path),
            ""
        )

    else:
        return AutobuildResult(
            False,
            [],
            {},
            "",
            "",
            ""
        )


class GetAutobuildResultInbuilt(GetAutobuildResultInterface):
    tag = "inbuilt"

    def __call__(self,
                 event_id: EventIDInterface,
                 dataset: DatasetInterface,
                 event: EventInterface,
                 pandda_fs: PanDDAFSModelInterface,
                 event_scoring_result: EventScoringResultInterface,
                 debug: Debug) -> AutobuildResultInterface:
        return autobuild_inbuilt(event_id, dataset, event, pandda_fs, event_scoring_result, debug)
