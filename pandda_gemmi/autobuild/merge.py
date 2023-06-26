import gemmi

from pandda_gemmi import constants
from ..interfaces import *


def get_pdb(pdb_file: Path):
    structure: gemmi.Structure = gemmi.read_structure(str(pdb_file))
    return structure


def merge_build(dataset, selected_build_path, path):
    receptor = get_pdb(dataset.structure.path)
    ligand = get_pdb(selected_build_path)

    for receptor_model in receptor:
        for receptor_chain in receptor_model:

            seqid_nums = []
            for receptor_res in receptor_chain:
                num = receptor_res.seqid.num
                seqid_nums.append(num)

            if len(seqid_nums) == 0:
                min_ligand_seqid = 1
            else:
                min_ligand_seqid = max(seqid_nums) + 1

            for model in ligand:
                for chain in model:
                    for residue in chain:
                        residue.seqid.num = min_ligand_seqid

                        receptor_chain.add_residue(residue, pos=-1)

            break
        break

    receptor.write_minimal_pdb(str(path))


def merge_autobuilds(
        datasets,
        events,
        autobuilds: Dict[Tuple[str, int], Dict[str, AutobuildInterface]],
        fs: PanDDAFSInterface,
        build_selection_method,
):
    all_dtags = list(set([event_id[0] for event_id in autobuilds]))

    merged_build_scores = {}
    for dtag in all_dtags:
        dataset = datasets[dtag]
        dtag_events = [event_id for event_id in events if event_id[0] == dtag]
        dtag_autobuilds = [[event_id, autobuilds[event_id]] for event_id in dtag_events]
        # print(dtag_autobuilds)

        #
        all_autobuilds = {}
        for event_id, event_autobuilds in dtag_autobuilds:
            for ligand_key in event_autobuilds.keys():
                autobuild_result = event_autobuilds[ligand_key]

                if autobuild_result.log_result_dict:
                    print(f"\t\t\t{autobuild_result.log_result_dict}")

                    for build_path, score in autobuild_result.log_result_dict.items():
                        all_autobuilds[build_path] = [score, event_id]

        #
        if len(all_autobuilds) == 0:
            print(f"\t\tNo autobuilds generated for dataset: {dtag}")
            continue

        #
        selected_build_path = build_selection_method(
            all_autobuilds,
            {_event_id: events[_event_id] for _event_id in dtag_events}
        )
        print(f"\tSlected build path: {selected_build_path}")
        model_building_dir = fs.output.processed_datasets[dtag] / constants.PANDDA_MODELLED_STRUCTURES_DIR
        merge_build(
            dataset,
            selected_build_path,
            model_building_dir / constants.PANDDA_EVENT_MODEL.format(dtag),
        )
        merged_build_scores[dtag] = all_autobuilds[selected_build_path][0]

    return merged_build_scores


class MergeHighestRSCC:
    def __call__(self, autobuilds: Dict[str, float], dtag_events: Dict[Tuple[str, int], EventInterface]):
        return max(autobuilds, key=lambda _path: autobuilds[_path])

class MergeHighestBuildScore:
    def __call__(self, autobuilds: Dict[str, float], dtag_events: Dict[Tuple[str, int], EventInterface]):
        return max(autobuilds, key=lambda _path: -autobuilds[_path])

class MergeHighestBuildAndEventScore:
    def __call__(
            self,
            autobuilds: Dict[str, Tuple[float, Tuple[str,int]]],
            dtag_events: Dict[Tuple[str, int], EventInterface],
    ):
        highest_scoring_event = max(
            dtag_events,
            key=lambda _event_id: dtag_events[_event_id].score,
        )
        highest_scoring_event_autobuilds = {
            _path: score_and_event_id[0]
            for _path, score_and_event_id
            in autobuilds.items()
            if highest_scoring_event[1] == score_and_event_id[1][1]
        }
        return max(
            highest_scoring_event_autobuilds,
            key=lambda _path: -highest_scoring_event_autobuilds[_path],
        )

