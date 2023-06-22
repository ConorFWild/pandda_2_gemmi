from ..interfaces import *


class RankHighBuildScore:
    def __call__(
            self,
            events: Dict[Tuple[str, int], EventInterface],
            autobuilds: Dict[Tuple[str, int], Dict[str, AutobuildInterface]],
    ):
        highest_event_build_scores = {}
        for event_id in events:
            if not event_id in autobuilds[event_id]:
                highest_event_build_scores[event_id] = 0.0
                continue

            event_autobuilds = autobuilds[event_id]
            ligand_build_scores = []
            for ligand_key, autobuild_result in event_autobuilds.items():
                if not autobuild_result.log_result_dict:
                    continue
                build_scores = []
                for build_path, score in autobuild_result.log_result_dict.items():
                    build_scores.append(score)

                if len(build_scores) == 0:
                    continue

                ligand_build_scores.append(max(build_scores))

            if len(ligand_build_scores) == 0:
                highest_event_build_scores[event_id] = 0.0
                continue

            highest_event_build_scores[event_id] = max(ligand_build_scores)



        sorted_event_ids = []
        for event_id in sorted(
                highest_event_build_scores,
                key=lambda _event_id: highest_event_build_scores[_event_id],
                reverse=True,
        ):
            sorted_event_ids.append(event_id)
            print(f"\t{event_id[0]} : {event_id[1]} : {highest_event_build_scores[event_id]}")

        return sorted_event_ids
