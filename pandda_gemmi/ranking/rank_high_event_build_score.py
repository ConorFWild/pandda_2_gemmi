from ..interfaces import *


class RankHighEventBuildScore:
    def __call__(
            self,
            events: Dict[Tuple[str, int], EventInterface],
            autobuilds: Dict[Tuple[str, int], Dict[str, AutobuildInterface]],
    ):
        sorted_event_ids = []
        for event_id in sorted(events, key=lambda _event_id: events[_event_id].build.score): #, reverse=True):
            sorted_event_ids.append(event_id)

        return sorted_event_ids

