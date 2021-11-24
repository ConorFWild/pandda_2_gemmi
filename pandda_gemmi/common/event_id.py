import dataclasses

from pandda_gemmi.common.dtag import Dtag



@dataclasses.dataclass()
class EventIDX:
    event_idx: int

    def __hash__(self):
        return hash(self.event_idx)


@dataclasses.dataclass()
class ClusterID:
    dtag: Dtag
    number: EventIDX

    def __hash__(self):
        return hash((self.dtag, self.number))

@dataclasses.dataclass()
class EventID:
    dtag: Dtag
    event_idx: EventIDX

    def __hash__(self):
        return hash((self.dtag, self.event_idx))
