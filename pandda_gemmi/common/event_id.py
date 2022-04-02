import dataclasses
from pandda_gemmi.analyse_interface import ClusterIDInterface, DtagInterface, EventIDInterface, EventIDXInterface

from pandda_gemmi.common.dtag import Dtag


@dataclasses.dataclass()
class EventIDX(EventIDXInterface):
    event_idx: int

    def __hash__(self):
        return hash(self.event_idx)

    def __int__(self) -> int:
        return int(self.event_idx)


@dataclasses.dataclass()
class ClusterID(ClusterIDInterface):
    dtag: DtagInterface
    number: int

    def __hash__(self):
        return hash((self.dtag, self.number))

    def __int__(self) -> int:
        return int(self.number)


@dataclasses.dataclass()
class EventID(EventIDInterface):
    dtag: DtagInterface
    event_idx: EventIDXInterface

    def __hash__(self):
        return hash((self.dtag, self.event_idx))
