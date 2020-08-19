from __future__ import annotations

import dataclasses

from pathlib import Path


@dataclasses.dataclass()
class Log:
    out_file: Path

    @staticmethod
    def from_dir(out_dir: Path) -> Log:
        pass
