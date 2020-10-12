from __future__ import annotations

import typing
import dataclasses

from pprint import PrettyPrinter
printer = PrettyPrinter(indent=1)
from pathlib import Path

from pandda_gemmi.pandda_types import *

@dataclasses.dataclass()
class Log:
    out_file: Path

    @staticmethod
    def from_dir(out_dir: Path) -> Log:
        pass


class XmapLogs:
    xmap_logs: typing.Dict[str, str]

    @staticmethod
    def from_xmaps(xmaps: Xmaps):
        logs = {}
        for dtag in xmaps:
            logs[dtag] = {}


class ModelLogs:

    @staticmethod
    def from_model(model: Model):
        logs_str = """
        Model Summary
            Per model stds
                {model_stds}
        """
        logs_str.format(model_stds=printer.pformat(model.stds))

        return logs_str