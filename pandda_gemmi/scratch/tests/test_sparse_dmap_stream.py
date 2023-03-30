import fire

from typing import Dict, List
import time
from pathlib import Path

from pandda_gemmi.scratch.interfaces import *
from pandda_gemmi.scratch.fs import PanDDAFS
from pandda_gemmi.scratch.dataset import XRayDataset
from pandda_gemmi.scratch.dmaps import DMap, SparseDMap, SparseDMapStream, TruncateReflections, SmoothReflections
from pandda_gemmi.scratch.alignment import Alignment, DFrame


def test_sparse_dmap_stream(data_dir, out_dir):
    print(f"Data dir is {data_dir} and output dir is {out_dir}")

    # Parse the FS
    print(f"##### Loading filesystem #####")
    fs: PanDDAFSInterface = PanDDAFS(Path(data_dir), Path(out_dir))

    # Get the datasets
    print(f"##### Getting datasets #####")
    datasets: Dict[str, DatasetInterface] = {
        dataset_dir.dtag: XRayDataset(
            dataset_dir.input_pdb_file,
            dataset_dir.input_mtz_file,
            dataset_dir.input_ligands,
        )
        for dataset_dir
        in fs.input.dataset_dirs.values()
    }
    print(f"Got {len(datasets)} datasets")

    # Get the test dataset
    print(f"##### Getting test dataset #####")
    dtag = list(datasets.keys())[0]
    dataset = datasets[dtag]
    print(f"Test dataset is {dtag}")

    # Get the alignments
    print(f"##### Getting alignments #####")
    begin_align = time.time()
    alignments: Dict[str, Alignment] = {_dtag: Alignment(datasets[_dtag], dataset) for _dtag in datasets}
    finish_align = time.time()
    print(f"Got {len(alignments)} alignments in {round(finish_align-begin_align, 1)}")

    # Get the reference frame
    print(f"##### Getting reference frame #####")
    begin_get_frame = time.time()
    reference_frame: DFrame = DFrame(dataset)
    finish_get_frame = time.time()
    print(f"Got reference frame in {round(finish_get_frame-begin_get_frame, 1)}")

    # Get the dmaps
    print(f"##### Getting sparse dmap loader #####")
    dmaps: SparseDMapStream = SparseDMapStream(
        datasets,
        reference_frame,
        alignments,
        [
            TruncateReflections(
                datasets,
                dataset.reflections.resolution(),
            ),
            SmoothReflections(dataset)
        ],
    )

    # Load
    print(f"##### Getting sparse xmaps #####")
    time_begin = time.time()
    dmaps_sparse: Dict[str, SparseDMap] = {
        dtag: dmaps.load(dtag)
        for dtag
        in datasets
    }
    time_finish = time.time()
    print(f"Got sparse xmaps in {round(time_finish-time_begin, 1)}")



if __name__ == "__main__":
    fire.Fire(test_sparse_dmap_stream)