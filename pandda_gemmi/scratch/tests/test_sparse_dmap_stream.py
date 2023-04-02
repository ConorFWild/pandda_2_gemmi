import fire

from typing import Dict, List
import time
from pathlib import Path

import gemmi
import numpy as np

from pandda_gemmi.scratch.interfaces import *
from pandda_gemmi.scratch.fs import PanDDAFS
from pandda_gemmi.scratch.dataset import XRayDataset
from pandda_gemmi.scratch.dmaps import DMap, SparseDMap, SparseDMapStream, TruncateReflections, SmoothReflections
from pandda_gemmi.scratch.alignment import Alignment, DFrame


def save_dmap(dmap, path):
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = dmap
    # if p1:
    #     ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    # else:
    #     ccp4.grid.symmetrize_max()
    ccp4.update_ccp4_header(2, True)
    ccp4.write_ccp4_map(str(path))


def test_sparse_dmap_stream(data_dir, out_dir):
    print(f"Data dir is {data_dir} and output dir is {out_dir}")

    # Parse the FS
    print(f"##### Loading filesystem #####")
    fs: PanDDAFSInterface = PanDDAFS(Path(data_dir), Path(out_dir))

    # Get the datasets
    print(f"##### Getting datasets #####")
    datasets: Dict[str, DatasetInterface] = {
        dataset_dir.dtag: XRayDataset.from_paths(
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
    print(f"Got {len(alignments)} alignments in {round(finish_align - begin_align, 1)}")

    # Get the reference frame
    print(f"##### Getting reference frame #####")
    begin_get_frame = time.time()
    reference_frame: DFrame = DFrame(dataset)
    finish_get_frame = time.time()
    print(f"Got reference frame in {round(finish_get_frame - begin_get_frame, 1)}")
    for resid, partition in reference_frame.partitioning.partitions.items():
        print(f"\tResid: {resid} : {partition.points.shape} {partition.positions[0,:]}")

    #
    grid = reference_frame.get_grid()
    grid_array = np.array(grid, copy=False)
    grid_array[reference_frame.mask.indicies] = 1.0
    save_dmap(
        grid,
        Path(out_dir) / f"reference.ccp4"
    )

    # Save a partition mask
    grid = reference_frame.get_grid()
    grid_array = np.array(grid, copy=False)
    for resid, partition in reference_frame.partitioning.partitions.items():
        grid_array[
            (
                partition.points[:, 0].flatten(),
                partition.points[:, 1].flatten(),
                partition.points[:, 2].flatten(),
            )
        ] = 1.0
    save_dmap(
        grid,
        Path(out_dir) / f"reference_partitions.ccp4"
    )

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
    # print(f"##### Getting sparse xmaps #####")
    # time_begin = time.time()
    # dmaps_sparse: Dict[str, SparseDMap] = {
    #     dtag: dmaps.load(dtag)
    #     for dtag
    #     in datasets
    # }
    # time_finish = time.time()
    # print(f"Got sparse xmaps in {round(time_finish - time_begin, 1)}")
    #
    # print(f"##### Saving aligned maps #####")
    # time_begin = time.time()
    # for dtag, dmap_sparse in dmaps_sparse.items():
    #     save_dmap(
    #         reference_frame.unmask(dmap_sparse),
    #         Path(out_dir) / f"{dtag}.ccp4"
    #     )
    # time_finish = time.time()
    # print(f"Saved xmaps in {round(time_finish - time_begin, 1)}")

    for dtag in datasets:
        dmap_sparse = dmaps.load(dtag)
        save_dmap(
                reference_frame.unmask(dmap_sparse),
                Path(out_dir) / f"{dtag}.ccp4"
            )


if __name__ == "__main__":
    fire.Fire(test_sparse_dmap_stream)
