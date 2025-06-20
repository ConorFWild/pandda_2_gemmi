import numpy as np
import mrcfile
from mrcfile.mrcfile import MrcFile
import gemmi

from pandda_gemmi.analyse_interface import *


def _metadata(mrc: MrcFile):
    """

    :param mrc: a MrcFile object
    :return: a dict with the following info:
                sampling_rate:float, the voxel size.
                xyz_ori_inA:Tuple[float], The xyz coordinates of the volume, in A. The origin of coordinates corresponds to the voxel 0,0,0
                n_voxels:int, the number of voxels in each axis. The volume is of shape (n_voxels,n_voxels,n_voxels)
                axis_order:Tuple[float], [0,1,2] for "zyx", [2,1,0] for "xyz", etc.
    """

    return dict(sampling_rate=mrc.voxel_size["x"], xyz_ori_inA=np.array([mrc.header["origin"][ax] for ax in "xyz"]),
                n_voxels=int(mrc.header["nx"]),
                axis_order=[int(mrc.header[name]) - 1 for name in ["mapc", "mapr", "maps"]])


def data_and_md_from_mrcfile(fname: str, desired_axes_order: Literal["zyx", "xyz"] = "zyx") \
        -> Tuple[np.array, Dict[str, Any]]:
    """

    :param fname: A .mrc filename
    :param desired_axes_order: Either "zyx" or "xyz"
    :return: (volume, metadata)
    """
    with mrcfile.open(fname, permissive=True) as mrc:
        md = _metadata(mrc)
        axes_order = md["axis_order"]
        if desired_axes_order == "zyx":
            desired_axes_order = [0, 1, 2]
        elif desired_axes_order == "xyz":
            desired_axes_order = [2, 1, 0]
        else:
            raise ValueError("desired_axes_order is not valid, only zyx and xyz are considered")
        if axes_order != desired_axes_order:  # Relion-like vols are zyx so axes_order == [1,2,3]
            data = mrc.data.copy().transpose(axes_order)
            md["axis_order"] = desired_axes_order

        else:
            data = mrc.data.copy()

        return data, md


def md_from_mrcfile(fname):
    with mrcfile.open(fname, permissive=True, header_only=True) as mrc:
        return _metadata(mrc)


def data_to_mrcfile(fname, data, sampling_rate: float = None, xyz_ori_inA=(0, 0, 0), n_voxels: int = None,
                    axis_order=None,
                    overwrite=True):
    '''

    :param fname: name where mrc file will be saved
    :param data: numpy array  that contains the data
    :param sampling_rate: the sampling rate of the data.
    :param xyz_ori_inA: the xyz_ori_inA rate of the data.
    :param n_voxels: the n_voxels per axis
    :param axis_order: the order of the axes. So far only relion-like zyx is accepted
    :param overwrite: If true, overwrite existing volumes
    :return: None
    '''

    assert sampling_rate is not None, "Error, when saving volume, either sampling rate or fnameIn to same size volume required"
    if n_voxels is not None:
        assert np.unique(data.shape)[0] == n_voxels, "Error, n_voxels do not match data.shape"

    if axis_order is not None:
        assert axis_order == [0, 1, 2], f"Error, only axis_order zyx (0,1,2) is supported. Provided ({axis_order})"

    with mrcfile.new(fname, overwrite=overwrite) as f:
        f.set_data(data.astype(np.float32))
        f.voxel_size = tuple([sampling_rate] * 3)
        f.header["origin"]["x"] = xyz_ori_inA[0]
        f.header["origin"]["y"] = xyz_ori_inA[1]
        f.header["origin"]["z"] = xyz_ori_inA[2]


def numpy_to_grid(data, md):
    grid_spacing = md["n_voxels"]
    cell_lengths = md["sampling_rate"] * grid_spacing

    spacing = [grid_spacing, grid_spacing, grid_spacing]
    unit_cell = gemmi.UnitCell(cell_lengths, cell_lengths, cell_lengths, 90.0, 90.0, 90.0)
    grid = gemmi.FloatGrid(spacing[0], spacing[1], spacing[2])
    grid.set_unit_cell(unit_cell)
    grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")

    grid_array = np.array(grid, copy=False)
    grid_array[:, :, :] = data[:, :, :]

    # ccp4 = gemmi.Ccp4Map()
    # ccp4.grid = grid
    # ccp4.update_ccp4_header(2, True)
    # ccp4.setup()
    # ccp4.write_ccp4_map(str(path))

    return grid


def grid_to_mtz(grid, data):
    sf = gemmi.transform_map_to_f_phi(grid)
    data = sf.prepare_asu_data(dmin=0)

    mtz = gemmi.Mtz(with_base=True)
    mtz.spacegroup = sf.spacegroup
    mtz.set_cell_for_all(sf.unit_cell)
    mtz.add_dataset('unknown')
    mtz.add_column('FWT', 'F')
    mtz.add_column('PHWT', 'P')
    mtz.set_data(data.astype(np.float32))

    return mtz


def mrc_to_mtz(mrc_file: Path, mtz_file: Path):
    data, md = data_and_md_from_mrcfile(str(mrc_file), "xyz")

    grid = numpy_to_grid(data, md)

    mtz = grid_to_mtz(grid, data)

    mtz.write_to_file(str(mtz_file))
