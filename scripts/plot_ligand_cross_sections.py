import itertools
from pathlib import Path
import os

import fire
import gemmi
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def get_ligand_pos_array(lig):
    poss = []
    for atom in lig:
        if atom.element.name != "H":
            pos = atom.pos
            poss.append([pos.x, pos.y, pos.z])

    return np.array(poss)


def get_ligand_plane_frame(pos_array):
    # Get the ligand plane
    pca = PCA(n_components=2)
    pca.fit(pos_array)

    return pca


def get_ligand_pos_in_ligand_plane(pos_array, ligand_plane_frame):
    return ligand_plane_frame.transform(pos_array)


def sample_ligand_plane(ligand_plane_frame, ligand_plane_pos_array, xmap, border=5.0, rate=0.1):
    # Get the grid boundaries around the ligand in its plane frame
    lower = np.min(ligand_plane_pos_array, axis=0) - border
    initial_upper = np.max(ligand_plane_pos_array, axis=0) + border
    upper = lower + (np.round((initial_upper - lower) / rate, ) * rate)

    # Get the grid coordinates
    nx = int((upper[0] - lower[0]) / rate)
    ny = int((upper[1] - lower[1]) / rate)

    # Get the lower coord in 3d
    mean_2d = np.mean(ligand_plane_pos_array, axis=0)
    # print(f'Mean 2d: {mean_2d}')
    # print(ligand_plane_frame.inverse_transform(mean_2d))
    # print(ligand_plane_frame.mean_)
    lower_rel_2d = lower - mean_2d
    lower_3d = ligand_plane_frame.inverse_transform(lower_rel_2d)
    # print(lower_3d)

    # Sample xmap
    samples = np.zeros((nx, ny))
    sample_poss = []
    grid_poss = []
    for u, v in itertools.product(range(nx), range(ny)):
        sample_pos = ligand_plane_frame.inverse_transform(np.array([[(u - (nx / 2)) * rate, (v - (ny / 2)) * rate]]))
        grid_poss.append([(u - (nx / 2)) * rate, (v - (ny / 2)) * rate])
        sample_poss.append([sample_pos[0][0], sample_pos[0][1], sample_pos[0][2]])
        sample = xmap.interpolate_value(gemmi.Position(sample_pos[0][0], sample_pos[0][1], sample_pos[0][2]))
        # print([[u, v], sample_pos, sample])
        samples[u, v] = sample

    # Sample lig
    samples_lig = (ligand_plane_pos_array / rate) + np.array([nx / 2, ny / 2]).reshape(1, -1)

    # print('Range ligand 2d')
    # print(np.min(np.array(ligand_plane_pos_array), axis=0))
    # print(np.max(np.array(ligand_plane_pos_array), axis=0))
    #
    # print('Range grid 2d')
    # print(np.min(np.array(grid_poss), axis=0))
    # print(np.max(np.array(grid_poss), axis=0))
    #
    # print('Range ligand 3d')
    # print(np.min(np.array(ligand_plane_frame.inverse_transform(ligand_plane_pos_array)), axis=0))
    # print(np.max(np.array(ligand_plane_frame.inverse_transform(ligand_plane_pos_array)), axis=0))
    #
    # print(f'Range grid 3d')
    # print(np.min(np.array(sample_poss), axis=0))
    # print(np.max(np.array(sample_poss), axis=0))
    #
    # print('range ligand 2d grid')
    # print(np.min(np.array(samples_lig), axis=0))
    # print(np.max(np.array(samples_lig), axis=0))
    #
    # print('range 2d grid')
    # print([nx, ny])

    return samples, samples_lig  # * np.array([[nx, ny]]) * rate


def plot_contours(
        samples_xmap,
        samples_lig,
        output_path,
        vmax=0.1
):
    fig, ax = plt.subplots()
    im = ax.imshow(samples_xmap.T, origin='lower', vmin=0.0, vmax=vmax, )
    fig.colorbar(im)
    ax.scatter(x=samples_lig[:, 0], y=samples_lig[:, 1], c='red')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.savefig(output_path)


def plot_ligand_cross_section(lig, xmap, fig_output_path, lig_output_path, map_output_path, vmax=0.1, ):
    # Get ligand pos array
    pos_array = get_ligand_pos_array(lig)

    # Get ligand plane frame
    ligand_plane_frame = get_ligand_plane_frame(pos_array)

    # Get the pos array in the ligand frame
    ligand_plane_pos_array = get_ligand_pos_in_ligand_plane(pos_array, ligand_plane_frame)

    # Get xmap samples
    samples_xmap, samples_lig = sample_ligand_plane(ligand_plane_frame, ligand_plane_pos_array, xmap)

    # Plot sample
    plot_contours(
        samples_xmap,
        samples_lig,
        fig_output_path,
        vmax
    )

    # Save arrays
    with open(lig_output_path, 'wb') as f:
        np.save(f, samples_lig)
    with open(map_output_path, 'wb') as f:
        np.save(f, samples_xmap)


def iterate_ligands(st):
    for model in st:
        for chain in model:
            for res in chain:
                if res.name in ['LIG', "XXX"]:
                    yield f"{chain.name}_{res.seqid.num}", res


def plot_cross_section(
        st_path,
        map_path,
        output_dir,
        vmax=0.1,
        norm=False
):
    # Make output dir
    if not Path(output_dir).exists():
        os.mkdir(output_dir, )

    # Get structure
    st = gemmi.read_structure(st_path)

    # Get map
    ccp4 = gemmi.read_ccp4_map(map_path)
    ccp4.setup(0.0)
    xmap = ccp4.grid
    if norm:
        print(f'Norming!')
        xmap_array = np.array(xmap, copy=False)
        xmap_array_nonzero = xmap_array[xmap_array!=0.0]
        std = np.std(xmap_array_nonzero)
        mean = np.mean(xmap_array_nonzero)
        xmap_array[:,:,:] = (xmap_array.copy()[:,:,:] - mean) / std

    # Plot crossection through each ligand residue
    for resid, lig in iterate_ligands(st):
        plot_ligand_cross_section(
            lig,
            xmap,
            Path(output_dir) / f'{resid}_{Path(map_path).stem}.png',
            Path(output_dir) / f'{resid}_{Path(map_path).stem}_lig.npz',
            Path(output_dir) / f'{resid}_{Path(map_path).stem}_map.npz',
            vmax=vmax
        )
    ...


if __name__ == "__main__":
    fire.Fire(plot_cross_section)
