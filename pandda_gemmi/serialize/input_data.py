import yaml

from pandda_gemmi.interfaces import *


def input_data(fs: PanDDAFSInterface, datasets: Dict[str, DatasetInterface], path):
    dic = {}

    dic["Summary"] = {
        "Number of Datasets": len(datasets)
    }

    dic["Datasets"] = {}
    for dtag in sorted(datasets):
        dataset = datasets[dtag]
        dic["Datasets"][dtag] = {
            "Files": {
                "PDB": str(dataset.structure.path),
                "MTZ": str(dataset.reflections.path),
                "Ligand Files": {
                    ligand_key: {
                        "PDB": str(ligand_files.ligand_pdb),
                        "CIF": str(ligand_files.ligand_cif),
                        "SMILES": str(ligand_files.ligand_smiles),
                    }
                    for ligand_key, ligand_files
                    in dataset.ligand_files.items()
                }
            },
            "Structure": {
                "Number of Chains": None
            },
            "Reflections": {
                "Resolution": round(dataset.reflections.resolution(), 2),
                "Unit Cell": {
                    "a": round(dataset.structure.structure.cell.a, 2),
                    "b": round(dataset.structure.structure.cell.b, 2),
                    "c": round(dataset.structure.structure.cell.c, 2),
                    "alpha": round(dataset.structure.structure.cell.alpha, 2),
                    "beta": round(dataset.structure.structure.cell.beta, 2),
                    "gamma": round(dataset.structure.structure.cell.gamma, 2),
                },
            }
        }

    with open(path, 'w') as f:
        yaml.dump(dic, f, sort_keys=False)
    ...
