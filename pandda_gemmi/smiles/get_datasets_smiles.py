from rdkit import Chem
from openbabel import pybel

from pandda_gemmi.analyse_interface import *
from pandda_gemmi.autobuild.cif import generate_cif


class GetDatasetSmiles(GetDatasetSmilesInterface):
    def __init__(self):
        ...

    def smiles_path_from_cif(
            self,
            processed_dataset_path,
            output_smiles_path,
            source_ligand_cif,
            debug: Debug = Debug.DEFAULT,
    ):
        # source_ligand_cif = fragment_dataset.source_ligand_cif

        pdb_path = processed_dataset_path / "tmp.pdb"
        try:
            pybel_mol = next(pybel.readfile("cif", str(source_ligand_cif)))
        except Exception as e:
            print(f"exception is: {e}")
            raise Exception(f"Error in extracting smiles from cif file: {str(source_ligand_cif)}. Unable to iterate file!")
        with open(pdb_path) as f:
            f.write(pybel_mol.write("pdb"))

        # # Run phenix to normalize cif
        # cif_path, pdb_path = generate_cif(
        #     source_ligand_cif,
        #     fragment_dataset.path
        # )

        # # Open new pdb with open babel
        # mol = next(pybel.readfile("pdb", str(pdb_path)))
        #
        # # Convert to cif and back again to deal with psky Hs that confuse RDKIT
        # smiles = pybel.readstring("cif", mol.write("cif")).write("smiles")

        # Read pdb to rdkit
        mol = Chem.MolFromPDBFile(str(pdb_path))

        # Write smiles
        smiles = Chem.MolToSmiles(mol)

        # Write the smiles
        smiles_path = output_smiles_path
        with open(smiles_path, "w") as f:
            f.write(smiles)

        return smiles_path

    def smiles_path_from_pdb(
            self,
            processed_dataset_path,
            output_smiles_path,
            source_ligand_pdb,
            debug: Debug = Debug.DEFAULT):
        # source_ligand_pdb = fragment_dataset.source_ligand_pdb

        # # Run phenix to normalize cif
        # cif_path, pdb_path = generate_cif(
        #     source_ligand_pdb,
        #     fragment_dataset.path
        # )

        # # Open new pdb with open babel
        # mol = next(pybel.readfile("pdb", str(pdb_path)))
        #
        # # Convert to cif and back again to deal with psky Hs that confuse RDKIT
        # smiles = pybel.readstring("cif", mol.write("cif")).write("smiles")

        # Read pdb to rdkit
        mol = Chem.MolFromPDBFile(str(source_ligand_pdb))

        # Write smiles
        smiles = Chem.MolToSmiles(mol)

        # Write the smiles
        smiles_path = output_smiles_path
        with open(smiles_path, "w") as f:
            f.write(smiles)

        return smiles_path

    def __call__(self,
                 processed_dataset_path: Path,
                 output_smiles_path: Path,
                 ligand_pdb_path: Optional[Path],
                 ligand_cif_path: Optional[Path],
                 ligand_smiles_path: Optional[Path],
                 ) -> Optional[Path]:
        """
        Ensure that the source ligand smiles of the dataset is populated, if necessary by generating it from the cif
        or pdb.
        """

        if ligand_smiles_path:
            return ligand_smiles_path

        elif ligand_pdb_path:
            return self.smiles_path_from_pdb(processed_dataset_path, output_smiles_path, ligand_pdb_path)

        elif ligand_cif_path:
            return self.smiles_path_from_cif(processed_dataset_path, output_smiles_path, ligand_cif_path)

        else:
            return None
