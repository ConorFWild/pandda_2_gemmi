import gemmi

from ..interfaces import *


class FilterCompatibleStructures:
    def __init__(self, dataset, similarity=100.0):
        self.dataset = dataset
        self.similarity = similarity
        self.resids = set(resid for resid in self.dataset.structure.protein_residue_ids())

#     def get_compatible(self, dataset):
#         ref_ents = self.dataset.structure.structure.entities
#         # ref_mod_sequence = ref_mod.full_sequence
#
#         mov_ents = dataset.structure.structure.entities
#         mov_ent_names = [ent.name for ent in mov_ents]
#
#         for ref_ent in ref_ents:
#             if not ref_ent.entity_type == gemmi.EntityType.Polymer:
#                 print(f"Entity type is: {ref_ent.entity_type}")
#                 continue
#             if ref_ent.name not in mov_ent_names:
#                 print(f"No matching mov ent to: {ref_ent.name}")
#                 return False
#             else:
#                 ref_seq = [gemmi.Entity.first_mon(item) for item in ref_ent.full_sequence]
#                 mov_seq = [gemmi.Entity.first_mon(item) for item in dataset.structure.structure.get_entity(ref_ent.name).full_sequence]
#                 result = gemmi.align_string_sequences(
#                     ref_seq,
#                     mov_seq,
#                     [], gemmi.prepare_blosum62_scoring()
# ,
#                 )
#
#                 identity = result.calculate_identity()
#                 print(f"\t\t\t{ref_ent.name} : {identity} : {len(ref_seq)} : {len(mov_seq)}")
#
#                 if not result.calculate_identity() >= self.similarity:
#                     return False
#
#         return True

    def get_compatible(self, dataset):
        dataset_resids = set([resid for resid in dataset.structure.protein_residue_ids()])
        if len(dataset_resids.intersection(self.resids)) != len(self.resids):
            return False
        else:
            return True

    def __call__(self, datasets: Dict[str, DatasetInterface]):
        new_datasets = {}
        for _dtag in datasets:
            dataset = datasets[_dtag]
            if self.get_compatible(dataset):
                new_datasets[_dtag] = dataset

        print(f"Structure Compatibility: New datasets {len(new_datasets)} vs old {len(datasets)}")

        return new_datasets
