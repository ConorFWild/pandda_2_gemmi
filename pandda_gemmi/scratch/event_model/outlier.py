import numpy as np

class PointwiseNormal:
    def __call__(self,
                 dataset_dmap_array,
                characterization_set_dmaps_array,
                 ):
        mean = np.mean(characterization_set_dmaps_array, axis=0)
        std = np.std(characterization_set_dmaps_array, axis=0)
        z = ((dataset_dmap_array - mean) / std)
        return mean, std, z