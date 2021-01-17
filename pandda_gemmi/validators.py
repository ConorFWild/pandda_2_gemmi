

import dataclasses

from  pandda_gemmi import pandda_exceptions
from pandda_gemmi import pandda_types


@dataclasses.dataclass()
class DatasetsValidator:
    min_characterisation_datasets: int
    
    def validate(self, dataset: pandda_types.Datasets, stage: str):
        if len(datasets) < self.min_characterisation_datasets:
            raise pandda_exceptions.ExceptionTooFewDatasets(stage)