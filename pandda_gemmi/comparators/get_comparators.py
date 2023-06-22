from ..interfaces import *

def get_comparators(datasets: Dict[str, DatasetInterface], filters):

    for filter in filters:
        datasets = filter(datasets)

    return datasets