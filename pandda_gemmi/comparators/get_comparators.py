from ..interfaces import *

def get_comparators(datasets: Dict[str, DatasetInterface], filters, debug=False):

    for filter in filters:
        datasets = filter(datasets)
        if debug:
            print(f'{type(filter)} : {[x for x in datasets]}')

    return datasets
