import numpy as np
from scipy.cluster.hierarchy import fclusterdata

from ..interfaces import *

from .site import Site

class HeirarchicalSiteModel:

    def __init__(self, t=8.0):
        self.t = t

    def __call__(self, events: Dict[Tuple[str, int], EventInterface]):

        #
        if len(events) == 0:
            return {}

        if len(events) == 1:
            return {0: Site()}

        # Get the array of centroids
        centroid_array = np.array(
            [
                np.mean(event.pos_array, axis=0)
                for event
                in events.values()
            ]
        )

        #
        clusters = fclusterdata(
            centroid_array,
            t=self.t,
            criterion="distance",
            method="centroid"
        )

        #

    ...