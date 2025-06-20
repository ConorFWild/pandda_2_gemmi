from pandda_gemmi.interfaces import *
import numpy as np
from scipy import optimize


def get_bdc(event, xmap_grid, mean_grid, median, ):
    # Get arrays of the xmap and mean map
    xmap_array = np.array(xmap_grid, copy=False)
    mean_array = np.array(mean_grid, copy=False)

    # Get the indicies corresponding to event density i.e. a selector of event density
    event_indicies = tuple(
        [
            event.point_array[:, 0].flatten(),
            event.point_array[:, 1].flatten(),
            event.point_array[:, 2].flatten(),
        ]
    )

    # Get the values of the 2Fo-Fc map and mean map for the event
    xmap_vals = xmap_array[event_indicies]
    mean_map_vals = mean_array[event_indicies]

    # Get the BDC by minimizing the difference between masked event map density and the median of protein density
    res = optimize.minimize(
        lambda _bdc: np.abs(
            np.median(
                (xmap_vals - (_bdc * mean_map_vals)) / (1 - _bdc)
            ) - median
        ),
        0.5,
        bounds=((0.0, 0.95),),
        tol=0.1
    )

    return float(res.x)
