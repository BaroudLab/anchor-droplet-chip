from magicgui import magicgui
import napari
from napari.layers import Shapes
import numpy as np

@magicgui
def make_matrix(
    Manual_ref_line:Shapes,
    n_cols : int = 5,
    n_rows : int = 5,
    row_multiplier : float = 1.,
    check : bool = True,
    size : int = 300
    
) -> napari.types.LayerDataTuple:
    manual_points = Manual_ref_line.data[0]
    assert len(manual_points == 2), "Select a line along your wells"
    manual_period = manual_points[1] - manual_points[0]
    col_period = manual_period / (n_cols - 1)
    
    row_period = np.zeros_like(col_period)
    row_period[-2:] = np.array([col_period[-1], -col_period[-2]])
    extrapolated_wells = np.stack([
        manual_points[0] + \
        col_period * i + \
        row_period * j * row_multiplier + \
        (col_period + row_period * row_multiplier) / 2 * k \
        for k in range(2 * check) \
        for i in range(n_cols) \
        for j in range(n_rows)
    ])
    
    return (
        extrapolated_wells[:,-2:],
        {"symbol": "square",
         "size" : size,
         "edge_color": "#ff0000",
         "face_color": "#00000000"
        },
        "points"
    )