import logging
import logging.config
from functools import partial

import dask.array as da
import napari
import numpy as np
from magicgui import magic_factory
from napari.layers import Image, Points, Shapes

# logging.config.fileConfig("logging.conf")

logger = logging.getLogger(__name__)


@magic_factory(auto_call=True)
def make_matrix(
    Manual_ref_line: Shapes,
    n_cols: int = 5,
    n_rows: int = 5,
    row_multiplier: float = 1.0,
    diagonal_elements: bool = True,
    limit_to_a_slice: bool = False,
    size: int = 300,
) -> napari.types.LayerDataTuple:
    logger.debug(f"Ref line: {Manual_ref_line.data[0]}")
    logger.debug(f"Scale: {Manual_ref_line.scale}")
    manual_points = Manual_ref_line.data[0] * Manual_ref_line.scale
    logger.debug(f"manual_points: {manual_points}")

    assert len(manual_points == 2), "Select a line along your wells"
    manual_period = manual_points[1] - manual_points[0]
    logger.debug(f"manual_period: {manual_period}")
    col_period = manual_period / (n_cols - 1)
    logger.debug(f"col_period: {col_period}")

    row_period = np.zeros_like(col_period)
    row_period[-2:] = np.array([col_period[-1], -col_period[-2]])
    logger.debug(f"row_period: {row_period}")
    extrapolated_wells = np.stack(
        [
            manual_points[0]
            + col_period * i
            + row_period * j * row_multiplier
            + (col_period + row_period * row_multiplier) / 2 * k
            for k in range(2 if diagonal_elements else 1)
            for i in range(n_cols)
            for j in range(n_rows)
        ]
    )
    logger.debug(f"extrapolated_wells: {extrapolated_wells}")

    out = (
        extrapolated_wells[:, -2:],
        {
            "name": "ROIs",
            "symbol": "square",
            "size": size,
            "edge_color": "#ff0000",
            "face_color": "#00000000",
        },
        "points",
    )
    logger.debug(f"Returning: {out}")
    return out


@magic_factory
def crop_rois(
    stack: Image,
    ROIs: Points,
) -> napari.types.LayerDataTuple:
    if any([stack is None, ROIs is None]):
        return
    data = stack.data.copy()
    meta = stack.metadata.copy()
    scale = stack.scale.copy()
    no_dim_scale = scale.max()
    centers = ROIs.data / no_dim_scale
    size = (ROIs.size // no_dim_scale).max()

    _crops = map(partial(crop_stack, stack=data, size=size), centers)
    axis = 1 if data.ndim > 3 else 0
    good_crops = list(filter(lambda a: a is not None, _crops))
    try:
        meta["sizes"] = update_dict_with_pos(
            meta["sizes"], axis, "P", len(good_crops)
        )
    except KeyError:
        meta = {"sizes": {"X": 0, "Y": 0, "P": len(good_crops)}}
        logger.error(
            rf"Failed updating meta[`sizes`] with %s ", {'P': len(good_crops)}
        )
    meta["sizes"]["X"] = size
    meta["sizes"]["Y"] = size
    meta["crop_centers"] = centers
    meta["crop_size"] = size

    return (
        da.stack(good_crops, axis=axis),
        {"scale": scale, "metadata": meta, "name": "crops"},
        "image",
    )


def update_dict_with_pos(input_dict: dict, pos, key, value):
    """Inserts {key: value} into position"""
    k = list(input_dict.keys())
    k.insert(pos, key)
    v = list(input_dict.values())
    v.insert(pos, value)
    return {kk: vv for kk, vv in zip(k, v)}


def crop_stack(center: np.ndarray, stack: np.ndarray, size: int) -> np.ndarray:
    """
    Crops a square of the size `size` px from last two axes accrding to
    2 last coordinates of the center.
    Returns stack[...,size,size] if crop fits into the stack size, otherwise returns None.
    """
    assert stack.ndim >= 2
    assert all(
        [center.ndim == 1, len(center) >= 2]
    ), f"Problem with center {center} of len {len(center)}"
    s = (size // 2).astype(int)
    y, x = center[-2:].astype(int)
    y1, y2 = y - s, y + s
    x1, x2 = x - s, x + s
    ylim, xlim = stack.shape[-2:]

    if any([y1 < 0, x1 < 0, y2 > ylim, x2 > xlim]):
        return
    return stack[..., y1:y2, x1:x2]
