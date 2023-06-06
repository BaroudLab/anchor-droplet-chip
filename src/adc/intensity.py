"""
Measure intensity and background with `get_intensity_of_single_crop`
"""

import functools
import logging
from typing import Tuple

import dask.array as da
import numpy as np
import pandas as pd
from tqdm import tqdm

from .count import crop2d, load_mem, make_table

logger = logging.getLogger("adc.count")


def log(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        logger.debug(
            f"function `{func.__name__}` called with args `{signature}`"
        )
        try:
            result = func(*args, **kwargs)
            logger.debug(f"function `{func.__name__}` returns `{result}`")
            return result
        except Exception as e:
            logger.exception(
                f"Exception raised in `{func.__name__}`. exception: `{str(e)}`"
            )
            raise e

    return wrapper


@log
def estimate_norm_mean(values, eps=0.1, dtype=np.float32):
    """iteratively shaves off the values higher than mean + 3 * std to get a good mean value"""
    assert len(values) > 0, f"Values empty: {values}"
    mean, std = values.mean(dtype=dtype), values.std(dtype=dtype)
    if std == 0:
        raise ValueError(f"Data has no variance {values}")
    good_values = values[values < mean + 3 * std]
    new_mean = good_values.mean()
    if np.abs(new_mean - mean) > eps:
        return estimate_norm_mean(good_values)
    else:
        return new_mean


@log
def bg_op(
    data, outline_thickness=1, intensity_ops=(np.mean, estimate_norm_mean)
):
    """
    Measures intensity of the edge pixels with `intensity_op`
    """
    logger.debug(
        f"bg_op: data {data.shape}, outline+thickness {outline_thickness}, intensity_ops {intensity_ops}"
    )
    t = outline_thickness
    values = np.concatenate(
        (
            np.ravel(data[:-t, :t]),
            np.ravel(data[-t:, :-t]),
            np.ravel(data[t:, -t:]),
            np.ravel(data[:t, t:]),
        )
    )
    logger.debug(f"outline produces {len(values)} values")
    bg = [op(values) for op in intensity_ops]
    logger.debug(f"returning {bg}")
    return tuple(bg)


@log
def measure_intensity_bg_of_crop(
    data: np.ndarray, intensity_op=np.mean, bg_op=bg_op, dtype=np.float32
):
    """
    measure the mean intensity of the data and background from the outer pixels.
    Parameters:
    -----------
    data: np.ndarray 2D
    intensity_op: callable, default np.mean
    bg_op: callable
    Return:
    -------
    intensity, bg_mean, bg_normal_mean (float, float, float)
    """
    intensity = intensity_op(data, dtype=dtype)
    bg_results = bg_op(data) if bg_op is not None else -1
    return (intensity, *bg_results)


@log
def get_intensity_of_single_crop(
    data: np.ndarray,
    center: np.ndarray,
    size: int,
    crop_op=crop2d,
    measure_op=measure_intensity_bg_of_crop,
):
    """
    Gets the intensity and background measure around the center `center` with the size `size`
    Parameters:
    -----------
    fluo_data: np.ndarray 2D
        Fluorescence slice
    center: tuple(y,x)
        Central coordinate of a roi
    size: int
        Square size of the ROI
    Return:
    -------
    intesities: tuple
    """
    results = measure_op(
        crop_op(data, center, size),
    )
    return results


@log
def measure_all_positions_2d(
    data,
    positions,
    size,
    crop_op=crop2d,
    single_coord_op=get_intensity_of_single_crop,
    loader=load_mem,
):
    if isinstance(data, da.Array):
        data = loader(data)
        logger.debug(f"loaded {data.shape}")

    results = np.vstack(
        [
            single_coord_op(
                data=data, center=center, size=size, crop_op=crop_op
            )
            for center in positions
        ]
    )
    return results


@log
def measure_recursive(
    data: da.Array,
    positions: list,
    size: int,
    index: list = [],
    progress=tqdm,
    measure_function=get_intensity_of_single_crop,
    crop_op=crop2d,
    table_function=make_table,
) -> Tuple[list, list, list, pd.DataFrame]:
    """
    Recurcively processing 2d arrays.
    data: np.ndarray n-dimensional
    positions: np.ndarray 2D (m, n')
        where m - number of positions
        n' - number of dimensions, can be smaller than n, but not bigger
        two last columns: y, x
        others: dimentionsl indices (from napari)
    returns:
    --------
    (loc_result, count_result:list, droplets_out: list, df: pd.DataFrame)
    """
    logger.debug(f"count {data}")
    if data.ndim > 2:
        pos = []
        tables = []
        for i, d in enumerate(progress(data)):
            new_ind = index + [i]
            logger.info(f"index {new_ind}")
            if positions.shape[-1] <= d.ndim:
                use_coords = positions
            else:
                use_coords = positions[positions[:, 0] == i][:, -d.ndim :]
            # print(use_coords)
            (
                coords_droplets,
                df,
            ) = measure_recursive(
                d,
                positions=use_coords,
                size=size,
                index=new_ind,
                measure_function=measure_function,
                crop_op=crop_op,
            )
            tables.append(df)
            pos += coords_droplets
        return (
            pos,
            pd.concat(tables, ignore_index=True),
        )
    else:
        coords = positions[:, -2:]
        results = measure_all_positions_2d(
            data=data,
            positions=coords,
            size=size,
            crop_op=crop_op,
        )
        logger.debug(
            f"Finished measuring index {index}: results {results.shape} "
        )

        droplets_out = [index + list(o) for o in coords]
        logger.debug(f"Added index {index} to {len(coords)} positions")

        try:
            labels = (np.arange(len(results)) + 1).reshape((len(results), 1))
            df = pd.DataFrame(
                data=np.hstack([droplets_out, results, labels]),
                columns=[
                    "frame",
                    "chip",
                    "y",
                    "x",
                    "mean_intensity",
                    "mean_bg",
                    "mean_norm_bg",
                    "label",
                ],
            )
            df.loc[:, "signal"] = df.mean_intensity - df.mean_norm_bg
        except Exception as e:
            logger.error(f"Making dataframe failed: {e}")
            raise (e)
        return droplets_out, df
