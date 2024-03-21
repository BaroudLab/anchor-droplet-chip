import logging
import pathlib
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from typing import Tuple, Union

import dask.array as da
import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from tifffile import imread
from tqdm import tqdm

from adc.fit import poisson as fit_poisson

try:
    __version__ = version("anchor-droplet-chip")
except PackageNotFoundError:
    # package is not installed
    __version__ = "Unknown"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s : %(message)s"
)
logger = logging.getLogger("adc.count")


def stack(bf_fluo_mask: np.ndarray):
    """
    Wraper for get_cell_numbers, accepting the aligned stack as a single parameter.
    """
    bf, fluo, mask = bf_fluo_mask
    return get_cell_numbers(multiwell_image=fluo, labels=mask, bf=bf)


def get_cell_numbers(
    multiwell_image: np.ndarray,
    labels: np.ndarray,
    plot=False,
    threshold_abs: float = 2,
    min_distance: float = 5,
    dif_gauss_sigma=(3, 5),
    bf: np.ndarray = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Counting fluorescence peaks inside the labels.
    The data is filtered by gaussian difference filter first.
    The table with labels, coordinates and number of particles returned.
    """

    props = regionprops(labels)

    def get_raw_peaks(i):
        if bf is None:
            return get_peaks(
                multiwell_image[props[i].slice],
                plot=plot,
                dif_gauss_sigma=dif_gauss_sigma,
                threshold_abs=threshold_abs,
                min_distance=min_distance,
                title=props[i].label,
            )
        else:
            return get_peaks(
                multiwell_image[props[i].slice],
                plot=plot,
                dif_gauss_sigma=dif_gauss_sigma,
                threshold_abs=threshold_abs,
                min_distance=min_distance,
                title=props[i].label,
                bf_crop=bf[props[i].slice],
            )

    peaks = list(map(get_raw_peaks, range(labels.max())))
    n_cells = list(map(len, peaks))
    return pd.DataFrame(
        [
            {
                "label": prop.label,
                "x": int(prop.centroid[1]),
                "y": int(prop.centroid[0]),
                "n_cells": n_cell,
                **kwargs,
            }
            for prop, n_cell in zip(props, n_cells)
        ]
    )


def cropNd(stack: np.ndarray, center: tuple, size: int):
    """
    Crops the last two dimensions around the center(y,x) with the size(size, size)
    """
    im = stack[
        :,
        int(center[0]) - size // 2 : int(center[0]) + size // 2,
        int(center[1]) - size // 2 : int(center[1]) + size // 2,
    ]
    return im


def crop2d(img: np.ndarray, center: tuple, size: int):
    """
    2D crop
    """
    im = img[
        int(center[0]) - size // 2 : int(center[0]) + size - size // 2,
        int(center[1]) - size // 2 : int(center[1]) + size - size // 2,
    ]
    return im


def gdif(
    array2d: np.ndarray,
    dif_gauss_sigma: Tuple[float, float] = (1, 3),
    op=ndi.gaussian_filter,
):
    """
    Computes gaussian difference filter with two sigmas frpm 2d array array2d
    """
    array2d = array2d.astype("f")
    return op(array2d, sigma=dif_gauss_sigma[0]) - op(
        array2d, sigma=dif_gauss_sigma[1]
    )


def add_chip_index_to_coords(coords: tuple, chip_index):
    return (chip_index, *coords)


def make_table(droplets_out: list, counts: list) -> pd.DataFrame:
    """
    Merging horizontally the provided tables and adding labels
    Parameters:
    -----------
    droplets_out: list[[index-0, index-1, ... index-n, y, x]]
    counts: list(n1,n2,...n[len(droplets_out)]) -
        list of counts of the same lenght as droplets
    Return:
    -------
    dataframe: pd.DataFrame with the following columns:
        [index-0, index-1, ..., y, x, n_cells, label],
        where label being automatic index starting with 1 and ending with len(counts)
    """
    droplets_out = np.array(droplets_out)
    counts = np.array(counts).reshape((len(counts), 1))
    labels = (np.arange(len(counts)) + 1).reshape((len(counts), 1))
    return pd.DataFrame(
        data=np.hstack([droplets_out, counts, labels]),
        columns=[f"index-{i}" for i in range(len(droplets_out[0][:-2]))]
        + ["y", "x", "n_cells", "label"],
    )


def load_mem(dask_array: da.Array) -> np.ndarray:
    """
    Loads dask array to memory
    """
    logger.debug(f"loading {dask_array}")
    return dask_array.compute()


def get_peaks(
    crop_2d: np.ndarray,
    dif_gauss_sigma: tuple = (3, 5),
    min_distance: int = 3,
    threshold_abs: float = 2,
    plot: bool = False,
    title: str = "",
    bf_crop: Union[np.ndarray, None] = None,
    peak_op=peak_local_max,
    gdif_op=gdif,
):
    """
    Applies gaussian difference using gdif_op and localizes the peaks using peak_op.
    Can plot the drop and localizations if plot is set to True.
    Returns:
    list of 2d coordinates
    """
    image_max = gdif_op(crop_2d, dif_gauss_sigma)
    peaks = peak_op(
        image_max, min_distance=min_distance, threshold_abs=threshold_abs
    )
    # logger.debug(f"found {len(peaks)} cells")
    if plot:
        if bf_crop is None:
            fig, ax = plt.subplots(1, 2, sharey=True)
            ax[0].imshow(crop_2d)
            ax[0].set_title(f"raw image {title}")
            ax[1].imshow(image_max)
            ax[1].set_title("Filtered + peak detection")
            ax[1].plot(peaks[:, 1], peaks[:, 0], "r.")
            plt.show()
        else:
            fig, ax = plt.subplots(1, 3, sharey=True)

            ax[0].imshow(bf_crop, cmap="gray")
            ax[0].set_title(f"BF {title}")

            ax[1].imshow(crop2d, vmax=crop_2d.mean() + 2 * crop_2d.std())
            ax[1].set_title(f"raw image {title}")

            ax[2].imshow(image_max)
            ax[2].set_title(
                f"Filtered + {len(peaks)} peaks (std {image_max.std():.2f})"
            )
            ax[2].plot(peaks[:, 1], peaks[:, 0], "r.")
            plt.show()

    return peaks


def get_global_peaks(
    fluo_data: np.ndarray,
    center: np.ndarray,
    size: int,
    crop_op=crop2d,
    localizer=get_peaks,
):
    """
    Localizes the peaks around the center, returns the coordinates inside big fluo_data array.
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
    array: np.array([[y0,x0], [y1,x1],...])
    """
    peaks = localizer(
        crop_op(fluo_data, center, size),
    )
    return np.array(peaks) + np.array(center) - size / 2


def count2d(
    data: Union[np.ndarray, da.Array],
    positions: list,
    size: int,
    localizer=get_global_peaks,
    loader=load_mem,
    crop_op=crop2d,
    **table_args,
):
    """
    returns 2d array of positions and list of counts per position
    """
    logger.debug(f"count 2d {data.shape}, {len(positions)} positions")
    if isinstance(data, da.Array):
        data = loader(data)
        logger.debug(f"loaded {data.shape}")

    logger.debug("Start counting")
    peaks = np.vstack(
        positions_per_droplet := [
            localizer(
                fluo_data=data, center=center, size=size, crop_op=crop_op
            )
            for center in positions
        ]
    )
    counts = list(map(len, positions_per_droplet))

    return peaks, counts


def count_recursive(
    data: da.Array,
    positions: list,
    size: int,
    index: list = [],
    progress=tqdm,
    counting_function=count2d,
    localizer=get_global_peaks,
    crop_op=crop2d,
    table_function=make_table,
    loc_result: list = [],
    count_result: list = [],
    droplet_pos: list = [],
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
        locs = []
        counts = []
        pos = []
        tables = []
        for i, d in enumerate(progress(data)):
            new_ind = index + [i]
            logger.debug(f"index {new_ind}")
            if positions.shape[-1] < len(data.shape):
                use_coords = positions
            else:
                use_coords = positions[positions[:, 0] == i]
            (
                bac_locs,
                per_droplet_counts,
                coords_droplets,
                df,
            ) = count_recursive(
                d,
                positions=use_coords,
                size=size,
                index=new_ind,
                localizer=localizer,
                counting_function=counting_function,
                crop_op=crop_op,
                table_function=table_function,
            )
            tables.append(df)
            locs += bac_locs
            loc_result = locs

            counts += per_droplet_counts
            count_result = counts

            pos += coords_droplets
            droplet_pos = pos
        return (
            loc_result,
            count_result,
            droplet_pos,
            pd.concat(tables, ignore_index=True),
        )
    else:
        coords = positions[:, -2:]
        peaks, counts = counting_function(
            data=data,
            positions=coords,
            size=size,
            localizer=localizer,
            crop_op=crop_op,
        )
        logger.debug(
            f"Finished counting index {index}: {len(peaks)} peaks found"
        )
        loc_out = [index + list(o) for o in peaks]
        count_out = counts
        droplets_out = [index + list(o) for o in coords]
        logger.debug(f"Added index {index} to {len(peaks)} peaks")

        try:
            df = table_function(droplets_out, counts)
        except Exception as e:
            df = None
            logger.error(f"Making dataframe failed: {e}")
        return loc_out, count_out, droplets_out, df


def get_global_coordinates_from_well_coordinates(
    napari_center: tuple, fluo, size
):
    *chip_index, y, x = napari_center
    peaks = get_global_peaks(
        fluo_data=fluo[int(chip_index)], center=(y, x), size=size
    )
    peaks_with_chip_index = [
        add_chip_index_to_coords(p, chip_index) for p in peaks
    ]
    return peaks_with_chip_index


def get_peaks_per_frame(
    stack3d: np.ndarray,
    dif_gauss_sigma: tuple = (1, 3),
    op=get_peaks,
    **kwargs,
):
    """Counts particles in the timelapse"""
    image_ref = gdif(stack3d[0], dif_gauss_sigma)
    thr = 5 * image_ref.std()
    return list(map(partial(op, threshold_abs=thr, **kwargs), stack3d))


def get_peaks_timelapse_all_wells(
    stack: np.ndarray, centers: list, size: int, plot: bool = 0
):
    n_peaks = []
    for c in centers:
        print(".", end="")
        well = cropNd(stack, c["center"], size)
        n_peaks.append(get_peaks_per_frame(well, plot=plot))
    return n_peaks


def main(
    aligned_path: str,
    save_path_csv: str = "",
    gaussian_difference_filter: tuple = (3, 5),
    threshold: float = 2,
    min_distance: float = 5,
    force=False,
    poisson=True,
    **kwargs,
):
    """
    Reads the data and saves the counting table
    Parameters:
    ===========
    aligned_path: str
        path to a tif stack with 3 layers: brightfield, fluorescence and labels
    save_path_csv: str
        path to csv file for the table
    gaussian_difference_filter: tuple(2), default (3,5)
        sigma values for gaussian difference filter
    threshold: float
        Detection threshold, default 2
    min_distance: float
        Minimal distance in pixel between the detecions
    force: bool
        If True, overrides existing csv file.
    **kwargs:
        Anything you want to include as additional column in the table, for example, concentration.
    """

    logger.info(f"anchor-droplet-chip {__version__}")

    if not save_path_csv.endswith(".csv"):
        save_path_csv = aligned_path.replace(".tif", "-counts.csv")
        logger.warning(
            f"No valid path for csv provided, using {save_path_csv}"
        )

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s : %(message)s"
    )
    ff = logging.FileHandler(save_path_csv.replace(".csv", ".log"))
    ff.setFormatter(formatter)
    logger.addHandler(ff)

    try:
        pathlib.Path(save_path_csv).touch(exist_ok=force)
    except Exception as e:
        logger.error("File exists! Use --force to overwrite.")
        exit(1)
    logger.info(f"Reading {aligned_path}")
    bf, fluo, mask = imread(aligned_path)
    logger.info(f"Data size: 3 x {bf.shape}")
    logger.info(f"Counting the cells inside {len(np.unique(mask)) - 1} wells")
    table = get_cell_numbers(
        multiwell_image=fluo,
        labels=mask,
        threshold_abs=threshold,
        min_distance=min_distance,
        dif_gauss_sigma=gaussian_difference_filter,
        bf=bf,
        plot=False,
        **kwargs,
    )
    table.to_csv(save_path_csv, index=None)
    logger.info(f"Saved table to {save_path_csv}")
    if poisson:
        try:
            logger.info("Fitting Poisson")
            _lambda = fit_poisson(
                table.n_cells,
                save_fig_path=save_path_csv.replace(".csv", ".png"),
            )
            logger.info(f"Mean number of cells: {_lambda:.2f}")
        except RuntimeError as e:
            logger.warning(f"no poisson fit due to {e}")
    return


if __name__ == "__main__":
    fire.Fire(main)
