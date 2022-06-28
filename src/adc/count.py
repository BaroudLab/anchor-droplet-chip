import logging
import pathlib
from functools import partial
from importlib.metadata import version, PackageNotFoundError

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from tifffile import imread

from adc.fit import poisson as fit_poisson

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s : %(message)s"
)
logger = logging.getLogger("adc.count")

try:
    __version__ = version("adc")
except PackageNotFoundError:
    # package is not installed
    __version__ = "Unknown"

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

    def get_n_peaks(i):

        if bf is None:
            return get_peak_number(
                multiwell_image[props[i].slice],
                plot=plot,
                dif_gauss_sigma=dif_gauss_sigma,
                threshold_abs=threshold_abs,
                min_distance=min_distance,
                title=props[i].label,
            )
        else:
            return get_peak_number(
                multiwell_image[props[i].slice],
                plot=plot,
                dif_gauss_sigma=dif_gauss_sigma,
                threshold_abs=threshold_abs,
                min_distance=min_distance,
                title=props[i].label,
                bf_crop=bf[props[i].slice],
                return_std=True,
            )

    n_cells = list(map(get_n_peaks, range(labels.max())))
    # return n_cells, props
    return pd.DataFrame(
        [
            {
                "label": prop.label,
                "x": prop.centroid[1],
                "y": prop.centroid[0],
                "n_cells": n_cell[0],
                **kwargs,
            }
            for prop, n_cell in zip(props, n_cells)
        ]
    )


def crop(stack: np.ndarray, center: tuple, size: int):
    im = stack[
        :,
        int(center[0]) - size // 2 : int(center[0]) + size // 2,
        int(center[1]) - size // 2 : int(center[1]) + size // 2,
    ]
    return im


def gdif(array2d, dif_gauss_sigma=(1, 3)):
    array2d = array2d.astype("f")
    return ndi.gaussian_filter(
        array2d, sigma=dif_gauss_sigma[0]
    ) - ndi.gaussian_filter(array2d, sigma=dif_gauss_sigma[1])


def get_peak_number(
    crop2d,
    dif_gauss_sigma=(1, 3),
    min_distance=3,
    threshold_abs=5,
    plot=False,
    title="",
    bf_crop=None,
    return_std=False,
):
    image_max = gdif(crop2d, dif_gauss_sigma)
    peaks = peak_local_max(
        image_max, min_distance=min_distance, threshold_abs=threshold_abs
    )
    logger.debug(f"found {len(peaks)} cells")
    if plot:
        if bf_crop is None:
            fig, ax = plt.subplots(1, 2, sharey=True)
            ax[0].imshow(crop2d)
            ax[0].set_title(f"raw image {title}")
            ax[1].imshow(image_max)
            ax[1].set_title("Filtered + peak detection")
            ax[1].plot(peaks[:, 1], peaks[:, 0], "r.")
            plt.show()
        else:
            fig, ax = plt.subplots(1, 3, sharey=True)

            ax[0].imshow(bf_crop, cmap="gray")
            ax[0].set_title(f"BF {title}")

            ax[1].imshow(crop2d, vmax=crop2d.mean() + 2 * crop2d.std())
            ax[1].set_title(f"raw image {title}")

            ax[2].imshow(image_max)
            ax[2].set_title(
                f"Filtered + {len(peaks)} peaks (std {image_max.std():.2f})"
            )
            ax[2].plot(peaks[:, 1], peaks[:, 0], "r.")
            plt.show()

    if return_std:
        return len(peaks), crop2d.std()
    else:
        return (len(peaks),)


def get_peaks_per_frame(stack3d, dif_gauss_sigma=(1, 3), **kwargs):
    image_ref = gdif(stack3d[0], dif_gauss_sigma)
    thr = 5 * image_ref.std()
    return list(
        map(partial(get_peak_number, threshold_abs=thr, **kwargs), stack3d)
    )


def get_peaks_all_wells(stack, centers, size, plot=0):
    n_peaks = []
    for c in centers:
        print(".", end="")
        well = crop(stack, c["center"], size)
        n_peaks.append(get_peaks_per_frame(well, plot=plot))
    return n_peaks


def main(
    aligned_path: str,
    save_path_csv: str = '',
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

    logger.info(f'anchor-droplet-chip {__version__}')


    if not save_path_csv.endswith(".csv"):
        save_path_csv = aligned_path.replace('.tif', '-counts.csv')
        logger.warning(f"No valid path for csv provided, using {save_path_csv}")

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s : %(message)s"
    )
    ff = logging.FileHandler(save_path_csv.replace(".csv", ".log"))
    ff.setFormatter(formatter)
    logger.addHandler(ff)

    try:
        pathlib.Path(save_path_csv).touch(exist_ok=force)
    except Exception as e:
        logger.error(
            "No path to save the table! Please provide a valid .csv path!"
        )
        raise e
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
        logger.info("Fitting Poisson")
        _lambda = fit_poisson(
            table.n_cells, save_fig_path=save_path_csv.replace(".csv", ".png")
        )
        logger.info(f"Mean number of cells: {_lambda:.2f}")
    return


if __name__ == "__main__":
    fire.Fire(main)
