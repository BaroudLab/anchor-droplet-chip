import logging
import sys
from importlib.metadata import PackageNotFoundError, version
from os import PathLike
from typing import Tuple

import fire
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb
from skimage.measure import label
from tifffile import imread, imwrite

import imreg_dft as reg

try:
    __version__ = version("anchor-droplet-chip")
except PackageNotFoundError:
    # package is not installed
    __version__ = "Unknown"
logger = logging.getLogger("adc.align")
logger.setLevel(logging.INFO)


GREY = np.array([np.arange(256)] * 3, dtype="uint8")
RED = np.array(
    [np.arange(256), np.zeros((256,)), np.zeros((256,))], dtype="uint8"
)
GREEN = np.array(
    [np.zeros((256,)), np.arange(256), np.zeros((256,))], dtype="uint8"
)
BLUE = np.array(
    [np.zeros((256,)), np.zeros((256,)), np.arange(256)], dtype="uint8"
)

META_ALIGNED = {
    "ImageJ": "1.53c",
    "images": 3,
    "channels": 3,
    "hyperstack": True,
    "mode": "composite",
    "unit": "",
    "loop": False,
    "min": 1878.0,
    "max": 30728.0,
    "Ranges": (1878.0, 30728.0, 430.0, 600.0, 0.0, 501.0),
    "LUTs": [GREY, GREEN, BLUE],
}

CONSTRAINTS = {  # [mean, std]
    "scale": [1, 0.2],
    "tx": [0, 50],
    "ty": [0, 50],
    "angle": [0, 30],
}


class PaddingError(Exception):
    pass


def align_stack(
    data: np.ndarray,
    template16: np.ndarray,
    mask2: np.ndarray,
    plot: bool = False,
    path_to_save: str = None,
    binnings: Tuple = (2, 16, 2),
    constraints: dict = CONSTRAINTS,
):
    """
    stack should contain two channels: bright field and fluorescence.
    BF will be binned 8 times and registered with template8 (aligned BF).
    When the transformation verctor will be applied to the original data and stacked with the mask.
    The output stack is of the same size as mask.
    The resulting 3-layer stack will be returned and also saved with suffix ".aligned.tif"

    Parameters:
    ===========
    data : np.ndarray
        Bright-field + fluorescence stack with the shape (2, Y, X)
    template16 : np.ndarray
        binned template of aligned bright-filed image of the chip
    mask2 : np.ndarray
        Labelled mask which you try to align with the data
    plot : bool, optional
        Plot results
    path_to_save :str
        Path to the .tif file to save aligned bright-filed + fluo + mask
    binnings : tuple(data, template, mask)
        Bright-field channel will be binned to match the scale of the template.
        The transformation vector will then be upscaled back to transform the original data.
        The aligned data will be binned to match the scale of the mask
    metadata: dict, optional
        ImageJ tif metadata.
        Default:
            META_ALIGNED = {'ImageJ': '1.53c',
                'images': 3,
                'channels': 3,
                'hyperstack': True,
                'mode': 'composite',
                'unit': '',
                'loop': False,
                'min': 1878.0,
                'max': 30728.0,
                'Ranges': (1878.0, 30728.0, 430.0, 600.0, 0.0, 501.0),
                'LUTs': [grey, green, blue]
            }

    Returns
    -------
    aligned_stack : np.ndarray,
    tvec : dict
        aligned_stack: bf+fluo+mask,
        tvec: transform dictionary

    """

    bf, tritc = data[:2]
    stack_temp_scale = binnings[1] // binnings[0]
    mask_temp_scale = binnings[1] // binnings[2]
    stack_mask_scale = binnings[2] // binnings[0]

    f_bf = bf[::stack_temp_scale, ::stack_temp_scale]

    tvec8 = get_transform(f_bf, template16, constraints, plot=plot)
    plt.show()
    tvec = scale_tvec(tvec8, mask_temp_scale)
    logger.info(f"Found transform: {tvec}")
    try:
        logger.info(f"Applying the transform to the brightfield channel")
        aligned_tritc = unpad(
            transform(tritc[::stack_mask_scale, ::stack_mask_scale], tvec),
            mask2.shape,
        )
        logger.info(f"Applying the transform to the fluorescence channel")
        aligned_bf = unpad(
            transform(bf[::stack_mask_scale, ::stack_mask_scale], tvec),
            mask2.shape,
        )
    except ValueError as e:
        print("stack_mask_scale: ", stack_mask_scale)
        logger.error(e.args)
        raise e

    if plot:
        plt.figure(dpi=300)
        plt.imshow(
            aligned_tritc,
            cmap="gray",
        )  # vmax=aligned_tritc.max()/5)
        plt.colorbar()
        plt.show()

        saturated_tritc = aligned_tritc.copy()
        saturated_tritc[saturated_tritc > 500] = 500
        plt.figure(dpi=300)
        plt.imshow(
            label2rgb(label(mask2), to_8bits(saturated_tritc), bg_label=0)
        )
        plt.show()

    aligned_stack = np.stack((aligned_bf, aligned_tritc, mask2)).astype(
        "uint16"
    )

    return aligned_stack, tvec


def get_transform(
    image,
    template,
    constraints,
    plot=False,
    pad_ratio=1.5,
    figsize=(10, 5),
    dpi=300,
):
    """
    Pads image and template, registers and returns tvec
    """
    print(f"padding to image.shape {image.shape} * pad_ratio {pad_ratio}")
    s = increase(image.shape, pad_ratio)
    print(f"new increased shape: {s}")

    print(f"padding template: {template.shape}")
    padded_template = pad(template, s)
    print(f"padded template: {padded_template.shape}")

    print(f"padding image: {image.shape}")
    padded_image = pad(image, s)
    print(f"padded image: {padded_image.shape}")

    tvec = register(padded_image, padded_template, constraints)
    print(f"Found transform: {tvec}")
    if plot:
        aligned_bf = unpad(tvec["timg"], template.shape)
        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(aligned_bf, cmap="gray")
    return tvec


def register(image, template, constraints):
    """
    Register image towards template
    Return:
    tvec:dict
    """
    assert np.array_equal(
        image.shape, template.shape
    ), f"unequal shapes {(image.shape, template.shape)}"
    return reg.similarity(template, image, constraints=constraints)


def pad(image: np.ndarray, to_shape: tuple = None, padding: tuple = None):
    """
    Pad the data to desired shape
    """
    print(f"padding image {image.shape} to shape {to_shape}")
    if padding is None:
        try:
            padding = calculate_padding(image.shape, to_shape)
        except PaddingError as e:
            new_to_shape = tuple(
                max(i, j) for i, j in zip(image.shape, to_shape)
            )
            logger.error(f"Error padding {e.args} try to_shape {new_to_shape}")

    try:
        padded = np.pad(image, padding, "edge")
    except TypeError as e:
        logger.error(f"padding {padding} failed: {e.args}")
        raise e
    return padded


def unpad(image: np.ndarray, to_shape: tuple = None, padding: tuple = None):
    """
    Remove padding to get desired shape
    """
    if any(np.array(image.shape) - np.array(to_shape) < 0):
        logger.warning(
            f"unpad warning: image.shape {image.shape} is within to_shape {to_shape}"
        )
        image = pad(image, np.array((image.shape, to_shape)).max(axis=0))
        logger.info(f"new image shape after padding {image.shape}")
    if padding is None:
        padding = calculate_padding(to_shape, image.shape)

    y = [padding[0][0], -padding[0][1]]
    if y[1] == 0:
        y[1] = None
    x = [padding[1][0], -padding[1][1]]
    if x[1] == 0:
        x[1] = None
    return image[y[0] : y[1], x[0] : x[1]]


def calculate_padding(shape1: tuple, shape2: tuple):
    """
    Calculates padding to get shape2 from shape1
    Return:
    2D tuple of indices
    """
    dif = np.array(shape2) - np.array(shape1)
    if not all(dif >= 0):
        raise PaddingError(
            f"Shape2 {shape2} must be bigger than shape1 {shape1}"
        )
    mid = dif // 2
    rest = dif - mid
    return (mid[0], rest[0]), (mid[1], rest[1])


def scale_tvec(tvec, scale=8):
    """
    Scale up transform vector from imreg_dft
    """
    tvec_8x = tvec.copy()
    tvec_8x["tvec"] = tvec["tvec"] * scale
    try:
        tvec_8x["timg"] = None
    except KeyError:
        pass
    finally:
        return tvec_8x


def transform(image, tvec):
    """
    apply transform
    """
    print(f"transform {image.shape}")
    fluo = reg.transform_img_dict(image, tvec)
    return fluo.astype("uint")


def main(
    data_path: PathLike,
    template_path: PathLike,
    mask_path: PathLike,
    binnings: tuple = (2, 16, 2),
    path_to_save: PathLike = "",
    sx=CONSTRAINTS["tx"][1],
    sy=CONSTRAINTS["ty"][1],
    cx=CONSTRAINTS["tx"][0],
    cy=CONSTRAINTS["ty"][0],
    metadata: dict = META_ALIGNED,
):
    """
    reads the data from disk and runs alignment
    all paths should be .tif
    Params:
    -------
    data_path, PathLike
        path to tif with brightfield and fluorescence channels
    template_path, PathLike
        path to the brightfield template, normally binned.
    mask_path, PathLike
        path to the labels tif
    binnings, tuple
        data, template, mask binnings. Default: (2,16,2)
    path_to_save, PathLike
        path to save the aligned stack
    sx, float
        standard deviation for x displacement in pixels
    sy, float
        standard deviation for y displacement in pixels
    cx, float
        mean value for x displacement in pixels
    cy, float
        mean value for y displacement in pixels



    """
    logging.basicConfig(level="INFO")
    fh = logging.FileHandler(data_path.replace(".tif", "-aligned.log"))
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s : %(message)s"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"anchor-droplet-chip {__version__}")

    if not path_to_save:
        path_to_save = data_path.replace(".tif", "-aligned.tif")
        logger.warning(f"No path_to_save provided, using {path_to_save}")
    stack = imread(data_path)
    logger.info(f"Open `{data_path}` with the shape {stack.shape}")
    template = imread(template_path)
    logger.info(f"Open `{template_path}` with the shape {template.shape}")
    mask = imread(mask_path)
    logger.info(f"Open `{mask_path}` with the shape {mask.shape}")
    logger.info(f"Using binnings {binnings}")

    constraints = CONSTRAINTS.copy()
    constraints["tx"] = (cx, sx)
    constraints["ty"] = (cy, sy)
    logger.info(f"Using constraints {constraints}")

    logger.info(f"Start aligning `{data_path}`")
    try:
        aligned_stack, tvec = align_stack(
            stack,
            template,
            mask,
            path_to_save=path_to_save,
            binnings=binnings,
            constraints=constraints,
        )
    except Exception as e:
        logger.error(f"Alignment failed due to {e.args}")
        raise e
    logger.info(f"Finished aligning. tvec: {tvec}")
    imwrite(path_to_save, aligned_stack, imagej=True, metadata=metadata)
    logger.info(f"Saved aligned stack {path_to_save}")
    sys.stdout.write(path_to_save)
    sys.exit(0)


def to_8bits(array2d: np.ndarray):
    """normalize to 0-255 uint8"""
    a = array2d.astype("f")
    new_array = (a - a.min()) * 255 / (a.max() - a.min())
    return new_array.astype("uint8")


def increase(shape, increase_ratio):
    assert increase_ratio > 1
    shape = np.array(shape)
    return tuple((shape * increase_ratio).astype(int))


if __name__ == "__main__":
    fire.Fire(main)
