import json
import logging
import os
from functools import partial

import dask.array as da
import h5py
import nd2
import numpy as np
import pandas as pd
import tifffile as tf

from ._align_widget import DROPLETS_CSV_SUFFIX, DROPLETS_LAYER_PROPS
from ._count_widget import (
    COUNTS_JSON_SUFFIX,
    COUNTS_LAYER_PROPS,
    DETECTION_CSV_SUFFIX,
    DETECTION_LAYER_PROPS,
    TABLE_NAME,
)

logger = logging.getLogger(__name__)


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    logger.debug("napari_get_reader starts")
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    if path.endswith(".nd2"):
        return read_nd2

    if path.endswith(".zarr"):
        return read_zarr

    if "ilastik" in path and not "cellpose" in path:
        logger.debug("ilastik in path, no cellpose")
        return read_ilastik_labels_tif

    if path.endswith(".tif"):
        if "POINT 00001" in path:
            logger.info("reading muvicyte")
            return read_muvicyte_tif

        logger.debug("ends with .tif")

        if "P=" in path or "pos" in path and not "ilastik" in path:
            logger.debug("pos in path, no ilastik")

            if "CP_labels" in path or "cellpose" in path or "cyto" in path:
                logger.debug("cellpose in path, return read_cellpose_labels")
                return read_cellpose_labels
            logger.debug("no cellpose in path, return read_tif_yeast")

            if "filter.tif" in path:
                return partial(read_ilastik_labels_tif, name="filter")

            return read_tif_yeast
        logger.debug("no pos in path, return read_tif")

        return read_tif

    if "Simple Segmentation_" in path and path.endswith(".tiff"):
        return read_ilastik_labels_tif

    if path.endswith(".csv"):
        return read_csv

    return None


def read_ilastik_labels_tif(path, name="ilastik"):
    data = tf.imread(path)
    labels = data - 1
    return [(labels, dict(name=name, metadata={"path": path}), "labels")]


def read_cellpose_labels(path):
    # print("reading cellpose labels")
    labels = tf.imread(path)
    try:
        properties = read_cellpose_seg_table(path.replace(".tif", ".csv"))
    except Exception as e:
        print(f"Failed loading csv: {e}")
        properties = None
    return [(labels, dict(name="cellpose", properties=properties), "labels")]


def read_cellpose_seg_table(table_path: str):
    try:
        table = pd.read_csv(
            table_path,
            index_col=0,
            # usecols=("index",
            #     "label",
            #     "area",
            #     "mean_intensity",
            #     "eccentricity",
            #     "solidity",
            #     "frame"
            # )
        )
    except ValueError:
        table = pd.read_csv(
            table_path,
            index_col=0,
            # usecols=(
            #     "index",
            #     "label",
            #     "area",
            #     "mean_intensity",
            #     "eccentricity",
            #     "frame"
            # )
        )
    return table.sort_index()


def get_yeast_reader(path):
    if path.endswith("*P=*.tif"):
        return read_tif_yeast

    if path.endswith(".h5"):
        return read_ilastik

    return None


def read_csv(path, props=DROPLETS_LAYER_PROPS):
    data = pd.read_csv(path, index_col=0)
    return [
        (
            data.values,
            props,
            "points",
        )
    ]


def read_tif_yeast(path):
    data = tf.TiffFile(path)
    z = data.aszarr()
    d = da.from_zarr(z)
    colormap = ["gray", "magenta", "green"]
    names = ["BF", "mCherry", "GFP"]
    assert d.ndim == 4, f"Expected 4D TCYX stack, got {d.shape}"
    assert d.shape[1] == len(
        names
    ), f"Expected {len(names)} channels, got {data.shape[1]} with total shape {data.shape}"

    contrast_limits = (None, (90, 600), (90, 600))

    return [
        (
            d,
            {
                "colormap": colormap,
                "name": names,
                "channel_axis": 1,
                "contrast_limits": contrast_limits,
                "metadata": {
                    "dask_data": d,
                    "path": path,
                    "colormap": colormap,
                    "names": names,
                    "sizes": {k: v for k, v in zip("TCYX", d.shape)},
                    "contrast_limits": contrast_limits,
                },
            },
            "image",
        )
    ]


def read_muvicyte_tif(path):
    data = tf.imread(path)
    return [(data, {"name": ["BF","TRITC", "GFP", "labels"], "channel_axis":1}, "image")]


def read_tif(path):
    data = tf.TiffFile(path)
    z = data.aszarr()
    d = da.from_zarr(z)
    if max(d.shape) > 4000:
        arr = [d[..., :: 2**i, :: 2**i] for i in range(4)]
        arr_shape = d.shape
    else:
        arr = d
    arr_shape = d.shape

    colormap = (
        ["gray", "yellow"]
        if all([a in path for a in ["BF", "TRITC"]])
        else None
    )
    try:
        channel_axis = (
            arr_shape.index(data.imagej_metadata["channels"])
            if data.is_imagej
            else None
        )
    except (ValueError, KeyError):
        channel_axis = None

    if data.is_imagej:
        try:
            ranges = data.imagej_metadata["Ranges"]
            logger.debug(f"Ranges: {ranges}")
            contrast_limits = [
                [ranges[2 * i], ranges[2 * i + 1]]
                for i in range(len(ranges) // 2)
            ]
        except KeyError:
            contrast_limits = None

        IJaxes = "TZCYX"
        sizes = {l: s for l, s in zip(IJaxes[-len(d.shape) :], d.shape)}
    else:
        sizes = None

    out = [
        (
            arr,
            {
                "channel_axis": channel_axis,
                "metadata": {
                    "path": path,
                    "dask_data": d,
                    "sizes": sizes,
                    "colormap": colormap,
                    "contrast_limits": contrast_limits,
                },
                "colormap": colormap,
                "contrast_limits": contrast_limits,
            },
            "image",
        )
    ]

    if os.path.exists(ppp := path + DETECTION_CSV_SUFFIX):
        detections = read_csv(ppp, props=DETECTION_LAYER_PROPS)[0]
        out.append(detections)

    if os.path.exists(ppp := path + DROPLETS_CSV_SUFFIX):
        droplets = read_csv(ppp, props=DROPLETS_LAYER_PROPS)[0]
        out.append(droplets)

    if os.path.exists(ppp := path + COUNTS_JSON_SUFFIX):
        with open(ppp) as f:
            counts = read_json(f)
        out.append(
            (droplets[0], {"text": counts, **COUNTS_LAYER_PROPS}, "points")
        )

    return out


def read_json(path):
    return json.load(path)


def read_zarr(path):
    print(f"read_zarr {path}")
    meta = {"path": path}
    try:
        attrs = json.load(open(os.path.join(path, ".zattrs")))
        info = attrs["multiscales"]["multiscales"][0]
        dataset_paths = [
            os.path.join(path, d["path"]) for d in info["datasets"]
        ]
        datasets = [da.from_zarr(p) for p in dataset_paths]
    except Exception as e:
        logger.error(f"Error opening .zattr: {e}")
        datasets = da.from_zarr(path)

    try:
        channel_axis = info["channel_axis"]
        print(f"found channel axis {channel_axis}")
    except Exception as e:
        logger.debug(f"no info found {e}")
        channel_axis = None

    try:
        contrast_limits = info["lut"]
    except Exception as e:
        logger.debug("no info found")
        contrast_limits = None

    try:
        scale = info["scale"]
        logger.debug("scale: %s", scale)
    except Exception as e:
        logger.debug("no scale info found")
        scale = None

    try:
        colormap = info["colormap"]
    except Exception as e:
        logger.debug("no info found")
        colormap = None

    try:
        name = info["name"]
    except KeyError:
        print("name not found")
        name = os.path.basename(path)

    try:
        pixel_size_um = info["pixel_size_um"]
    except (UnboundLocalError, KeyError):
        pixel_size_um = None
    meta["pixel_size_um"] = pixel_size_um

    try:
        if "sizes" in info:
            meta["sizes"] = info["sizes"]
    except UnboundLocalError:
        pass

    try:
        if not datasets[0].shape == tuple(meta["sizes"].values()):
            logger.error(
                f"dataset shape {datasets[0].shape} is not the same as size: {meta['sizes'].values()}"
            )
        else:
            meta["dask_data"] = datasets[0]
    except Exception as e:
        logger.error(f"Error setting dask_data: {e}")

    output = [
        (
            datasets,
            {
                "channel_axis": channel_axis,
                "colormap": colormap,
                "contrast_limits": contrast_limits,
                "name": name,
                "metadata": meta,
                "scale": scale
            },
            "image",
        )
    ]

    if os.path.exists(det_path := os.path.join(path, DETECTION_CSV_SUFFIX)):
        try:
            table = pd.read_csv(det_path, index_col=0)
            output.append(
                (
                    table.values,
                    {"metadata": {"path": det_path}, **DETECTION_LAYER_PROPS},
                    "points",
                )
            )
        except Exception as e:
            print(f"no detections found: {e}")
    else:
        print(f"{det_path} doesn't exists")

    if os.path.exists(droplet_path := os.path.join(path, DROPLETS_CSV_SUFFIX)):
        try:
            droplets_df = pd.read_csv(droplet_path, index_col=0)
            if os.path.exists(
                count_path := os.path.join(path, COUNTS_JSON_SUFFIX)
            ):
                with open(count_path) as fp:
                    counts = json.load(fp)
            else:
                counts = None
            output.append(
                (
                    droplets_df.values,
                    {
                        "metadata": {"path": det_path},
                        "text": counts,
                        **COUNTS_LAYER_PROPS,
                    },
                    "points",
                )
            )
        except Exception as e:
            print(f"no detections found: {e}")
    else:
        print(f"{det_path} doesn't exists")

    if not os.path.exists(
        ppp := os.path.join(os.path.dirname(path), TABLE_NAME)
    ):
        from .count import make_table

        try:
            df = make_table(droplets_out=droplets_df.values, counts=counts)
            df.to_csv(ppp)
            print(f"Saved table to {ppp}")
        except Exception as e:
            print(f"Unable to create table")
    else:
        print("Table found")
    return output


def read_nd2(path):
    print(f"opening {path}")
    data = nd2.ND2File(path)
    print(data.sizes)
    ddata = data.to_dask()
    try:
        pixel_size_um = data.metadata.channels[0].volume.axesCalibration[0]
    except Exception as e:
        print(f"Pixel information unavailable: {e}")
        pixel_size_um = 1
    # colormap = ["gray", "green"]
    try:
        channel_axis = list(data.sizes.keys()).index("C")
        channel_axis_name = "C"
    except ValueError:
        print(f"No channels, {data.sizes}")
        channel_axis = None
        channel_axis_name = None
        # colormap = ["gray"]
    return [
        (
            ddata
            if max(ddata.shape) < 4000
            else [ddata[..., :: 2**i, :: 2**i] for i in range(4)],
            {
                "channel_axis": channel_axis,
                "metadata": {
                    "sizes": data.sizes,
                    "path": path,
                    "dask_data": ddata,
                    "pixel_size_um": pixel_size_um,
                    "channel_axis": channel_axis,
                    "channel_axis_name": channel_axis_name,
                },
            },
            "image",
        )
    ]
