import json
import os

import dask
import nd2
import pandas as pd
import tifffile as tf

from ._align_widget import DROPLETS_CSV_SUFFIX, DROPLETS_LAYER_PROPS
from ._count_widget import (
    COUNTS_JSON_SUFFIX,
    COUNTS_LAYER_PROPS,
    DETECTION_CSV_SUFFIX,
    DETECTION_LAYER_PROPS,
)


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

    if path.endswith(".tif"):
        return read_tif

    if path.endswith(".csv"):
        return read_csv

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


def read_tif(path):
    data = tf.TiffFile(path)
    arr = data.asarray()
    colormap = (
        ["gray", "yellow"]
        if all([a in path for a in ["BF", "TRITC"]])
        else None
    )
    try:
        channel_axis = (
            arr.shape.index(data.imagej_metadata["channels"])
            if data.is_imagej
            else None
        )
    except (ValueError, KeyError):
        channel_axis = None

    try:
        ranges = data.imagej_metadata["Ranges"]
        contrast_limits = [
            [ranges[2 * i], ranges[2 * i + 1]] for i in range(len(ranges) // 2)
        ]
    except (ValueError, KeyError):
        contrast_limits = None

    out = [
        (
            arr,
            {
                "channel_axis": channel_axis,
                "metadata": {"path": path},
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

    try:
        attrs = json.load(open(os.path.join(path, ".zattrs")))
        info = attrs["multiscales"]["multiscales"][0]
    except Exception as e:
        raise e

    dataset_paths = [os.path.join(path, d["path"]) for d in info["datasets"]]
    datasets = [dask.array.from_zarr(p) for p in dataset_paths]

    try:
        channel_axis = info["channel_axis"]
        print(f"found channel axis {channel_axis}")
    except KeyError:
        channel_axis = None

    except Exception as e:
        raise e

    try:
        contrast_limits = info["lut"]
    except KeyError:
        contrast_limits = None

    try:
        colormap = info["colormap"]
    except KeyError:
        colormap = None

    try:
        name = info["name"]
    except KeyError:
        print("name not found")
        name = [os.path.basename(path)] * datasets[0].shape[channel_axis]
    except Exception as e:
        print("name exception", e.args)
        name = os.path.basename(path)

    output = [
        (
            datasets,
            {
                "channel_axis": channel_axis,
                "colormap": colormap,
                "contrast_limits": contrast_limits,
                "name": name,
                "metadata": {"path": path},
            },
            "image",
        )
    ]

    if os.path.exists(det_path := os.path.join(path, DETECTION_CSV_SUFFIX)):
        try:
            table = pd.read_csv(det_path, index_col=0)
            output.append(
                (
                    table[["axis-0", "axis-1", "axis-2"]].values,
                    {"metadata": {"path": det_path}, **DETECTION_LAYER_PROPS},
                    "points",
                )
            )
        except Exception as e:
            print(f"no detections found: {e}")
    else:
        print(f"{det_path} doesn't exists")

    if os.path.exists(det_path := os.path.join(path, DROPLETS_CSV_SUFFIX)):
        try:
            table = pd.read_csv(det_path, index_col=0)
            if os.path.exists(
                count_path := os.path.join(path, COUNTS_JSON_SUFFIX)
            ):
                with open(count_path) as fp:
                    counts = json.load(fp)
            else:
                counts = None
            output.append(
                (
                    table[["axis-0", "axis-1", "axis-2"]].values,
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
