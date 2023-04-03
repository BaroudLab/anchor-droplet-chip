import json
import logging
import os

import dask.array as da
import nd2
import pandas as pd

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

    return None


def read_zarr(path):
    print(f"read_zarr {path}")

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
        colormap = info["colormap"]
    except Exception as e:
        logger.debug("no info found")
        colormap = None

    try:
        name = info["name"]
    except KeyError:
        print("name not found")
        name = [os.path.basename(path)] * datasets[0].shape[channel_axis]
    except Exception as e:
        print("name exception", e.args)
        name = os.path.basename(path)
    meta = {"path": path}

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
            },
            "image",
        )
    ]

    if os.path.exists(det_path := os.path.join(path, ".detections.csv")):
        try:
            table = pd.read_csv(det_path, index_col=0)
            output.append(
                (
                    table[["axis-0", "axis-1", "axis-2"]].values,
                    {
                        "name": "detections",
                        "face_color": "#ffffff00",
                        "edge_color": "#ff007f88",
                        "size": 20,
                        "metadata": {"path": det_path},
                    },
                    "points",
                )
            )
        except Exception as e:
            print(f"no detections found: {e}")
    else:
        print(f"{det_path} doesn't exists")

    if os.path.exists(det_path := os.path.join(path, ".droplets.csv")):
        try:
            table = pd.read_csv(det_path, index_col=0)
            if os.path.exists(
                count_path := os.path.join(path, ".counts.json")
            ):
                with open(count_path) as fp:
                    counts = json.load(fp)
            else:
                counts = None
            output.append(
                (
                    table[["axis-0", "axis-1", "axis-2"]].values,
                    {
                        "name": "droplets",
                        "face_color": "#ffffff00",
                        "edge_color": "#55aa0088",
                        "size": 300,
                        "metadata": {"path": det_path},
                        "text": counts,
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
