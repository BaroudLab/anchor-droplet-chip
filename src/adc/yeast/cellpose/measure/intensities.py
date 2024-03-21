import logging
import os
import re
import shutil
import time
from typing import List, Union

import dask.array as da
import numpy as np
import pandas as pd
import tifffile as tf
from napari.layers import Image, Labels
from skimage.measure import regionprops_table

from adc.yeast.plot.tools import Layer, filters, read_data

from . import plot

log = logging.getLogger(__name__)

GFP_POSITIVE_THRESHOLD = 140
RE_TITLE = r"pos\d+"


def get_title(prefix, re_title=RE_TITLE):
    try:
        title = re.search(re_title, prefix).group()
    except AttributeError:
        log.error("unable to to find pos## in the prefix")
        title = prefix
    log.debug(f"title will be `{title}`")
    return title


def analyse_all_layers(
    prefix: str = "",
    filter_tif: Union[str, None] = "filter.tif",
    data_path: str = "stack.tif",
    cp_path: str = "cellpose.tif",
    frames_per_hour=2,
    backup_folder="backup",
    reader=read_data,
):
    title = get_title(prefix)

    prefix = str(prefix)
    save_dir = os.path.dirname(prefix).replace("input", "output")
    log.info(f"output directory will be `{save_dir}`")
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
        log.warning(f"output directory not empty `{save_dir}`")
        if backup_folder:
            backup_path = "_".join(
                [save_dir, backup_folder, time.strftime("%Y%m%d-%H%M%S")]
            )
            os.makedirs(backup_path, exist_ok=True)
            shutil.move(save_dir, backup_path)
            log.warning(f"Backed up to {backup_path}")
        else:
            log.error(
                f"Data exists! Please remove or indicate backup_folder to backup the existing data"
            )
            return

    log.info(f"filters: {filters}")
    os.makedirs(save_dir, exist_ok=True)
    bf_layer, mcherry_layer, gfp_layer = reader(
        path := os.path.join(prefix, data_path)
    )
    log.info("read stack")
    cellpose_layer = Layer(
        tf.imread(os.path.join(prefix, cp_path)),
        name="cellpose",
        kind="labels",
    )
    log.info("read cellpose")

    if filter_tif is not None:
        filter_layer = Layer(
            tf.imread(os.path.join(prefix, filter_tif)) - 1,
            name="filter",
            kind="labels",
        )
        log.info(f"read filter {filter_layer.data.shape}")
    else:
        filter_layer = Layer(
            np.ones_like(mcherry_layer.data, dtype=bool),
            name="filter",
            kind="labels",
        )
        log.info(f"generate blank filter {filter_layer.data.shape}")

    cellpose_layer_filtered = Layer(
        cellpose_layer.data * (filter_layer.data),
        name="cellpose",
        kind="labels",
        metadata=dict(
            source={
                "filter": filter_layer.source.path,
                "cellpose": cellpose_layer.source.path,
            },
            op="filter * cellpose",
        ),
    )
    log.info("apply filter layer to cellpose stack")

    log.info("processing table")
    df = get_table(
        fluo_layers=[mcherry_layer, gfp_layer],
        label_layers=[
            cellpose_layer_filtered,
        ],
        path=path,
    )
    log.info("table with regonprops recovered across 2 channels and 3 labels")

    df.loc[:, "hours"] = df.frame / frames_per_hour
    log.info("hours added")

    log.info("filter cellpose")
    df = get_good_cellpose_labels(df, filters=filters)
    df.loc[df["mask"] == "cyto - nuc", "manual_filter"] = df.loc[
        df["mask"] == "cellpose", "manual_filter"
    ].values

    log.info("filter gfp")
    df = filter_gfp_intensity(df, filters=filters)

    log.info("label gfp positive cells")
    df = get_positive_gfp(df)

    log.info("filter gfp empty frames")
    df.loc[(df["mean_intensity"] == 0), "manual_filter"] = 0

    log.info("apply filters: getting filtered table")
    filt_df = df.query("manual_filter == 1")

    log.info("pivot table")

    save_path = path.replace(".tif", ".csv")
    assert save_path != path
    log.info(f"saving everything into {save_dir}")
    df.to_csv(ppp := os.path.join(save_dir, "table.csv"))
    log.info(f"table saved `{ppp}`")

    filt_df.to_csv(ppp := os.path.join(save_dir, "filt_table.csv"))
    log.info(f"filt table saved `{ppp}`")
    log.info(f"pivot table saved `{ppp}`")

    try:
        log.info(f"plot top 10 px")
        plot.plot_10(
            filt_df,
            title=title,
            save_path=(
                ppp := os.path.join(save_dir, "top10px_intensity_ratio.png")
            ),
        )
        log.info(f"plot saved `{ppp}`")

        log.info(f"plot saved `{ppp}`")

        log.info(f"plot max_intensity_ratio")
        plot.plot_max(
            filt_df,
            title=title,
            save_path=(
                ppp := os.path.join(save_dir, "max_intensity_ratio.png")
            ),
        )
        log.info(f"plot saved `{ppp}`")

        log.info(f"plot all_measurements")
        plot.plot_table(
            filt_df,
            title=title,
            save_path=(ppp := os.path.join(save_dir, "all_measurements.png")),
        )
        log.info(f"plot saved `{ppp}`")
    except Exception as e:
        print(e)
    finally:
        return filt_df


def top10px(regionmask, intensity):
    return np.sort(np.ravel(intensity[regionmask]))[-10:].mean()


def top20px(regionmask, intensity):
    return np.sort(np.ravel(intensity[regionmask]))[-20:].mean()


def top100px(regionmask, intensity):
    return np.sort(np.ravel(intensity[regionmask]))[-100:].mean()


def top1percent(regionmask, intensity):
    """Select 1% of the brightest pixels and return mean intensity"""
    return np.sort(vector := np.ravel(intensity[regionmask]))[
        -(len(vector) // 100) :
    ].mean()

def top1percent_ratio(regionmask, intensity):
    """Select 1% of the brightest pixels and return mean intensity as a ratio of the total mean intensity"""
    total_mean = intensity[regionmask].mean()
    top1percent_mean = np.sort(vector := np.ravel(intensity[regionmask]))[
        -(len(vector) // 100) :
    ].mean()
    return top1percent_mean / mean_intensity


def cyto_wo100px(regionmask, intensity):
    log.info(f"mask size {regionmask.sum()}")
    vector = np.sort(np.ravel(intensity[regionmask]))
    if (lll := len(vector)) > 100:
        return vector[:-100].mean()
    else:
        log.warning(
            f"area is smaller than 100 px ({lll}), returning lower {lll // 2} intensities"
        )
        return vector[: (lll // 2)].mean()


def get_table(
    fluo_layers: List[Image],
    label_layers: List[Labels],
    properties: List[str] = [
        "label",
        "centroid",
        "area",
        "mean_intensity",
        "max_intensity",
    ],
    path: str = "",
    extra_properties=(top10px, top1percent),
):
    intensities = [
        {
            **(
                props := regionprops_table(
                    label_image=mask.data,
                    intensity_image=fluo.data.compute()
                    if isinstance(fluo.data, da.Array)
                    else fluo.data,
                    properties=properties,
                    extra_properties=extra_properties,
                )
            ),
            "path": path,
            "channel": fluo.name,
            "mask": mask.name,
        }
        for mask in label_layers
        for fluo in fluo_layers
        if fluo.data.mean() > 0
    ]
    df = pd.concat([pd.DataFrame(i) for i in intensities], ignore_index=True)
    df = df.rename(
        columns={"centroid-0": "frame", "centroid-1": "y", "centroid-2": "x"}
    )
    return df


def get_good_cellpose_labels(df, filters=filters):
    table = df.copy()
    table.loc[table["mask"] == "cellpose", "manual_filter"] = np.logical_and(
        table[table["mask"] == "cellpose"]["area"]
        > filters["filters"]["cyto"]["area"]["min"],
        table[table["mask"] == "cellpose"]["area"]
        < filters["filters"]["cyto"]["area"]["max"],
    )
    # table.loc[table["mask"] == "cellpose","manual_filter"] = np.logical_and(table.loc[table["mask"] == "cellpose","manual_filter"],  table[table["mask"] == "cellpose"]["mean_intensity"] < filters["filters"]["cyto"]["mean_intensity"]["mCherry"]["max"])
    # table.loc[table["mask"] == "cellpose","manual_filter"] = np.logical_and(table.loc[table["mask"] == "cellpose","manual_filter"],  table[table["mask"] == "cellpose"]["mean_intensity"] > filters["filters"]["cyto"]["mean_intensity"]["mCherry"]["min"])
    return table


def filter_gfp_intensity(df, filters=filters):
    table = df.copy()
    table.loc[
        (df["mask"] == "cellpose") & (df.channel == "GFP"), "manual_filter"
    ] = (
        table[(df["mask"] == "cellpose") & (df.channel == "GFP")][
            "mean_intensity"
        ]
        > filters["filters"]["cyto"]["mean_intensity"]["GFP"]["min"]
    )
    return table


def get_positive_gfp(df, thr=GFP_POSITIVE_THRESHOLD):
    table = df.copy()
    table.loc[:, "GFP_positive"] = table.mean_intensity > thr
    return table
