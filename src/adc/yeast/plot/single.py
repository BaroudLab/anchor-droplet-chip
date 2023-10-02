import logging
import os
import re
import shutil
import time
from typing import List, Union

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile as tf

# import plot_tools
import yaml
from napari import Viewer
from napari.layers import Image, Labels
from skimage.measure import regionprops, regionprops_table
from tqdm import tqdm

from .tools import Layer, filters, read_data

# from cellpose import models


log = logging.getLogger(__name__)

GFP_POSITIVE_THRESHOLD = 140
RE_TITLE = r"pos\d+"


def get_title(prefix, re_title=RE_TITLE):
    try:
        title = re.search(re_title, prefix).group()
    except AttributeError:
        log.error("unable to to find pos## in the prefix")
        title = prefix
    # log.info(f"title will be `{title}`")
    return title


def analyse_all_layers(
    prefix: str = "",
    filter_tif: Union[str, None] = "filter.tif",
    data_path: str = "stack.tif",
    cp_path: str = "cellpose.tif",
    frames_per_hour=2,
    backup_folder="backup",
    title_re=RE_TITLE,
    reader=read_data,
):
    title = get_title(prefix)
    log.info(f"title will be `{title}`")

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
        plot_10(
            filt_df,
            title=title,
            save_path=(
                ppp := os.path.join(save_dir, "top10px_intensity_ratio.png")
            ),
        )
        log.info(f"plot saved `{ppp}`")

        log.info(f"plot saved `{ppp}`")

        log.info(f"plot max_intensity_ratio")
        plot_max(
            filt_df,
            title=title,
            save_path=(
                ppp := os.path.join(save_dir, "max_intensity_ratio.png")
            ),
        )
        log.info(f"plot saved `{ppp}`")

        log.info(f"plot all_measurements")
        plot_table(
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
                    extra_properties=(top10px, top20px),
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


def get_positive_gfp(df, thr=GFP_POSITIVE_THRESHOLD):
    table = df.copy()
    table.loc[:, "GFP_positive"] = table.mean_intensity > thr
    return table


def get_gfp_positive_number(df):
    return (
        df.query("channel == 'GFP' and mask == 'cellpose'")
        .groupby("hours")
        .sum()
        .GFP_positive
    )


def count_nuc_number(df):
    return (
        df.query("mask == 'nuclei' and channel == 'mCherry'")
        .groupby("hours")
        .count()["channel"]
    )


def get_cell_number(df):
    return (
        df.query("mask == 'cellpose' and channel=='mCherry'")
        .groupby("hours")
        .count()["frame"]
    )


def plot_10(df, title="", save_path="P=19-top10px.png", backup_folder=""):
    if os.path.exists(save_path):
        if backup_folder:
            backup_path = os.path.join(
                os.path.dirname(save_path), backup_folder
            )
            os.makedirs(backup_path, exist_ok=True)
            shutil.move(
                save_path,
                os.path.join(backup_path, os.path.basename(save_path)),
            )
    plt.rc("font", family="Arial")
    data = df.query("mask=='cellpose' and channel=='mCherry'")
    fig, ax1 = plt.subplots(figsize=(5, 5), dpi=150, facecolor="w")
    sns.lineplot(
        ax=ax1,
        x=data.hours,
        y=data.top10px / data.mean_intensity,
        label="mCherry-Rad53",
        color="mediumvioletred",
    )
    ax2 = ax1.twinx()
    get_gfp_positive_number(df).plot(
        ax=ax2, label="GFP positive cells", color="seagreen", linewidth=3
    )
    get_cell_number(df).plot(
        ax=ax2, label="total number of cells", color="steelblue", linewidth=2
    )
    # sns.lineplot(ax=ax2,
    #              x="frame",
    #              y="GFP_positive",
    #              label="GFP positive cells",
    #              data=df
    #             )
    ax1.set_ylim(1, 2.5)
    ax1.set_ylabel("norm intensity")
    ax2.set_ylabel("cell count")
    ax2.legend(loc="upper right")
    ax1.legend(loc="upper left")
    ax1.set_title(f"top 10 px {title}")
    if save_path:
        fig.savefig(save_path)
        plt.close()
    plt.show()


def plot_max(df, title="", save_path="P=19-max_intensity.png"):
    plt.rc("font", family="Arial")
    data = df.query("mask=='cellpose' and channel=='mCherry'")
    fig, ax1 = plt.subplots(figsize=(5, 5), dpi=150, facecolor="w")
    sns.lineplot(
        ax=ax1,
        x=data.hours,
        y=data.max_intensity / data.mean_intensity,
        label="mCherry-Rad53",
        color="mediumvioletred",
    )
    ax2 = ax1.twinx()
    get_gfp_positive_number(df).plot(
        ax=ax2, label="GFP positive cells", color="seagreen", linewidth=3
    )
    get_cell_number(df).plot(
        ax=ax2, label="total number of cells", color="steelblue", linewidth=2
    )
    # sns.lineplot(ax=ax2,
    #              x="frame",
    #              y="GFP_positive",
    #              label="GFP positive cells",
    #              data=df
    #             )
    ax1.set_ylim(1, 2.5)
    ax1.set_ylabel("norm intensity")
    ax2.set_ylabel("cell count")
    ax2.legend(loc="upper right")
    ax1.legend(loc="upper left")
    ax1.set_title(f"max/mean intensity ratio {title}")

    fig.savefig(save_path)
    plt.close()


def plot_ilastik_intensity(
    pv_nuc, title="", save_path="P=19-nuc-cyto-intensity-ratio.png"
):
    fig, ax = plt.subplots(dpi=150, facecolor="w")
    ax = sns.lineplot(
        data=pv_nuc,
        x="hours",
        y="ratio",
    )
    ax.set_title(f"Ratio mean_intensity Ilastik nuc / cellpose cyto {title}")
    plt.savefig(save_path)
    plt.close()


def plot_num_nuc(filt_df, title="", save_path="P=19_nuc_ilastik.png"):
    fig, ax1 = plt.subplots(dpi=150, facecolor="w")
    (count_nuc_number(filt_df) / get_cell_number(filt_df)).plot(
        ax=ax1,
        label="Ilastik nuclei detections vs cell number",
        color="mediumvioletred",
    )
    ax2 = ax1.twinx()
    get_cell_number(filt_df).plot(
        ax=ax2, color="steelblue", label="total number of cells"
    )
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.set_ylabel("ratio")
    ax2.set_ylabel("count")
    ax1.set_title(f"Nuc Numbers {title}")
    fig.savefig(save_path)
    plt.close()


def plot_table(df, x="hours", title="", save_path="all_measurements.png"):
    # df = pd.read_csv(path)
    # df.loc[df["mask"]== 'cellpose']

    # df.loc[df.manual_filter == pd.nan]

    fdf = df

    fig, ax = plt.subplots()

    sns.lineplot(
        ax=ax,
        data=fdf.query("channel == 'mCherry' and mask == 'cellpose'"),
        x=x,
        y="mean_intensity",
        label="cellpose",
    )

    sns.lineplot(
        ax=ax,
        data=fdf.query("channel == 'mCherry' and mask == 'cyto - nuc'"),
        x=x,
        y="mean_intensity",
        label="cyto - nuc",
    )

    sns.lineplot(
        ax=ax,
        data=fdf.query("channel == 'mCherry' and mask == 'nuclei'"),
        x=x,
        y="mean_intensity",
        label="nuc",
    )

    sns.lineplot(
        ax=ax,
        data=fdf.query("channel == 'mCherry' and mask == 'cellpose'"),
        x=x,
        y="top10px",
        label="top10",
    )
    sns.lineplot(
        ax=ax,
        data=fdf.query("channel == 'mCherry' and mask == 'cellpose'"),
        x=x,
        y="cyto_wo100px",
        label="cyto_wo100px",
    )

    sns.lineplot(
        ax=ax,
        data=fdf.query(
            "channel == 'GFP' and mask == 'cellpose' and mean_intensity > 0"
        ),
        x=x,
        y="mean_intensity",
        label="GFP",
    )
    ax.set_title(title)
    if save_path:
        fig.savefig(save_path)
        plt.close()
    return ax


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


# get_good_cellpose_labels(pd.read_csv("pos/pos2/output/table.csv")).query("frame==13")
