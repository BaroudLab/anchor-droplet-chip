from typing import List, Tuple

import numpy as np
import pandas as pd

from adc.yeast.plot.single import (
    Layer,
    get_positive_gfp,
    get_title,
    log,
    os,
    plot_10,
    plot_max,
    plot_table,
    read_data,
    regionprops_table,
    shutil,
    sns,
    tf,
    time,
)

from ...cellpose.measure.intensities import (
    cyto_wo100px,
    top1percent,
    top10px,
    top20px,
    top100px,
)
from .plot import plot_tracked_labels as _plot_tracked_labels


def analyse_manual_tracks(prefix):
    # analyze data using analyse_all_layers_manual function wtih specific parameters
    manual_table, save_dir = _analyse_all_layers_manual(
        prefix=prefix,
        cp_path="manual_tracks.tif",
    )
    # plot tracked labels using plot_tracked_labels function
    fig = _plot_tracked_labels(manual_table, title=get_title(prefix))
    # save generated plot as png image
    fig.savefig(os.path.join(save_dir, "manual_tracks.png"))
    return fig


def _analyse_all_layers_manual(
    prefix: str = "",
    data_path: str = "stack.tif",  # path to stack.tif
    cp_path: str = "manual_tracks.tif",  # path to manual track tifs
    frames_per_hour=2,  # number of frames per hour
    backup_folder="backup",  # folder name for backup
    title_re=r"pos\d+",  # regular expression pattern for titles of folders
    input_output_folders=(
        "input",
        "output_manual",
    ),  # input and output folders
    reader=read_data,  # data reader function
):
    title = get_title(prefix)  # extract title from prefix
    log.info(f"title will be `{title}`")

    prefix = str(prefix)
    save_dir = os.path.dirname(prefix).replace(
        *input_output_folders
    )  # generate save directory
    assert save_dir != prefix  # assert save directory is not same as prefix
    log.info(f"output directory will be `{save_dir}`")
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
        log.warning(
            f"output directory not empty `{save_dir}`"
        )  # so output doesn't overwrite if data is already there
        if backup_folder:
            backup_path = "_".join(
                [save_dir, backup_folder, time.strftime("%Y%m%d-%H%M%S")]
            )
            os.makedirs(backup_path, exist_ok=True)  # create backup directory
            shutil.move(
                save_dir, backup_path
            )  # move current save directory to backup
            log.warning(f"Backed up to {backup_path}")  # log backup
        else:
            log.error(
                f"Data exists! Please remove or indicate backup_folder to backup the existing data"
            )
            return

    bf_layer, mcherry_layer, gfp_layer = reader(
        path := os.path.join(prefix, data_path)
    )
    log.info("read stack")  # log that stack data is read
    cyto_layer = Layer(
        tf.imread(os.path.join(prefix, cp_path)),
        name="cellpose",
        kind="labels",
    )
    # log.info("read cellpose")

    log.info("processing table")  # log processing of table
    df = _get_table(
        fluo_layers=[mcherry_layer, gfp_layer],
        label_layers=[
            cyto_layer,
        ],
        path=path,
    )
    log.info(
        "table with regonprops recovered across 2 channels and 2 labels"
    )  # log successful table creation

    df.loc[:, "hours"] = (
        df.frame / frames_per_hour
    )  # add hours column to dataframe
    log.info("hours added")  # log that hours are added

    log.info("label gfp positive cells")  # log labeling of GFP positive cells
    df = get_positive_gfp(df)  # apply function to label GFP positive cells

    save_path = path.replace(".tif", ".csv")  # generate save path
    assert save_path != path  # assert save path is not same as input path
    # df.to_csv(save_path)
    # df.to_excel(path.replace(".tif",".xlsx"))
    log.info(f"saving everything into {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(ppp := os.path.join(save_dir, "table.csv"))
    log.info(f"table saved `{ppp}`")

    try:
        log.info(f"plot top 10 px")  # log plotting top 10 pixels
        plot_10(
            df,
            title=title,
            save_path=(
                ppp := os.path.join(save_dir, "top10px_intensity_ratio.png")
            ),
        )
        log.info(f"plot saved `{ppp}`")  # log successful plotting

        log.info(f"plot max_intensity_ratio")  # log max intensity ratio
        plot_max(
            df,
            title=title,
            save_path=(
                ppp := os.path.join(save_dir, "max_intensity_ratio.png")
            ),
        )
        log.info(f"plot saved `{ppp}`")

        # log.info(f"plot count_nuc_ilastik")
        # plot_num_nuc(
        #     df,
        #     title=title,
        #     save_path=(ppp := os.path.join(save_dir, "count_nuc_ilastik.png")),
        # )
        # log.info(f"plot saved `{ppp}`")

        log.info(f"plot all_measurements")
        plot_table(
            df,
            title=title,
            save_path=(ppp := os.path.join(save_dir, "all_measurements.png")),
        )
        log.info(f"plot saved `{ppp}`")
    except Exception as e:
        print(e)
    finally:
        return df, save_dir


def _get_table(
    fluo_layers: List,  # list fluorescent layers
    label_layers: List,  # list label layers
    properties: List[str] = [  # list of properties to extract
        "label",  # cell masks
        "centroid",  # centroid coordinates of cell
        "area",  # area of cell
        "mean_intensity",
        "max_intensity",
    ],
    extra_properties: Tuple = (
        top10px,
        top1percent,
    ),  # tuple = unmodifiable list; here additional characteristic
    path: str = "",  # path information
):
    intensities = [
        {
            **(  # use dictionary unpacking to merge dictionaries
                props := regionprops_table(  # calculate region properties
                    label_image=m,  # label image of cell
                    intensity_image=f,  # intensity of cell
                    properties=properties,  # properties to extract
                    extra_properties=extra_properties,  # additional properties
                )
            ),
            "path": path,  # path info again
            "channel": fluo.name,  # name of fluorescent channel
            "mask": mask.name,  # name of mask
            "frame": i,  # frame index
        }
        for mask in label_layers  # iterate through label layers
        for fluo in fluo_layers  # iterate through fluorescence layers
        for i, (
            m,
            f,
        ) in enumerate(  # iterate through mask and fluorescent data pairs
            zip(
                (
                    mask.data
                    if isinstance(mask.data, np.ndarray)
                    else mask.data.compute()
                ),  # make data in ndarray format
                (
                    fluo.data
                    if isinstance(fluo.data, np.ndarray)
                    else fluo.data.compute()
                ),  # make data in ndarray format
            )
        )
        if m.mean() > 0  # check if mean value of mask is greater than 0
    ]
    df = pd.concat(
        [pd.DataFrame(i) for i in intensities], ignore_index=True
    )  # concatenate dictionaries into dataframe
    df = df.rename(
        columns={"centroid-0": "y", "centroid-1": "x"}  # rename coordinates
    )
    return df  # return to resulting dataframe