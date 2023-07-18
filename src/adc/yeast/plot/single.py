from napari import Viewer
from napari.layers import Labels, Image
from typing import List, Union, Tuple
from skimage.measure import regionprops, regionprops_table
import dask.array as da
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn  as sns
import tifffile as tf
# from cellpose import models

import os
from glob import glob
import yaml
import matplotlib.pyplot as plt
# import plot_tools
import pathlib
import yaml
from .tools import Layer, filters, read_data


GFP_POSITIVE_THRESHOLD = 140



def analyse_all_layers(prefix:str = '', filter_tif="filter.tif", data_path:str = "stack.tif", cp_path:str="cellpose.tif", il_path:str="ilastik.tif", frames_per_hour=2, reader=read_data):
    
    save_dir = os.path.dirname(prefix).replace("input", "output")
    os.makedirs(save_dir, exist_ok=True)
    # if os.path.exists(os.path.join(save_dir, "table.csv")):
    #     print("exists!")
    #     return
    bf_layer, mcherry_layer, gfp_layer = reader(path := os.path.join(prefix, data_path))
    [cellpose_layer] = reader(os.path.join(prefix, cp_path))
    [ilastik_layer] = reader(os.path.join(prefix, il_path))
    filter_layer = Layer(tf.imread(os.path.join(prefix, filter_tif)) - 1,
                          name="filter",
                          kind = "labels",
                          )
    
    cellpose_layer_filtered = Layer(
        cellpose_layer.data * (filter_layer.data),
        name="cellpose",
        kind="labels",
        metadata=dict(
            source={
                "filter": filter_layer.source.path, 
                "cellpose": cellpose_layer.source.path
            },
            op="filter * cellpose"
        )
    )
        
    ilastik_times_cellpose_layer = Layer(
        ilastik_layer.data * cellpose_layer_filtered.data, 
        name="nuclei", 
        kind="labels",
        metadata=dict(
            source={
                "ilastik": ilastik_layer.source.path, 
                "cellpose": cellpose_layer_filtered.source.path
            },
            op="ilastik * cellpose"
        )
    )
    ilastik_properties=(
            "label",
            "centroid",
            "area",
            # "mean_intensity",
            "eccentricity"
        )
    props = [
        regionprops_table(
            label_image=l, 
            # intensity_image=i, 
            properties=ilastik_properties
        )
        for l in ilastik_times_cellpose_layer.data
    ]        
    ilastik_props = pd.concat(pd.DataFrame(p, index=p["label"]) for p in props)
    ilastik_props.loc[0] = [0] * len(ilastik_props.columns)
    ilastik_props = ilastik_props.reset_index()

    ilastik_layer.properties = ilastik_props

    cellpose_minus_ilastik_layer = Layer(
        cellpose_layer_filtered.data - ilastik_times_cellpose_layer.data, 
        name="cyto - nuc",
        metadata=dict(
           source={
                "ilastik": ilastik_times_cellpose_layer.metadata, 
                "cellpose": cellpose_layer_filtered.source.path
            },
            op="cellpose - ilastik"
        ),
        kind="labels"
    )
    cellpose_minus_ilastik_layer.properties = cellpose_layer.properties

    df = get_table(
        fluo_layers=[mcherry_layer, gfp_layer],
        label_layers=[cellpose_layer_filtered, cellpose_minus_ilastik_layer, ilastik_times_cellpose_layer],
        path = path
    )
    df.loc[:,"hours"] = df.frame / frames_per_hour
    
    df = get_good_cellpose_labels(df, filters=filters)
    df.loc[df["mask"] == 'cyto - nuc', "manual_filter"] = df.loc[df["mask"] == 'cellpose', "manual_filter"].values
    df = filter_nuc_size(df, filters=filters)
    df = filter_gfp_intensity(df, filters=filters)
    df.loc[:,"hours"] = df.frame / 2
    df = get_positive_gfp(df)
    df.loc[(df["mean_intensity"] == 0),"manual_filter"] = 0

    filt_df = df.query("manual_filter == 1")
    
    pv_nuc = filt_df.pivot_table(index=["label","hours","channel"], columns="mask", values=["max_intensity","mean_intensity"]).dropna().reset_index()
    pv_nuc.loc[:, "ratio"] = pv_nuc["mean_intensity"]["nuclei"] / pv_nuc["mean_intensity"]["cellpose"]
  
    save_path = path.replace(".tif",".csv")
    assert save_path != path
    # df.to_csv(save_path)
    # df.to_excel(path.replace(".tif",".xlsx"))
    print(save_dir)
    df.to_csv(os.path.join(save_dir, "table.csv"))
    filt_df.to_csv(os.path.join(save_dir, "filt_table.csv"))
    pv_nuc.to_csv(os.path.join(save_dir, "pivot_table_nuc.csv"))
    try:
        plot_10(filt_df, save_path=os.path.join(save_dir, 'top10px_intensity_ratio.png'))
        plot_ilastik_intensity(pv_nuc, save_path=os.path.join(save_dir, 'nuc-cyto-intensity-ratio.png'))
        plot_max(filt_df, save_path=os.path.join(save_dir, 'max_intensity_ratio.png'))
        plot_num_nuc(filt_df, save_path=os.path.join(save_dir, 'count_nuc_ilastik.png'))
        plot_table(filt_df, prefix=prefix, save_path = os.path.join(save_dir, "all_measurements.png"))
    except Exception as e:
        print(e)
    finally:
        return filt_df


def top10px(regionmask, intensity):
    return np.sort(np.ravel(intensity[regionmask]))[-10:].mean()
def top20px(regionmask, intensity):
    return np.sort(np.ravel(intensity[regionmask]))[-20:].mean()
    

def filter_nuc_size(df, filters=filters):
    table = df.copy()
    table.loc[df["mask"] == "nuclei","manual_filter"] = table[table["mask"] == "nuclei"]["area"] > filters["filters"]["nuc"]["area"]["min"]
    return table
    
def filter_gfp_intensity(df, filters=filters):
    table = df.copy()
    table.loc[(df["mask"] == "cellpose") & (df.channel == "GFP"), "manual_filter"] = (
        table[(df["mask"] == "cellpose") & (df.channel == "GFP")]["mean_intensity"] > 
        filters["filters"]["cyto"]["mean_intensity"]["GFP"]["min"]
    )
    return table

def get_table(
    fluo_layers:List[Image],
    label_layers:List[Labels],
    properties:List[str]=["label", "centroid","area","mean_intensity","max_intensity"],
    path:str = ""
):
    intensities = [{
        **(props := regionprops_table(
            label_image=mask.data, 
            intensity_image=fluo.data.compute() if isinstance(fluo.data, da.Array) else fluo.data, 
            properties=properties,
            extra_properties=(top10px,top20px)
        )), 
        "path": path,
        "channel": fluo.name, 
        "mask": mask.name,
        }
        for mask in label_layers
        for fluo in fluo_layers
        if fluo.data.mean() > 0
    ]
    df = pd.concat([pd.DataFrame(i) for i in intensities], ignore_index=True)
    df = df.rename(columns={"centroid-0": "frame","centroid-1": "y", "centroid-2": "x"})
    return df

def get_positive_gfp(df, thr=GFP_POSITIVE_THRESHOLD):
    table = df.copy()
    table.loc[:,"GFP_positive"] = table.mean_intensity > thr
    return table

def get_gfp_positive_number(df):
    return df.query("channel == 'GFP' and mask == 'cellpose'").groupby("hours").sum().GFP_positive

def count_nuc_number(df):
    return df.query("mask == 'nuclei' and channel == 'mCherry'").groupby("hours").count()["channel"]



def count_nuc_number(df):
    return df.query("mask == 'nuclei' and channel=='mCherry'").groupby("hours").count()["frame"]
def get_cell_number(df):
    return df.query("mask == 'cellpose' and channel=='mCherry'").groupby("hours").count()["frame"]




def plot_10(df, save_path="P=19-top10px.png"):
    plt.rc('font',family='Arial')
    data=df.query("mask=='cellpose' and channel=='mCherry'")
    fig, ax1 = plt.subplots(figsize = (5,5), dpi=150, facecolor='w')
    sns.lineplot(ax=ax1,
                 x=data.hours,
                 y=data.top10px / data.mean_intensity,
                 label = "mCherry-Rad53",
                 color = 'mediumvioletred'
                )
    ax2=ax1.twinx()
    get_gfp_positive_number(df).plot(ax=ax2, label="GFP positive cells", color="seagreen", linewidth=3)
    get_cell_number(df).plot(ax=ax2, label="total number of cells", color="steelblue", linewidth=2)
    # sns.lineplot(ax=ax2,
    #              x="frame",
    #              y="GFP_positive",
    #              label="GFP positive cells",
    #              data=df
    #             )
    ax1.set_ylim(1,2.5)
    ax1.set_ylabel("norm intensity")
    ax2.set_ylabel("cell count")
    ax2.legend(loc="upper right")
    ax1.legend(loc="upper left")
    ax1.set_title("top 10 px")
    fig.savefig(save_path)
    plt.close()

def plot_max(df, save_path="P=19-max_intensity.png"):
    plt.rc('font',family='Arial')
    data=df.query("mask=='cellpose' and channel=='mCherry'")
    fig, ax1 = plt.subplots(figsize = (5,5), dpi=150, facecolor='w')
    sns.lineplot(ax=ax1,
                 x=data.hours,
                 y=data.max_intensity / data.mean_intensity,
                 label = "mCherry-Rad53",
                 color = 'mediumvioletred'
                )
    ax2=ax1.twinx()
    get_gfp_positive_number(df).plot(ax=ax2, label="GFP positive cells", color="seagreen", linewidth=3)
    get_cell_number(df).plot(ax=ax2, label="total number of cells", color="steelblue", linewidth=2)
    # sns.lineplot(ax=ax2,
    #              x="frame",
    #              y="GFP_positive",
    #              label="GFP positive cells",
    #              data=df
    #             )
    ax1.set_ylim(1,2.5)
    ax1.set_ylabel("norm intensity")
    ax2.set_ylabel("cell count")
    ax2.legend(loc="upper right")
    ax1.legend(loc="upper left")
    ax1.set_title("max/mean intensity ratio")
    
    fig.savefig(save_path)
    plt.close()

def plot_ilastik_intensity(pv_nuc, save_path="P=19-nuc-cyto-intensity-ratio.png"):
    fig, ax = plt.subplots(dpi=150, facecolor='w')
    ax = sns.lineplot(data=pv_nuc, x="hours", y="ratio", )
    ax.set_title("Ratio mean_intensity Ilastik nuc / cellpose cyto")
    plt.savefig(save_path)
    plt.close()

def plot_num_nuc(filt_df, save_path="P=19_nuc_ilastik.png"):
    fig, ax1 = plt.subplots(dpi=150, facecolor='w')
    (count_nuc_number(filt_df) / get_cell_number(filt_df)).plot(ax=ax1, label="Ilastik nuclei detections vs cell number", color="mediumvioletred")
    ax2 = ax1.twinx()
    get_cell_number(filt_df).plot(ax=ax2, color="steelblue", label="total number of cells")
    ax1.legend(loc = 'upper left')
    ax2.legend(loc = 'upper right')
    ax1.set_ylabel("ratio")
    ax2.set_ylabel("count")
    ax1.set_title("Nuc Numbers")
    fig.savefig(save_path)
    plt.close()
    
def plot_table(df, prefix, save_path="all_measurements.png"):
    # df = pd.read_csv(path)
    # df.loc[df["mask"]== 'cellpose']

    # df.loc[df.manual_filter == pd.nan]

    fdf = df
    
    fig, ax = plt.subplots()

    sns.lineplot(
        ax=ax,
        data=fdf.query("channel == 'mCherry' and mask == 'cellpose'"), 
        x='hours', 
        y='mean_intensity',
        label="cellpose"
    )

    sns.lineplot(
        ax=ax,
        data=fdf.query("channel == 'mCherry' and mask == 'cyto - nuc'"), 
        x='hours', 
        y='mean_intensity',
        label="cyto - nuc"
    )

    sns.lineplot(
        ax=ax,
        data=fdf.query("channel == 'mCherry' and mask == 'nuclei'"), 
        x='hours', 
        y='mean_intensity',
        label="nuc"
    )

    sns.lineplot(
        ax=ax,
        data=fdf.query("channel == 'mCherry' and mask == 'nuclei'"), 
        x='hours', 
        y='top10px',
        label="top10"
    )


    sns.lineplot(
        ax=ax,
        data=fdf.query("channel == 'GFP' and mask == 'nuclei'"), 
        x='hours', 
        y='mean_intensity',
        label="GFP"
    )
    ax.set_title(prefix.split("/")[-3])
    
    fig.savefig(save_path)
    plt.close()



def get_good_cellpose_labels(df, filters=filters):
    table = df.copy()
    table.loc[table["mask"] == "cellpose","manual_filter"] = np.logical_and(
        table[table["mask"] == "cellpose"]["area"] > filters["filters"]["cyto"]["area"]["min"], 
        table[table["mask"] == "cellpose"]["area"] < filters["filters"]["cyto"]["area"]["max"]
    )
    # table.loc[table["mask"] == "cellpose","manual_filter"] = np.logical_and(table.loc[table["mask"] == "cellpose","manual_filter"],  table[table["mask"] == "cellpose"]["mean_intensity"] < filters["filters"]["cyto"]["mean_intensity"]["mCherry"]["max"])
    # table.loc[table["mask"] == "cellpose","manual_filter"] = np.logical_and(table.loc[table["mask"] == "cellpose","manual_filter"],  table[table["mask"] == "cellpose"]["mean_intensity"] > filters["filters"]["cyto"]["mean_intensity"]["mCherry"]["min"])
    return table


# get_good_cellpose_labels(pd.read_csv("pos/pos2/output/table.csv")).query("frame==13")