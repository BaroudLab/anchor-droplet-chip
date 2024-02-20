""" CLI to merge YeastTube tables """

import os

import fire
import pandas as pd

GFP_POSITIVE_THRESHOLD = 140


def read_csv(path):
    df = pd.read_csv(path)
    # print(df.head())
    # print(df.channel.unique())
    df.loc[:, "path"] = os.path.sep.join(path.split(os.path.sep)[-6:])
    df.loc[:, "mask"] = "cellpose"
    df.loc[:, "hours"] = df.frame / 2
    df.loc[:, "GFP_positive"] = df.mean_intensity > GFP_POSITIVE_THRESHOLD
    df.loc[:, "ratio"] = df.max_intensity / df.mean_intensity
    gfp_hour = df.query("GFP_positive and channel=='GFP'").hours.min()
    df.loc[:, "GFPhour"] = df.hours - gfp_hour
    df1 = pd.read_csv(
        os.path.join(*path.split(os.path.sep)[:-2], "input/cellpose.csv"),
        index_col=0,
    )
    df11 = df1[["label", "area", "centroid-0", "centroid-1"]].rename(
        columns={"centroid-0": "y", "centroid-1": "x"}
    )
    df2 = df.merge(right=df11, on="label")
    return df2


def process(*table_paths):
    """
    Merging tables from YeastTube platform
    table_paths: list of strings
        like: /2024-01-19_MLY003_Cas9_sync/pos/Included/pos90/output/table_0.csv'
    Return path of saved csv
    """
    merged_table = pd.concat(
        map(read_csv, filter(os.path.exists, table_paths)), ignore_index=True
    )
    commonpath = os.path.commonpath(table_paths)
    merged_table.to_csv(
        ppp := os.path.join(commonpath, "merged_table.csv"), index=None
    )
    return ppp


if __name__ == "__main__":
    fire.Fire(process)
