""" CLI to merge YeastTube tables """

import os
from pathlib import Path

import fire
import pandas as pd

GFP_POSITIVE_THRESHOLD = 140


def read_csv(path, query="frame <= 48"):
    try:
        df = pd.read_csv(path).query(query)
        # print(df.head())
        # print(df.channel.unique())
        df.loc[:, "path"] = os.path.sep.join(path.split(os.path.sep)[-6:])
        df.loc[:, "mask"] = "cellpose"
        df.loc[:, "hours"] = df.frame / 2
        df.loc[:, "GFP_positive"] = df.mean_intensity > GFP_POSITIVE_THRESHOLD
        gfp_hour = df.query("GFP_positive and channel=='GFP'").hours.min()
        df.loc[:, "GFPhour"] = df.hours - gfp_hour
    
        cellpose_path = Path(path).parent.parent / "input" / "cellpose.csv"

        merge_cols = ["label", "area", "centroid-0", "centroid-1"]
        if not "top10px" in list(df.columns):
            merge_cols.append("top10px")
            
        df1 = pd.read_csv(
            cellpose_path,
            index_col=0,
        ).query(query)
        assert "top10px" in df1.columns
        # print(df1.head)
        df11 = df1[
            merge_cols
        ].rename(columns={"centroid-0": "y", "centroid-1": "x"})
        df2 = df.merge(right=df11, on="label")
        df2.loc[:, "ratio"] = df2.top10px / df2.mean_intensity
    except (AttributeError, ValueError) as e:
        print(f"problem with {path}", e)
        return 
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
