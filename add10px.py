import os
import shutil
import numpy as np
import pandas as pd
import tifffile as tf
from skimage.measure import regionprops_table
from tqdm import tqdm
from fire import Fire


def top10px(regionmask, intensity):
    return np.sort(np.ravel(intensity[regionmask]))[-10:].mean()

def isin(name, column_names):
    return any(name in _name for _name in column_names)

def cells(
    path: str,
    stop_frame=49,
    suffix=("stack.tif", "cellpose.tif"),
    table_suffix=(".tif", ".csv"),
    params_suffix=(".tif", ".params.yml"),
    properties=("label", "centroid", "area", "mean_intensity", "max_intensity", "eccentricity"),
    extra_properties=(top10px, ),
):
    """
    measures the labels
    saves  csv with regionprops
    
    """
    props = (*properties, *(a.__name__ for a in extra_properties))

    labels_path = path.replace(*suffix)
    table_path = labels_path.replace(*table_suffix)
    if os.path.exists(table_path):
        columns = list(pd.read_csv(table_path).columns)
        if all(isin(p, columns) for p in props):
            print("all columns are there, skip")
            return table_path
        shutil.move(
            table_path, table_path+".bak")
    
    print("quantfy path:", path)

    if path.endswith(suffix[1]):
        print("skip cellpose output!")
        return
    labels_path = path.replace(*suffix)
    assert labels_path != path, f"Something wrong with the suffix `{suffix}` in `{path}`"
    
    mcherry = tf.imread(path)[:stop_frame, 1]
    print(mcherry.shape)
    max_label = 0
    labels_stack = tf.imread(labels_path)[:stop_frame]
       
    props = []
    for frame, (l, d) in enumerate(zip(labels_stack, mcherry)):
        try:
            prop = {
                **regionprops_table(
                    label_image=l, intensity_image=d, properties=properties, extra_properties=extra_properties
                ),
            }
            
            prop["frame"] = frame
            props.append(pd.DataFrame(prop))
        except ValueError as e:
            print(
                f"table failed on frame {frame} label {l.shape}, mcherry {d.shape}: {e}"
            )

            
            return
    df = pd.concat(props, ignore_index=True)
    
        
    df.to_csv(table_path)
    
    return table_path


def main(*paths):
    """Segements movies tifs"""
    return [cells(p) for p in tqdm(paths)]

if __name__ == "__main__":
    Fire(main)