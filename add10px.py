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
            if frame == 0:
                for k in prop:
                    values = list(prop[k])
                    values.insert(0, 0)
                    prop[k] = values
            prop["frame"] = frame
            props.append(pd.DataFrame(prop))
        except ValueError as e:
            print(
                f"table failed on frame {frame} label {l.shape}, mcherry {d.shape}: {e}"
            )

            
            return
    df = pd.concat(props, ignore_index=True)
    if os.path.exists(table_path := labels_path.replace(*table_suffix)):
        shutil.move(
            table_path, table_path+".bak")
        
    df.to_csv(table_path)
    
    return table_path


def main(*paths):
    """Segements movies tifs"""
    return [cells(p) for p in tqdm(paths)]

if __name__ == "__main__":
    Fire(main)