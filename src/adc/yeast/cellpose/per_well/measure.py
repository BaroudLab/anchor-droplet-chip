import os
import pandas as pd
from skimage.measure import regionprops_table
import tifffile as tf
import numpy as np
import dask.array as da


properties=(
            "label",
            "centroid",
            "area",
            "mean_intensity",
            "eccentricity"
        )


def labels(labels_path:str, stack_path:str, suffix=("cellpose.tif", "cellpose.csv")):
    if not labels_path.endswith(suffix[0]):
        print(f"skip {labels_path}")
    save_path = labels_path.replace(*suffix)
    print(save_path)
    if os.path.exists(save_path):
        print(f"exists {save_path}")
        return pd.read_csv(save_path)
    
    labels = tf.imread(labels_path)
   
    mcherry = da.from_zarr(tf.TiffFile(stack_path).aszarr())[:,1].compute()
    props = []
    for frame, (l, d) in enumerate(zip(labels, mcherry)):
        prop = {**regionprops_table(label_image=l, intensity_image=d,
                              properties=properties),
                }
        if frame == 0:
            for k in prop:
                values= list(prop[k])
                values.insert(0,0)
                prop[k] = values
        prop["frame"] =  frame
        props.append(pd.DataFrame(prop))
    df = pd.concat(props, ignore_index=True)
    df.to_csv(save_path)
    return df
              