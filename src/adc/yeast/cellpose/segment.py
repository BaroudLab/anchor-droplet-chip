import os
import shutil
import time

import numpy as np
import pandas as pd
import tifffile as tf
import torch
import yaml
from cellpose import models
from fire import Fire
from skimage.measure import regionprops_table
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device
pretrained_model = os.path.expanduser(
    "~/Multicell/Madison/Cellpose_Model/CP_20220916_140544"
)


def top10px(regionmask, intensity):
    return np.sort(np.ravel(intensity[regionmask]))[-10:].mean()


def cells(
    path: str,
    stop_frame=None,
    model_params=dict(
        gpu=True,
        diam_mean=37,
        device=device,
        pretrained_model=pretrained_model,
    ),
    suffix=("stack.tif", "cellpose.tif"),
    table_suffix=(".tif", ".csv"),
    params_suffix=(".tif", ".params.yml"),
    properties=("label", "centroid", "area", "mean_intensity", "eccentricity"),
    extra_properties=(top10px,),
    backup_folder="backup",
    eval_kwargs={},
    model_kwargs={},
):
    """
    segments and measures the labels
    saves tif file with labels and csv with regionprops
    model:
        custom_model = models.CellposeModel(
            gpu=True,
            diam_mean=37,
            device=torch.device("mps"),
            pretrained_model="/Volumes/Multicell/Madison/Cellpose_Model/CP_20220916_140544"
        )
    """
    print("segmenting path:", path)

    if path.endswith(suffix[1]):
        print("skip cellpose output!")
        return
    save_path = path.replace(*suffix)
    assert (
        save_path != path
    ), f"Something wrong with the suffix `{suffix}` in `{path}`"
    if os.path.exists(save_path):
        if backup_folder:
            backup_path = os.path.join(
                os.path.dirname(save_path),
                "_".join([backup_folder, time.strftime("%Y%m%d-%H%M%S")]),
            )
            os.makedirs(backup_path, exist_ok=True)
            shutil.move(
                save_path,
                os.path.join(backup_path, os.path.basename(save_path)),
            )
            print(f"Backup existing data to {backup_folder}")
        else:
            print(f"segmentaion exists, return {save_path}!")
            return save_path
    custom_model = models.CellposeModel(
        gpu=model_params["gpu"],
        diam_mean=model_params["diam_mean"],
        device=torch.device(model_params["device"]),
        pretrained_model=model_params["pretrained_model"],
        **model_kwargs,
    )

    yaml_params = {}
    yaml_params["cellpose"] = model_params.copy()

    with tf.TiffWriter(save_path, imagej=True) as tif:
        data = tf.imread(path)[:stop_frame]
        print(data.shape)
        mcherry = data[:, 1]
        max_label = 0
        labels_stack = []
        for d in tqdm(mcherry):
            labels = custom_model.eval(d, **eval_kwargs)[0]
            labels = labels + max_label
            labels[labels == max_label] = 0
            if labels.max() > max_label:
                max_label = labels.max()
            labels_stack.append(labels)
        label = np.stack(labels_stack)
        t, y, x = label.shape
        tif.write(label.reshape((t, 1, 1, y, x)))

    yaml_params["source"] = path
    yaml_params["labels"] = save_path
    yaml_params["stop_frame"] = stop_frame

    props = []
    for frame, (l, d) in enumerate(zip(label, mcherry)):
        try:
            prop = {
                **regionprops_table(
                    label_image=l,
                    intensity_image=d,
                    properties=properties,
                    extra_properties=extra_properties,
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

            with open(
                (yaml_path := save_path.replace(*params_suffix)), "w"
            ) as f:
                yaml.safe_dump(yaml_params, f)
            print(f"saving yaml params")
            return
    df = pd.concat(props, ignore_index=True)
    if os.path.exists(table_path := save_path.replace(*table_suffix)):
        shutil.move(
            table_path, os.path.join(backup_path, os.path.basename(table_path))
        )
    df.to_csv(table_path)
    yaml_params["regonprops"] = properties
    yaml_params["extra_properties"] = [a.__name__ for a in extra_properties]
    yaml_params["table_path"] = table_path

    if os.path.exists(yaml_path := save_path.replace(*params_suffix)):
        shutil.move(
            yaml_path, os.path.join(backup_path, os.path.basename(yaml_path))
        )
    with open(yaml_path, "w") as f:
        yaml.safe_dump(yaml_params, f)
    return save_path


def main(*paths):
    """Segements movies tifs"""
    return [cells(p) for p in tqdm(paths)]


if __name__ == "__main__":
    Fire(main)
