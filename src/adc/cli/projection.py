import os

import dask.array as da
import nd2
import numpy as np
from zarr_tools import convert


def make_projection(nd2_path: str, axis_name: str = "Z", op="mean"):
    try:
        print(nd2_path)
        d = nd2.ND2File(nd2_path)
        print(d.sizes)
        axes = list(d.sizes)
        axis = axes.index(axis_name)
        kwargs = {"dtype": np.float32} if op == "mean" else {}
        proj = d.to_dask().__getattribute__(op)(axis=axis, **kwargs)
        new_axes = axes.pop(axis)
        new_sizes = {a: d.sizes[a] for a in axes}
        print(new_sizes)
        save_path = nd2_path.replace(".nd2", f"-{op}{axis_name}.zarr")
        assert not os.path.exists(save_path)
        convert.to_zarr(
            proj,
            path=save_path,
            steps=4,
            channel_axis=None,
            name=f"TRITC_{op}{axis_name}",
            sizes=new_sizes,
            colormap="green",
            lut=[400, 600],
        )
        return save_path

    except OSError:
        d = None
        print(f"Unamble to read {nd2_path}")
    except AssertionError:
        print(f"File Exists! {save_path}")
