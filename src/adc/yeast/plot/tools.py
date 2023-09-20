import logging
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Union

import numpy as np
import yaml

from adc._reader import napari_get_reader, read_tif_yeast

logger = logging.getLogger(__name__)


filters = {'filters': {'cyto': {'area': {'min': 100, 'max': 3500},
   'mean_intensity': {'mCherry': {'min': 150, 'max': 400},
    'GFP': {'threshold': 140, 'min': 10}}},
  'nuc': {'area': {'min': 20, 'max': 300}}}}

def slice_from_axis(array, *, axis, element):
    """Take a single index slice from array using slicing.

    Equivalent to :func:`np.take`, but using slicing, which ensures that the
    output is a view of the original array.

    Parameters
    ----------
    array : NumPy or other array
        Input array to be sliced.
    axis : int
        The axis along which to slice.
    element : int
        The element along that axis to grab.

    Returns
    -------
    sliced : NumPy or other array
        The sliced output array, which has one less dimension than the input.
    """
    slices = [slice(None) for i in range(array.ndim)]
    slices[axis] = element
    return array[tuple(slices)]


def read_data(path, reader=napari_get_reader):
    out = reader(path)
    # print(out)
    if isinstance(out, List):
        return out
    data, props, kind = read_data(path, out)[0]
    if "channel_axis" in props and (ca := props["channel_axis"]) is not None:
        return [
            Layer(
                data=d,
                kind=kind,
                metadata=props["metadata"],
                source=Source(path=path),
                colormap=c,
                name=n,
                contrast_limits=cl,
            )
            for d, c, n, cl in zip(
                (
                    slice_from_axis(data, axis=ca, element=i)
                    for i in range(data.shape[ca])
                ),
                props["colormap"],
                props["name"],
                props["contrast_limits"],
            )
        ]
    return [Layer(data=data, kind=kind, **props)]


@dataclass
class Source:
    path: str = ""


@dataclass
class Layer:
    data: np.typing.ArrayLike
    name: Union[str, List[str]]
    metadata: dict = field(default_factory=dict)
    contrast_limits: Tuple = field(default_factory=tuple)
    colormap: List = field(default_factory=list)
    kind: str = "image"
    properties: dict = field(default_factory=dict)
    channel_axis: int = None
    source: Source = Source(path="")
