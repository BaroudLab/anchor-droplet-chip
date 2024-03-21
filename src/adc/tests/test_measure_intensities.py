import numpy as np

from adc.yeast.cellpose.measure.intensities import (
    regionprops_table,
    top1percent,
)


def test_top1pc():
    data = np.random.randint(100, 2**16, size=(32, 32))
    mask = np.zeros_like(data, dtype=bool)
    mask[8:24, 8:24] = True
    out = top1percent(mask, data)
    assert out is not None
    print(out)


def test_top1pc_small():
    data = np.random.randint(100, 2**16, size=(32, 32))
    mask = np.zeros_like(data, dtype=bool)
    mask[12:20, 12:20] = True
    out = top1percent(mask, data)
    assert out is not None
    print("out:", out)
