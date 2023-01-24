import logging
import os
from functools import partial

import dask.array as da
import napari
import numpy as np
from magicgui import magic_factory
from magicgui.widgets import (
    Container,
    FileEdit,
    Label,
    PushButton,
    RadioButtons,
    SliceEdit,
    Table,
    create_widget,
)
from napari.layers import Image, Points, Shapes
from napari.utils import progress
from napari.utils.notifications import show_warning
from qtpy.QtWidgets import QVBoxLayout, QWidget

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
from napari.layers.utils.stack_utils import slice_from_axis
from tifffile import imwrite


class SplitAlong(QWidget):
    def __init__(self, napari_viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self.data_widget = create_widget(
            annotation=Image,
            label="data",
        )
        self.path_widget = FileEdit(mode="d")
        self.path_widget.changed.connect(self.update_table)
        self.saving_table = Table(value=[{}])
        self.data_widget.changed.connect(self.init_data)
        self.axis_selector = RadioButtons(
            label="Choose axis", orientation="horizontal", choices=()
        )
        self.split_btn = PushButton(text="Split it!")
        self.split_btn.clicked.connect(self.make_new_layer)
        self.save_btn = PushButton(text="Save tifs!")
        self.save_btn.clicked.connect(self.save_tifs)

        self.input_container = Container(
            widgets=[
                self.data_widget,
                self.axis_selector,
                self.split_btn,
                self.path_widget,
                self.saving_table,
                self.save_btn,
            ]
        )
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.input_container.native)
        self.layout.addWidget(self.path_widget.native)
        self.layout.addStretch()
        self.setLayout(self.layout)

        self.init_data()

    def save_tifs(self):
        for i, (name, shape, path, _) in progress(
            enumerate(ttt := self.saving_table.data.to_list()), total=len(ttt)
        ):
            logger.debug(f"Saving {name} into {path}")
            try:
                data = self.data_list[i].compute()
                meta = self.meta.copy()
                meta["spacing"] = meta["pixel_size_um"]
                meta["unit"] = "um"
                data_formatted_imagej = (
                    np.expand_dims(data, axis=1)
                    if "Z" not in meta["sizes"]
                    else data
                )
                imwrite(
                    path, data_formatted_imagej, imagej=True, metadata=meta
                )
                self.saving_table.data[i] = [name, data.shape, path, "Saved!"]
            except Exception as e:
                logger.error(f"Failed saving {name} into {path}: {e}")

    def make_new_layer(self):
        channel_axis = self.axis_selector.choices.index(
            axis_sel := self.axis_selector.current_choice
        )
        self.meta = self.dataset.metadata
        self.data_list = [
            slice_from_axis(array=self.dask_data, axis=channel_axis, element=i)
            for i in range(self.sizes[axis_sel.split(":")[0]])
        ]
        letter, total = self.axis_selector.current_choice.split(":")
        self.names = [
            f"{self.data_widget.current_choice}_{letter}={i}"
            for i, _ in enumerate(self.data_list)
        ]
        self.update_table()

    def update_table(self):
        self.saving_table.value = [
            {
                "name": name,
                "shape": array.shape,
                "path": os.path.join(
                    self.path_widget.value,
                    name + ".tif",
                ),
                "saved": "...",
            }
            for array, name in zip(self.data_list, self.names)
        ]

    def init_data(self):
        try:
            self.dataset = self.viewer.layers[self.data_widget.current_choice]
        except KeyError:
            logger.debug("no dataset")
            self.sizes = None
            self.path = None
            logger.debug("set sizes and path to None")

            return

        try:
            self.dask_data = self.dataset.metadata["dask_array"]
            logger.debug(f"Found dask_data in layer metadata {self.dask_data}")
        except KeyError:
            self.dask_data = da.from_array(self.dataset.data)
            logger.debug(
                f"created dask_array from layer data {self.dask_data}"
            )
        try:
            self.sizes = self.dataset.metadata["sizes"]
            logger.debug(f"set sizes {self.sizes}")

        except KeyError:
            logger.debug(
                f"generating sizes from shape {self.dataset.data.shape}"
            )
            self.sizes = {
                f"dim-{i}": s for i, s in enumerate(self.dataset.data.shape)
            }
            show_warning(f"No sizes found in metadata")
            logger.debug(f"set sizes {self.sizes}")
        logger.debug("init_meta")

        try:
            self.path = self.dataset.metadata["path"]
            logger.debug(f"set path {self.path}")
        except KeyError:
            self.path = None
            logger.debug(f"set path to None")
            show_warning(f"No path found in metadata")

        try:
            self.pixel_size_um = self.dataset.metadata["pixel_size_um"]
            logger.debug(f"set pixel_size_um {self.pixel_size_um}")
        except KeyError:
            self.pixel_size_um = None
            logger.debug(f"set pixel_size_um to None")
            show_warning(f"No pixel_size_um found in metadata")

        self.axis_selector.choices = list(
            f"{ax}:{size}" for ax, size in list(self.sizes.items())[:-2]
        )
        logger.debug(f"update choices with {self.axis_selector.choices}")

    def reset_choices(self):
        self.data_widget.reset_choices()
        logger.debug(f"reset choises from input {self.data_widget.choices}")


class SubStack(QWidget):
    def __init__(self, napari_viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self.data_widget = create_widget(
            annotation=Image,
            label="data",
        )
        self.data_widget.label_changed.connect(self.init_data)
        self.data_widget.changed.connect(self.init_data)
        self.input_shape_widget = Label(label="Input Shape:", value="")
        self.out_shape_widget = Label(value="", label="Output Shape:")
        self.crop_it = PushButton(text="Crop it!")
        self.crop_it.clicked.connect(self.make_new_layer)
        self.out_container = Container(
            widgets=[self.out_shape_widget, self.crop_it]
        )
        self.input_container = Container(
            widgets=[self.data_widget, self.input_shape_widget]
        )
        self.viewer.layers.events.inserted.connect(self.reset_choices)
        self.slice_container = Container(scrollable=False, widgets=())

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.input_container.native)
        self.layout.addWidget(self.slice_container.native)
        self.layout.addWidget(self.out_container.native)
        self.layout.addStretch()
        self.setLayout(self.layout)

        self.init_data()

    def init_data(self):
        logger.debug(f"init_data with {self.data_widget.current_choice}")
        try:
            self.dataset = self.viewer.layers[self.data_widget.current_choice]
            self.init_meta()
        except KeyError:
            self.dataset = None
            self.empty_slice_container()
            self.input_shape_widget.value = ""
            self.out_shape_widget.value = ""
            show_warning("No data")

    def make_new_layer(self):
        self.viewer.add_image(
            self.out_dask,
            name=f"Substack {self.crop_coords}",
            metadata={
                "pixel_size_um": self.pixel_size_um,
                "sizes": self.out_sizes,
                "substack_coords": self.crop_coords,
                "source_path": self.path,
            },
        )

    def compute_substack(self):
        logger.debug("Compute substack")
        slices = []
        sizes = {}
        crop_coords = {}
        for item in self.slice_container:
            dim = (
                slice(item.start.value, item.stop.value)
                if (size := item.stop.value - item.start.value) > 1
                else (val := item.start.value)
            )
            slices.append(dim)
            if isinstance(dim, slice):
                sizes[item.label] = size
            else:
                crop_coords[item.label] = val
        logger.debug(f"Slices: {slices}")
        if len(slices) == 6:
            self.out_dask = self.dask_array[
                slices[0],
                slices[1],
                slices[2],
                slices[3],
                slices[4],
                slices[5],
            ]
        elif len(slices) == 5:
            self.out_dask = self.dask_array[
                slices[0],
                slices[1],
                slices[2],
                slices[3],
                slices[4],
            ]
        elif len(slices) == 4:
            self.out_dask = self.dask_array[
                slices[0], slices[1], slices[2], slices[3]
            ]
        elif len(slices) == 3:
            self.out_dask = self.dask_array[slices[0], slices[1], slices[2]]
        elif len(slices) == 2:
            self.out_dask = self.dask_array[slices[0], slices[1]]
        else:
            show_warning(f"Problem with substack. Slices: {slices}")
            self.out_dask = self.dask_array
            logger.debug(
                f"Problem with substack. Slices: {slices}. Out Dask stays the same ass input {self.out_dask}"
            )

        logger.debug(f"Out dask: {self.out_dask}")
        self.out_shape_widget.value = self.out_dask.shape
        self.out_sizes = sizes
        self.crop_coords = crop_coords

    def populate_dims(self):
        logger.debug("populate_dims")
        if self.sizes is None:
            logger.debug("populate_dims: no sizes")
            return
        self.empty_slice_container()
        self.input_shape_widget.value = self.dask_array.shape
        for name, size in self.sizes.items():
            if size:
                logger.debug(f"add {name} of size {size} to the container")
                self.slice_container.append(
                    SliceEdit(max=size, min=0, stop=size, label=name)
                )
            else:
                logger.debug(f"skip {name} of size {size}")
        logger.debug(self.slice_container.asdict())

        self.compute_substack()
        self.slice_container.changed.connect(self.compute_substack)

    def empty_slice_container(self):
        logger.debug("empty_slice_container")
        self.slice_container.clear()

    def init_meta(self):
        logger.debug(f"init_meta for {self.data_widget.current_choice}")
        if self.dataset is None:
            logger.debug("no dataset")
            self.sizes = None
            self.path = None
            logger.debug("set sizes and path to None")

            return
        try:
            self.sizes = self.dataset.metadata["sizes"]
            logger.debug(f"set sizes {self.sizes}")

        except KeyError:
            logger.debug(
                f"generating sizes from shape {self.dataset.data.shape}"
            )
            self.sizes = {
                f"dim-{i}": s for i, s in enumerate(self.dataset.data.shape)
            }
            show_warning(f"No sizes found in metadata")
            logger.debug(f"set sizes {self.sizes}")

        try:
            self.path = self.dataset.metadata["path"]
            logger.debug(f"set path {self.path}")
        except KeyError:
            self.path = None
            logger.debug(f"set path to None")
            show_warning(f"No path found in metadata")

        try:
            self.pixel_size_um = self.dataset.metadata["pixel_size_um"]
            logger.debug(f"set pixel_size_um {self.pixel_size_um}")
        except KeyError:
            self.pixel_size_um = None
            logger.debug(f"set pixel_size_um to None")
            show_warning(f"No pixel_size_um found in metadata")
        try:
            self.dask_array = self.dataset.metadata["dask_array"]
            logger.debug(f"dask array from metadata {self.dask_array}")

        except KeyError:
            if not isinstance(self.dataset.data, da.Array):
                self.dask_array = da.from_array(self.dataset.data)
            else:
                self.dask_array = self.dataset.data
            show_warning(
                f"No dask_array found in metadata, creating one {self.dask_array}"
            )
            logger.debug(f"dask array from array {self.dask_array}")

        self.out_dask = self.dask_array
        self.populate_dims()
        self.update_axis_labels()

    def update_axis_labels(self):
        logger.debug("Update axis labels")
        labels = list(self.sizes)
        if len(labels) > len(self.dataset.data.shape):  # axis_channel used
            labels = list(filter(lambda a: a != "C", labels))
            logger.debug("exclude 'C' ")

        self.viewer.dims.axis_labels = labels
        logger.debug(f"new labels: {self.viewer.dims.axis_labels}")

    def reset_choices(self):

        self.data_widget.reset_choices()
        logger.debug(f"reset choises from input {self.data_widget.choices}")
        self.init_meta()


@magic_factory(auto_call=True)
def make_matrix(
    Manual_ref_line: Shapes,
    n_cols: int = 5,
    n_rows: int = 5,
    row_multiplier: float = 1.0,
    diagonal_elements: bool = True,
    limit_to_a_slice: bool = False,
    size: int = 300,
) -> napari.types.LayerDataTuple:
    logger.debug(f"Ref line: {Manual_ref_line.data[0]}")
    logger.debug(f"Scale: {Manual_ref_line.scale}")
    manual_points = Manual_ref_line.data[0] * Manual_ref_line.scale
    logger.debug(f"manual_points: {manual_points}")

    assert len(manual_points == 2), "Select a line along your wells"
    manual_period = manual_points[1] - manual_points[0]
    logger.debug(f"manual_period: {manual_period}")
    col_period = manual_period / (n_cols - 1)
    logger.debug(f"col_period: {col_period}")

    row_period = np.zeros_like(col_period)
    row_period[-2:] = np.array([col_period[-1], -col_period[-2]])
    logger.debug(f"row_period: {row_period}")
    extrapolated_wells = np.stack(
        [
            manual_points[0]
            + col_period * i
            + row_period * j * row_multiplier
            + (col_period + row_period * row_multiplier) / 2 * k
            for k in range(2 if diagonal_elements else 1)
            for i in range(n_cols)
            for j in range(n_rows)
        ]
    )
    logger.debug(f"extrapolated_wells: {extrapolated_wells}")

    out = (
        extrapolated_wells[:, -2:],
        {
            "name": "ROIs",
            "symbol": "square",
            "size": size,
            "edge_color": "#ff0000",
            "face_color": "#00000000",
        },
        "points",
    )
    logger.debug(f"Returning: {out}")
    return out


@magic_factory
def crop_rois(
    stack: Image,
    ROIs: Points,
) -> napari.types.LayerDataTuple:
    if any([stack is None, ROIs is None]):
        return
    data = stack.data.copy()
    meta = stack.metadata.copy()
    scale = stack.scale.copy()
    no_dim_scale = scale.max()
    centers = ROIs.data / no_dim_scale
    size = (ROIs.size // no_dim_scale).max()

    _crops = map(partial(crop_stack, stack=data, size=size), centers)
    axis = 1 if data.ndim > 3 else 0
    good_crops = list(filter(lambda a: a is not None, _crops))
    try:
        meta["sizes"] = update_dict_with_pos(
            meta["sizes"], axis, "P", len(good_crops)
        )
    except KeyError:
        logger.error(
            rf"Failed updating meta[`sizes`] with \{'P': {len(good_crops)}\}"
        )
    meta["sizes"]["X"] = size
    meta["sizes"]["Y"] = size
    meta["crop_centers"] = centers
    meta["crop_size"] = size

    return (
        da.stack(good_crops, axis=axis),
        {"scale": scale, "metadata": meta},
        "image",
    )


def update_dict_with_pos(input_dict: dict, pos, key, value):
    """Inserts {key: value} into position"""
    k = list(input_dict.keys())
    k.insert(pos, key)
    v = list(input_dict.values())
    v.insert(pos, value)
    return {kk: vv for kk, vv in zip(k, v)}


def crop_stack(center: np.ndarray, stack: np.ndarray, size: int) -> np.ndarray:
    """
    Crops a square of the size `size` px from last two axes accrding to
    2 last coordinates of the center.
    Returns stack[...,size,size] if crop fits into the stack size, otherwise returns None.
    """
    assert stack.ndim >= 2
    assert all(
        [center.ndim == 1, len(center) >= 2]
    ), f"Problem with center {center} of len {len(center)}"
    s = (size // 2).astype(int)
    y, x = center[-2:].astype(int)
    y1, y2 = y - s, y + s
    x1, x2 = x - s, x + s
    ylim, xlim = stack.shape[-2:]

    if any([y1 < 0, x1 < 0, y2 > ylim, x2 > xlim]):
        return
    return stack[..., y1:y2, x1:x2]
