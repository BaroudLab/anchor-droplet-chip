import logging
from functools import partial

import dask.array as da
import napari
import numpy as np
from magicgui import magic_factory
from magicgui.widgets import (
    Container,
    Label,
    PushButton,
    SliceEdit,
    create_widget,
)
from napari.layers import Image, Points, Shapes
from napari.utils.notifications import show_warning
from qtpy.QtWidgets import QVBoxLayout, QWidget

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
        self.viewer.add_image(self.out_dask, name="Crop")

    def compute_substack(self):
        logger.debug("Compute substack")
        slices = [
            slice(item.start.value, item.stop.value)
            if item.stop.value - item.start.value > 1
            else item.start.value
            for item in self.slice_container
        ]
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
                    s := SliceEdit(max=size, min=0, stop=size, label=name)
                )
            else:
                logger.debug(f"skip {name} of size {size}")

        self.compute_substack()
        self.slice_container.changed.connect(self.compute_substack)

    def empty_slice_container(self):
        logger.debug("empty_slice_container")
        self.slice_container.clear()

    def init_meta(self):
        logger.debug("init_meta")
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
            logger.debug(f"generating sizes from shape {self.dataset.data}")
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
        self.viewer.dims.axis_labels = list(
            filter(lambda a: a != "C", list(self.sizes))
        )

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
    data = stack.data
    scale = stack.scale
    no_dim_scale = scale.max()
    centers = ROIs.data / no_dim_scale
    size = (ROIs.size // no_dim_scale).max()

    _crops = map(partial(crop_stack, stack=data, size=size), centers)
    axis = 1 if data.ndim > 3 else 0
    good_crops = filter(lambda a: a is not None, _crops)
    meta = stack.metadata

    return (
        da.stack(good_crops, axis=axis),
        {"scale": scale, "metadata": meta},
        "image",
    )


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
