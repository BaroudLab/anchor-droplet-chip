import logging
import logging.config

import dask.array as da
import napari
from magicgui.widgets import (
    ComboBox,
    Container,
    PushButton,
    RadioButtons,
    create_widget,
)
from napari.layers import Image
from napari.utils.notifications import show_warning
from qtpy.QtWidgets import QVBoxLayout, QWidget

from ._sub_stack import SubStack

logger = logging.getLogger(__name__)


OPS = ["max", "mean", "min"]


class ProjectAlong(QWidget):
    def __init__(self, napari_viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self.data_widget = create_widget(
            annotation=Image,
            label="data",
        )
        self.data_widget.changed.connect(self.init_data)
        self.axis_selector = RadioButtons(
            label="Choose axis", orientation="horizontal", choices=()
        )
        self.op_widget = ComboBox(label="op", choices=OPS)
        self.run_btn = PushButton(text="Make projection!")
        self.run_btn.clicked.connect(self.make_projection)

        self.input_container = Container(
            widgets=[
                self.data_widget,
                self.axis_selector,
                self.op_widget,
                self.run_btn,
            ]
        )
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.input_container.native)
        self.layout.addStretch()
        self.setLayout(self.layout)

        self.init_data()

    def make_projection(self):
        logger.info(f"Making projection of dask array {self.dask_data.shape}")
        axis_sel = self.axis_selector.current_choice
        letter, size = axis_sel.split(":")
        self.total = int(size)
        axis = self.axis_selector.choices.index(axis_sel)

        self.meta = self.selected_layer.metadata.copy()

        selected_op = self.op_widget.current_choice

        self.projection = self.dask_data.__getattribute__(selected_op)(
            axis=axis
        )

        _ = self.meta["sizes"].pop(letter)
        logger.info(f"Projection result: {self.projection}")

        labels = list(self.meta["sizes"])
        try:
            channel_axis = labels.index("C")
        except ValueError:
            channel_axis = None
        self.meta["dask_data"] = self.projection
        ddata = self.projection

        return self.viewer.add_image(
            ddata
            if max(ddata.shape) < 4000
            else [ddata[..., :: 2**i, :: 2**i] for i in range(4)],
            channel_axis=channel_axis,
            name=self.selected_layer.name + "_" + selected_op + letter,
            metadata=self.meta,
        )

    def init_data(self):
        try:
            self.selected_layer = self.viewer.layers[
                self.data_widget.current_choice
            ]
        except KeyError:
            logger.debug("no data in selected_layer")
            self.sizes = None
            self.path = None
            logger.debug("set sizes and path to None")

            return

        try:
            self.dask_data = self.selected_layer.metadata["dask_data"]
            logger.debug(f"Found dask_data in layer metadata {self.dask_data}")
        except KeyError:
            logger.debug(
                f"No dask_data in layer metadata {self.selected_layer.metadata}"
            )
            self.dask_data = da.asarray(self.selected_layer.data)
            logger.debug(
                f"created dask_array from layer data {self.dask_data}"
            )
        try:
            self.sizes = self.selected_layer.metadata["sizes"]
            logger.debug(f"set sizes {self.sizes}")

        except KeyError:
            logger.debug(
                f"generating sizes from shape {self.selected_layer.data.shape}"
            )
            self.sizes = {
                f"dim-{i}": s
                for i, s in enumerate(self.selected_layer.data.shape)
            }
            show_warning(f"No sizes found in metadata")
            logger.debug(f"set sizes {self.sizes}")
        logger.debug("init_meta")

        try:
            self.path = self.selected_layer.metadata["path"]
            logger.debug(f"set path {self.path}")
        except KeyError:
            self.path = None
            logger.debug(f"set path to None")
            show_warning(f"No path found in metadata")

        try:
            self.pixel_size_um = self.selected_layer.metadata["pixel_size_um"]
            logger.debug(f"set pixel_size_um {self.pixel_size_um}")
        except KeyError:
            self.pixel_size_um = None
            logger.debug(f"set pixel_size_um to None")
            show_warning(f"No pixel_size_um found in metadata")

        self.axis_selector.choices = list(
            f"{ax}:{size}" for ax, size in list(self.sizes.items())[:-2]
        )
        logger.debug(f"update choices with {self.axis_selector.choices}")

        SubStack.update_axis_labels(
            self.sizes, self.selected_layer.data, self.viewer.dims
        )

    def reset_choices(self):
        self.data_widget.reset_choices()
        logger.debug(f"reset choises from input {self.data_widget.choices}")
