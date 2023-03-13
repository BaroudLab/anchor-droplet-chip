import logging
import logging.config
import os

import dask.array as da
import napari
import numpy as np
from magicgui.widgets import (
    Container,
    FileEdit,
    PushButton,
    RadioButtons,
    Table,
    create_widget,
)
from napari.layers import Image
from napari.utils import progress
from napari.utils.notifications import show_error, show_info, show_warning
from qtpy.QtWidgets import QVBoxLayout, QWidget

logging.config.fileConfig("logging.conf")

logger = logging.getLogger(__name__)


from napari.layers.utils.stack_utils import slice_from_axis
from napari.qt.threading import thread_worker
from tifffile import imwrite

from ._sub_stack import SubStack


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
        self.split_btn.clicked.connect(self.split_data)
        self.save_btn = PushButton(text="Save tifs!")
        self.save_btn.clicked.connect(self.start_export)

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
        self.layout.addStretch()
        self.setLayout(self.layout)

        self.init_data()

    def finished():
        show_info("Saving done!")

    def errored(e: Exception):
        show_error(f"Error saving: {e}")

    def started():
        show_info("Saving started in the background")

    def update_progress(data):
        i, total, path, pr = data
        # show_info(f"{i+1}/{total} saved to {path}")
        pr.update(1)

    def start_export(self):
        logger.info("Start export")

        self.progress = progress(total=len(self.data_list))
        self.worker = self.save_tifs()

    def stop_export(self):
        logger.info("Stop requested")
        self.stop = True

    def abort():
        show_warning("Export aborted")

    @thread_worker(
        connect={
            "started": started,
            "finished": finished,
            "errored": errored,
            "yielded": update_progress,
            "aborted": abort,
        },
    )
    def save_tifs(self):
        data = self.saving_table.data.to_list().copy()
        self.save_btn.label = "STOP"
        self.save_btn.clicked.disconnect()
        self.save_btn.clicked.connect(self.stop_export)
        self.stop = False
        for i, (name, shape, path, _) in enumerate(data):
            if self.stop:
                logger.warning("Manual stop!")
                return "stopped"
            logger.info(f"Saving {name} into {path}")
            try:
                data = self.data_list[i].compute()
                meta = self.meta.copy()
                meta["spacing"] = (px_size := meta["pixel_size_um"])
                meta["unit"] = "um"
                data_formatted_imagej = (
                    np.expand_dims(data, axis=1)
                    if "Z" not in meta["sizes"] and len(data.shape) > 3
                    else data
                )
                imwrite(
                    path,
                    data_formatted_imagej,
                    imagej=True,
                    resolution=(1 / px_size, 1 / px_size),
                    metadata=meta,
                )
                self.saving_table.data[i] = [name, data.shape, path, "Saved!"]
                yield i, self.total, path, self.progress
            except Exception as e:
                logger.error(f"Failed saving {name} into {path}: {e}")
                return False

        self.save_btn.clicked.disconnect()
        self.save_btn.clicked.connect(self.start_export)
        self.progress.close()

    def split_data(self):
        logger.info(f"Splitting dask array {self.dask_data.shape}")
        axis_sel = self.axis_selector.current_choice
        letter, size = axis_sel.split(":")
        self.total = int(size)
        axis = self.axis_selector.choices.index(axis_sel)
        self.meta = self.selected_layer.metadata
        self.data_list = [
            slice_from_axis(array=self.dask_data, axis=axis, element=i)
            for i in range(self.total)
        ]
        logger.info(
            f"Split result: {self.total} arrays of the size {self.data_list[0].shape}"
        )
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
            self.selected_layer = self.viewer.layers[
                self.data_widget.current_choice
            ]
        except KeyError:
            logger.debug("no dataset")
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
        self.save_btn.clicked.disconnect()
        self.save_btn.clicked.connect(self.start_export)

        SubStack.update_axis_labels(
            self.sizes, self.selected_layer.data, self.viewer.dims
        )

    def reset_choices(self):
        self.data_widget.reset_choices()
        logger.debug(f"reset choises from input {self.data_widget.choices}")
