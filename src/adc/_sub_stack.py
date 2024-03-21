import logging
import logging.config

import dask.array as da
import napari
from magicgui.widgets import (
    Container,
    Label,
    PushButton,
    SliceEdit,
    create_widget,
)
from napari.layers import Image
from napari.utils.notifications import show_warning
from qtpy.QtWidgets import QVBoxLayout, QWidget

logger = logging.getLogger(__name__)


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
            self.selected_layer = self.viewer.layers[
                self.data_widget.current_choice
            ]
            self.init_meta()
        except KeyError:
            self.selected_layer = None
            self.empty_slice_container()
            self.input_shape_widget.value = ""
            self.out_shape_widget.value = ""
            show_warning("No data")

    def make_new_layer(self):
        labels = list(self.out_sizes)
        try:
            channel_axis = labels.index("C")
        except ValueError:
            channel_axis = None
        try:
            names = (
                f"{nnn}_{self.crop_coords}"
                for nnn in self.selected_layer.metadata["names"]
            )
        except KeyError:
            names = f"{self.selected_layer.name}_{self.crop_coords}"

        try:
            colormap = self.selected_layer.metadata["colormap"]
        except KeyError:
            colormap = None
        try:
            contrast_limits = self.selected_layer.metadata["contrast_limits"]
        except KeyError:
            contrast_limits = None

        return self.viewer.add_image(
            self.out_dask,
            name=names,
            channel_axis=channel_axis,
            contrast_limits=contrast_limits,
            colormap=colormap,
            metadata={
                "pixel_size_um": self.pixel_size_um,
                "sizes": self.out_sizes,
                "substack_coords": self.crop_coords,
                "path": self.path,
                "dask_data": self.out_dask,
            },
        )

    def compute_substack(self):
        logger.debug("Compute substack")
        slices = []
        sizes = {}
        crop_coords = {}
        for item in self.slice_container:
            start = item.start.value
            stop = item.stop.value
            step = item.step.value
            dim = (
                slice(start, stop, step)
                if (size := stop - start) > 1
                else start
            )
            slices.append(dim)
            if isinstance(dim, slice):
                sizes[item.label] = size // step + (1 if step > 1 else 0)
            else:
                crop_coords[item.label] = start
            if size // step < self.sizes[item.label]:
                crop_coords[item.label] = f"{start}-{stop}" + (
                    f"-{step}" if step > 1 else ""
                )

        try:
            logger.debug(f"Slices: {slices}")
            self.out_dask = self.dask_array[tuple(slices)]
        except Exception as e:
            show_warning(f"Problem with substack. Slices: {slices}, Exc: {e}")
            self.out_dask = self.dask_array
            logger.debug(
                f"Problem with substack. Slices: {slices}. Out Dask stays the same as input {self.out_dask}"
            )

        logger.debug(f"Out dask: {self.out_dask}")
        self.out_shape_widget.value = self.out_dask.shape
        self.out_sizes = sizes
        self.crop_coords = "_".join(
            [f"{k}={v.replace(':','-')}" for k, v in crop_coords.items()]
        )

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
        if self.selected_layer is None:
            logger.debug("no dataset")
            self.sizes = None
            self.path = None
            logger.debug("set sizes and path to None")

            return
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
        try:
            self.dask_array = self.selected_layer.metadata["dask_data"]
            logger.debug(f"dask array from metadata {self.dask_array}")

        except KeyError:
            if self.selected_layer.multiscale:
                data = self.selected_layer.data[0]
            else:
                data = self.selected_layer.data

            if not isinstance(data, da.Array):
                self.dask_array = da.from_array(data)
            else:
                self.dask_array = data
            show_warning(
                f"No dask_data found in metadata {self.selected_layer.metadata}, creating one {self.dask_array}"
            )
            logger.debug(f"dask array from array {self.dask_array}")

        self.out_dask = self.dask_array
        self.populate_dims()
        self.update_axis_labels(
            self.sizes, self.selected_layer.data, self.viewer.dims
        )

    @staticmethod
    def update_axis_labels(sizes, data, dims):
        labels = list(sizes)
        logger.debug(f"Update axis labels: {labels}")

        logger.debug(f"data.shape {data.shape}")
        if len(labels) > len(data.shape):  # axis_channel used
            labels = list(filter(lambda a: a != "C", labels))
            logger.debug("exclude 'C' ")
        else:
            logger.debug(
                f"data.shape {data.shape} same len as labels {labels}"
            )

        dims.__setattr__("axis_labels", labels)
        logger.debug(f"new labels: {dims.axis_labels}")

    def reset_choices(self):
        self.data_widget.reset_choices()
        logger.debug(f"reset choises from input {self.data_widget.choices}")
        self.init_meta()
