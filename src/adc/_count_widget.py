import json
import logging
import os
from asyncio.log import logger

import dask.array as da
import numpy as np
import pandas as pd
from functools import partial

from magicgui.widgets import Container, create_widget
from napari import Viewer
from napari.layers import Image, Points
from napari.utils import progress
from napari.utils.notifications import show_error, show_info
from qtpy.QtWidgets import QLineEdit, QPushButton, QVBoxLayout, QWidget

from adc import count

from ._align_widget import DROPLETS_CSV_SUFFIX

TABLE_NAME = "table.csv"

COUNTS_LAYER_PROPS = dict(
    name="Counts", size=300, face_color="#00000000", border_color="#00880088"
)
COUNTS_JSON_SUFFIX = ".counts.json"

DETECTION_LAYER_PROPS = dict(
    name="Detections",
    size=20,
    face_color="#ffffff00",
    border_color="#ff007f88",
)



DETECTION_CSV_SUFFIX = ".detections.csv"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CountCells(QWidget):
    "Detects cells in TRITC"

    def __init__(self, napari_viewer: Viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self.select_TRITC = create_widget(
            annotation=Image,
            label="TRITC",
        )
        
        # Add parameter widgets as text inputs
        self.size_widget = create_widget(
            value=300, 
            label="size",
            widget_type="LineEdit"
        )
        self.dif_gauss_sigma_widget = create_widget(
            value="(3, 5)",
            label="dif_gauss_sigma",
            widget_type="LineEdit"
        )
        self.min_distance_widget = create_widget(
            value=3,
            label="min_distance",
            widget_type="LineEdit"
        )
        self.threshold_abs_widget = create_widget(
            value=2.0,
            label="threshold_abs",
            widget_type="LineEdit"
        )
        
        self.select_centers = create_widget(label="centers", annotation=Points)
        self.container = Container(
            widgets=[
                self.select_TRITC, 
                self.select_centers,
                self.size_widget,
                self.dif_gauss_sigma_widget,
                self.min_distance_widget,
                self.threshold_abs_widget
            ]
        )
        self.out = []
        self.counts_layer = None
        self.detections_layer = None

        self.out_path = ""
        self.output_filename_widget = QLineEdit("path")
        self.btn = QPushButton("Localize!")
        self.btn.clicked.connect(self.process_stack)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.container.native)
        self.layout.addWidget(self.btn)
        self.layout.addStretch()

        self.setLayout(self.layout)

    def process_stack(self):
        self._pick_data_ref()
        self._pick_centers()

        show_info("Data loaded. Counting")

        self.viewer.window._status_bar._toggle_activity_dock(True)

        self._update_detections()

    def _pick_data_ref(self):
        "Get dask array to know the shape etc"
        self.selected_layer = self.viewer.layers[
            self.select_TRITC.current_choice
        ]
        logger.debug(f"selected_layer: {self.selected_layer}")
        if self.selected_layer.multiscale:
            self.ddata_ref = self.selected_layer.data[0]
            logger.debug(
                f"multiscale data: selecting highest resolution: {self.ddata_ref}"
            )
        else:
            self.ddata_ref = self.selected_layer.data
            logger.debug(f"not multiscale data: {self.ddata_ref}")

    def _pick_centers(self):
        self.centers_layer = self.viewer.layers[
            self.select_centers.current_choice
        ]
        self.centers = self.centers_layer.data
        logger.debug(f"selected centers: {len(self.centers)}")

    def _update_detections(self):
        logger.debug("Creating output layers")
        self.detections_layer = self.viewer.add_points(
            data=[[0] * self.ddata_ref.ndim], **DETECTION_LAYER_PROPS
        )
        self.counts_layer = self.viewer.add_points(
            data=[[0] * self.ddata_ref.ndim], text=[], **COUNTS_LAYER_PROPS
        )
        logger.debug("Creating worker")
        
        # Get parameter values from widgets and parse them
        size = int(self.size_widget.value)
        dif_gauss_sigma = eval(self.dif_gauss_sigma_widget.value)  # Parse tuple string
        min_distance = int(self.min_distance_widget.value)
        threshold_abs = float(self.threshold_abs_widget.value)
        
        # Create partial function with custom parameters
        # First, create a customized get_peaks with our parameters
        custom_get_peaks = partial(
            count.get_peaks,
            dif_gauss_sigma=dif_gauss_sigma,
            min_distance=min_distance,
            threshold_abs=threshold_abs
        )
        
        # Then, create a customized get_global_peaks that uses our custom_get_peaks
        custom_localizer = partial(
            count.get_global_peaks,
            localizer=custom_get_peaks
        )
        
        self.out = count.count_recursive(
            data=self.ddata_ref,
            positions=self.centers,
            size=size,
            progress=progress,
            localizer=custom_localizer
        )

        self.save_results()

    def save_results(self):
        show_info("Done localizing ")

        locs, n_peaks_per_well, drops, table_df = self.out

        try:
            path = self.selected_layer.source.path
            if path is None:
                try:
                    path = self.selected_layer.metadata["path"]
                except KeyError:
                    show_error("Unable to find the path")
                    return
            self.detections_layer.save(
                ppp := os.path.join(path, DETECTION_CSV_SUFFIX)
            )
        except Exception as e:
            logger.debug(f"Unable to save detections inside the zarr: {e}")
            logger.debug(f"Saving in a separate file")
            self.detections_layer.save(
                ppp := os.path.join(path + DETECTION_CSV_SUFFIX)
            )
        logger.info(f"Saving detections into {ppp}")

        try:
            with open(
                ppp := os.path.join(path, COUNTS_JSON_SUFFIX), "w"
            ) as fp:
                json.dump(n_peaks_per_well, fp, indent=2)
        except Exception as e:
            logger.debug(f"Unable to save counts inside the zarr: {e}")
            logger.debug(f"Saving in a separate file")

            with open(ppp := path + COUNTS_JSON_SUFFIX, "w") as fp:
                json.dump(n_peaks_per_well, fp, indent=2)
        logger.info(f"Saving counts into {ppp}")

        try:
            ppp = os.path.join(path, DROPLETS_CSV_SUFFIX)
            droplets_df = pd.DataFrame(
                data=drops, columns=[f"axis-{i}" for i in range(len(drops[0]))]
            )
            droplets_df.to_csv(ppp)
        except Exception as e:
            logger.debug(f"Unable to save droplets inside the zarr: {e}")
            logger.debug(f"Saving in a separate file")

            droplets_df.to_csv(ppp := path + DROPLETS_CSV_SUFFIX)
        logger.info(f"Saving counts into {ppp}")

        try:
            ppp = os.path.join(os.path.dirname(path), TABLE_NAME)

            table_df.to_csv(ppp)
            logger.info(f"Saving table into {ppp}")
        except Exception as e:
            logger.error(f"Unable to save table into {ppp}: {e}")

        print("locs type:", type(locs), "shape:", getattr(locs, "shape", None))
        print("drops type:", type(drops), "shape:", getattr(drops, "shape", None))
        print("n_peaks_per_well:", n_peaks_per_well)
        
        self.detections_layer.data = locs
        self.counts_layer.data = drops
        self.counts_layer.text = n_peaks_per_well

    def show_counts(self, counts):
        self.counts = counts
        logger.debug(counts)

    def _update_path(self):
        BF = self.select_BF.current_choice
        TRITC = self.select_TRITC.current_choice
        maxz = "maxZ" if self.zmax_box.checkState() > 0 else ""
        self.out_path = "_".join((BF, TRITC, maxz)) + ".zarr"
        logger.debug(self.out_path)
        self.output_filename_widget.setText(self.out_path)
        self._combine(dry_run=True)

    def reset_choices(self, event=None):
        self.select_centers.reset_choices(event)
        self.select_TRITC.reset_choices(event)