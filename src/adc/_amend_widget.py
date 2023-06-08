import json
import logging
import os
from asyncio.log import logger

import dask.array as da
import numpy as np
import pandas as pd
from magicgui.widgets import Container, create_widget, TextEdit
from napari import Viewer
from napari.layers import Image, Points
from napari.utils import progress
from napari.utils.notifications import show_error, show_info
from qtpy.QtWidgets import QLineEdit, QPushButton, QVBoxLayout, QWidget

from adc import count

from ._align_widget import DROPLETS_CSV_SUFFIX

TABLE_NAME = "table.csv"

COUNTS_LAYER_PROPS = dict(
    name="Counts", size=300, face_color="#00000000", edge_color="#00880088"
)
COUNTS_JSON_SUFFIX = ".counts.json"

DETECTION_LAYER_PROPS = dict(
    name="Detections",
    size=20,
    face_color="#ffffff00",
    edge_color="#ff007f88",
)
DETECTION_CSV_SUFFIX = ".detections.csv"

AXES = ["frame", "chip", "y", "x"]

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class AmendDroplets(QWidget):
    "Detects cells in TRITC"

    def __init__(self, napari_viewer: Viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self.select_labels = create_widget(
            annotation=Image,
            label="Labels",
        )
        self.radius = 300
        self.select_droplets = create_widget(label="droplets", annotation=Points)
        self.text_widget = TextEdit(label="output")
        self.container = Container(
            widgets=[
                self.select_labels, 
                self.select_droplets,
                self.text_widget]
        )
        
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.container.native)
        self.layout.addStretch()

        self.setLayout(self.layout)

        self.set_callback()

    def set_callback(self):
        self.selected_droplet_layer = self.viewer.layers[
            self.select_droplets.current_choice
        ]
        self.original_droplet_set = self.selected_droplet_layer.data.copy()
        self.new_droplet_set = self.selected_droplet_layer.data.copy()
        self.horigin = {make_hash(o):i for i, o in enumerate(self.original_droplet_set)}
        self.selected_droplet_layer.events.data.connect(self.callback)
        
        
    def callback(self, event):
        self.new_droplet_set = event.source.data
        hdata = [make_hash(o) for o in self.new_droplet_set]
        out = [self.horigin[o] for o in self.horigin if o not in hdata]
        self.text_widget.value = out
        print(out)


    def reset_choices(self, event=None):
        self.select_droplets.reset_choices(event)
        self.select_labels.reset_choices(event)


def make_hash(data):
    return hash(data.sum())
        