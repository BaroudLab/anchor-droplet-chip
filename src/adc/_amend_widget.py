import logging
import re
from asyncio.log import logger
from collections import defaultdict

import dask.array as da
import numpy as np
import pandas as pd
import requests
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    PushButton,
    TextEdit,
    create_widget,
)
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
        self.select_droplets = create_widget(
            label="droplets", annotation=Points
        )
        self.features_widget = TextEdit(label="features")
        self.text_widget = TextEdit(label="buffer")
        self.container = Container(
            widgets=[
                self.select_droplets,
                self.select_labels,
                self.text_widget,
            ]
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
        self.horigin = {
            make_hash(o): i for i, o in enumerate(self.original_droplet_set)
        }
        self.selected_droplet_layer.events.data.connect(self.callback)

        self.table_path = self.selected_droplet_layer.metadata["path"]
        self.norm_path = make_path(self.table_path)
        print(self.norm_path)
        res = requests.get(
            "https://nocodb01.pasteur.fr/api/getfeatures",
            params={"path": self.norm_path},
            timeout=5,
        ).json()
        print(res)
        self.grouped_features, self.n_droplets_per_chip = group_features(
            res["features"], self.original_droplet_set
        )
        self.all_features = res["all_features"]
        self.widgets = {}
        for feature in self.grouped_features:
            self.widgets[feature] = (
                c := CheckBox(
                    text=f"{feature}: ({len(self.grouped_features[feature])})",
                    value=True,
                )
            )
            c.changed.connect(self.update_viewer)
        for feature in self.all_features:
            name = feature["name"]
            if name not in self.widgets:
                self.widgets[name] = (
                    c := CheckBox(text=f"{name}: (0)", value=False)
                )

        self.grouped_checkboxes = Container(widgets=self.widgets.values())
        self.container.insert(2, self.grouped_checkboxes)
        self.features_widget.value = self.grouped_features

        self.mark_as_combo = ComboBox(
            choices=[f["name"] for f in self.all_features], name="Mark as:"
        )
        self.container.append(self.mark_as_combo)

        self.mark_as_btn = PushButton(name="Apply Label")
        self.mark_as_btn.clicked.connect(self.apply_labels)
        self.container.append(self.mark_as_btn)

    def apply_labels(self):
        selected_feature = self.mark_as_combo.current_choice
        selected_droplets = self.text_widget.value
        print(
            f"Apply `{selected_feature}` to the droplets `{selected_droplets}`"
        )

    def update_viewer(self, event):
        print(event)
        print([c.value for c in self.widgets.values()])
        boxes = {c: self.widgets[c].value for c in self.widgets}
        self.selected_droplet_layer.data = [
            c
            for i, c in enumerate(self.original_droplet_set)
            for feature in self.grouped_features
            if (c[0], i % self.n_droplets_per_chip)
            in self.grouped_features[feature]
            and boxes[feature]
        ]

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


def make_path(path):
    "Matches the path to the database"
    out = re.compile(r"Multicell/(.*)/((final_table.csv)|(day))").findall(path)
    return out[0][0]


def group_features(features: list, original_droplet_set):
    """
    Groups feature list by feature name
    feture is the list of dicts with the fields:
        [{droplet_id': 115, 'feature_id': 5, 'feature_name': 'negative', 'stack': 0},...]
    return:
        default_dict({"feature_name": {(stack, droplet_id), ... }, ...}
    """
    n_droplets_per_chip = len(original_droplet_set) / len(
        np.unique(original_droplet_set[:, 0])
    )

    fff = defaultdict(set)
    all_features = []
    for f in features:
        id = (int(f["stack"]), int(f["droplet_id"]))
        fff[f["feature_name"]].add(id)
        all_features.append(id)

    for i, (chip, y, x) in enumerate(original_droplet_set):
        droplet = i % n_droplets_per_chip
        if (chip, droplet) not in all_features:
            fff["unlabeled"].add((chip, i % n_droplets_per_chip))
    return fff, n_droplets_per_chip
