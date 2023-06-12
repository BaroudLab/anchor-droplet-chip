import logging
import os
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

API_ENDPOINT = "https://nocodb01.pasteur.fr/api/getfeatures"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class AmendDroplets(QWidget):
    "Detects cells in TRITC"

    def __init__(self, napari_viewer: Viewer, endpoint=API_ENDPOINT) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self.endpoint = endpoint
        self.select_labels = create_widget(
            annotation=Image,
            label="Labels",
        )

        self.radius = 300
        self.select_droplets = create_widget(
            label="droplets", annotation=Points
        )
        self.select_droplets.changed.connect(self.load_table)

        self.features_widget = TextEdit(label="features")
        self.buffer_widget = TextEdit(label="buffer")
        self.container = Container(
            widgets=[
                self.select_droplets,
                self.select_labels,
                self.buffer_widget,
            ]
        )

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.container.native)
        self.layout.addStretch()

        self.setLayout(self.layout)
        self.grouped_features = defaultdict(set)
        self.n_droplets_per_chip = 0
        self.all_features = []
        self.feature_list = []
        self.widgets = {}
        self.buffer = []
        self.new_droplet_set = []
        self.deleted_droplets = []
        self.df = None
        self.table_path = ""
        self.save_path = ""
        self.selected_droplet_layer = None
        self.original_droplet_set = None
        self.features_loaded = False
        self.currently_visible_hash = []

        self.load_table()

    def load_table(self):
        self.selected_droplet_layer = self.viewer.layers[
            self.select_droplets.current_choice
        ]
        self.selected_droplet_layer.events.data.connect(self.callback)

        self.original_droplet_set = self.selected_droplet_layer.data.copy()
        self.new_droplet_set = self.selected_droplet_layer.data.copy()

        self.horigin = {
            make_hash(o): global_index
            for global_index, o in enumerate(self.original_droplet_set)
        }

        self.currently_visible_hash = self.horigin.copy()

        self.df = self.selected_droplet_layer.metadata["data"]
        try:
            self.n_droplets_per_chip = len(self.original_droplet_set) / len(
                self.df.chip.unique()
            )
        except AttributeError:
            print(f"Not a good table, select layer with Final_table.csv")
            return

        self.table_path = self.selected_droplet_layer.metadata["path"]
        self.save_path = self.table_path.replace(".csv", "-features.csv")
        assert self.save_path != self.table_path
        if not os.path.exists(self.save_path):
            self.df.to_csv(self.save_path)
            print(f"duplicated table {self.save_path}")
        else:
            self.df = pd.read_csv(self.save_path, index_col=0)
            print(f"Found table with features, loading {self.save_path}")

        if not self.features_loaded:
            self.load_features()

    def load_features(self):
        self.widgets = {}

        try:
            self.norm_path = make_path(self.table_path)
        except IndexError:
            print(
                "table path is not good, select a layer with final_table.csv data"
            )
            return
        print(self.norm_path)
        res = requests.get(
            self.endpoint,
            params={"path": self.norm_path},
            timeout=5,
        ).json()
        print(res)
        self.feature_list = res["features"]
        self.group_features()
        self.all_features = res["all_features"]

        self.create_checkboxes()

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

        self.undo_btn = PushButton(name="Undo")
        self.undo_btn.clicked.connect(self.undo)
        self.container.append(self.undo_btn)
        self.undo_btn.visible = False

        self.features_loaded = True

    def create_checkboxes(self):
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
                c.changed.connect(self.update_viewer)

    def update_checkboxes(self):
        for feature in self.grouped_features:
            self.widgets[
                feature
            ].text = f"{feature}: ({len(self.grouped_features[feature])})"

    def apply_labels(self):
        selected_feature = self.mark_as_combo.current_choice
        selected_droplets = self.buffer
        print(
            f"Apply `{selected_feature}` to the droplets `{selected_droplets}`"
        )
        for global_index in selected_droplets:
            i = int(global_index)
            n = self.n_droplets_per_chip
            self.feature_list.append(
                {
                    "droplet_id": int(i % n),
                    "feature_name": selected_feature,
                    "stack": int(i // n),
                }
            )

        self.group_features()
        self.update_checkboxes()
        self.widgets[selected_feature].value = True
        print(self.feature_list)
        print(len(self.grouped_features["unlabeled"]))

        self.df.loc[selected_droplets, f"feature:{selected_feature}"] = True
        self.df.to_csv(self.save_path)
        print("updated csv file ")
        self.buffer = []

    def update_viewer(self):
        print("Checkbox clicked", [c.value for c in self.widgets.values()])
        boxes = {c: self.widgets[c].value for c in self.widgets}
        self.selected_droplet_layer.data = [
            c
            for i, c in enumerate(self.original_droplet_set)
            for feature in self.grouped_features
            if (c[0], i % self.n_droplets_per_chip)
            in self.grouped_features[feature]
            and boxes[feature]
        ]
        self.buffer = []
        self.buffer_widget.value = []
        self.currently_visible_hash = [
            make_hash(o) for o in self.selected_droplet_layer.data
        ]

    def callback(self, event):
        self.new_droplet_set = event.source.data
        hdata = [make_hash(o) for o in self.new_droplet_set]
        missing_indices = [
            self.horigin[o]
            for o in self.horigin
            if o not in hdata and o in self.currently_visible_hash
        ]
        self.text_widget.value = missing_indices
        self.buffer = missing_indices
        self.deleted_droplets = self.original_droplet_set[missing_indices]
        self.undo_btn.visible = len(self.deleted_droplets) > 0

    def undo(self):
        self.selected_droplet_layer.data = self.original_droplet_set.copy()
        self.group_features()
        self.update_checkboxes()

    def group_features(self):
        """
        Groups feature list by feature name
        feture is the list of dicts with the fields:
            [{droplet_id': 115, 'feature_id': 5, 'feature_name': 'negative', 'stack': 0},...]
        return:
            default_dict({"feature_name": {(stack, droplet_id), ... }, ...}
        """

        grouped_features = defaultdict(set)
        for f in self.all_features:
            grouped_features[f["name"]] = set()

        all_features = []
        for f in self.feature_list:
            stack_id = (int(f["stack"]), int(f["droplet_id"]))
            grouped_features[f["feature_name"]].add(stack_id)
            all_features.append(stack_id)

        for i, (chip, y, x) in enumerate(self.original_droplet_set):
            droplet = int(i % self.n_droplets_per_chip)
            chip = int(chip)
            if (chip, droplet) not in all_features:
                grouped_features["unlabeled"].add((chip, droplet))
        self.grouped_features = grouped_features

    def reset_choices(self, event=None):
        self.select_droplets.reset_choices(event)
        self.select_labels.reset_choices(event)


def make_hash(data):
    return hash(data.sum())


def make_path(path):
    "Matches the path to the database"
    out = re.compile(r"Multicell/(.*)/((final_table.csv)|(day))").findall(path)
    return out[0][0]
