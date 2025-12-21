import logging

import dask.array as da
import numpy as np
from magicgui.widgets import Container, create_widget
from napari import Viewer
from napari.layers import Image
from napari.utils import progress
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget

from adc import count

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PreviewGaussianDifference(QWidget):
    "Preview Gaussian Difference filter on TRITC image"

    def __init__(self, napari_viewer: Viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        
        self.select_TRITC = create_widget(
            annotation=Image,
            label="TRITC",
        )
        
        # Add parameter widget for dif_gauss_sigma
        self.dif_gauss_sigma_widget = create_widget(
            value="(3, 5)",
            label="dif_gauss_sigma",
            widget_type="LineEdit"
        )
        
        self.container = Container(
            widgets=[
                self.select_TRITC,
                self.dif_gauss_sigma_widget
            ]
        )
        
        self.gdif_layer = None
        
        self.btn = QPushButton("Preview Gaussian Difference")
        self.btn.clicked.connect(self.process_stack)
        
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.container.native)
        self.layout.addWidget(self.btn)
        self.layout.addStretch()
        
        self.setLayout(self.layout)

    def process_stack(self):
        self._pick_data_ref()
        
        show_info("Computing Gaussian Difference")
        
        self.viewer.window._status_bar._toggle_activity_dock(True)
        
        self._compute_and_display_gdif()

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

    def _compute_and_display_gdif(self):
        logger.debug("Computing Gaussian Difference recursively")
        
        # Get parameter value from widget
        dif_gauss_sigma = eval(self.dif_gauss_sigma_widget.value)
        
        # Compute recursively
        gdif_result = self._apply_gdif_recursive(
            data=self.ddata_ref,
            dif_gauss_sigma=dif_gauss_sigma,
            progress=progress
        )
        
        # Remove old gdif layer if it exists
        if self.gdif_layer is not None:
            try:
                self.viewer.layers.remove(self.gdif_layer)
            except ValueError:
                pass
        
        # Add new layer
        self.gdif_layer = self.viewer.add_image(
            gdif_result,
            name=f"Gaussian Diff {dif_gauss_sigma}",
            colormap="turbo",
            blending="additive"
        )
        self.viewer.window._status_bar._toggle_activity_dock(False)

        show_info(f"Gaussian Difference computed with sigma={dif_gauss_sigma}")

    def _apply_gdif_recursive(
        self,
        data: da.Array,
        dif_gauss_sigma: tuple,
        progress=progress
    ):
        """
        Recursively apply Gaussian difference filter to n-dimensional data.
        Returns dask array of the same shape with gdif applied to each 2D slice.
        """
        logger.debug(f"Processing data shape: {data.shape}, ndim: {data.ndim}")
        
        if data.ndim > 2:
            # Process recursively along first dimension
            result_list = []
            for i, d in enumerate(progress(data)):
                logger.debug(f"Processing slice {i}")
                gdif_slice = self._apply_gdif_recursive(
                    data=d,
                    dif_gauss_sigma=dif_gauss_sigma,
                    progress=progress
                )
                result_list.append(gdif_slice)
            
            # Stack results back together
            return da.stack(result_list, axis=0)
        else:
            # Base case: apply gdif to 2D array
            logger.debug(f"Applying gdif to 2D slice with sigma={dif_gauss_sigma}")
            
            # Apply gdif - need to compute for 2D slices
            if isinstance(data, da.Array):
                data_computed = data.compute()
            else:
                data_computed = data
            
            gdif_result = count.gdif(data_computed, dif_gauss_sigma=dif_gauss_sigma)
            
            return da.from_array(gdif_result)

    def reset_choices(self, event=None):
        self.select_TRITC.reset_choices(event)