import functools
import warnings

import numpy as np

from .dataset import IndentationDataSet


class DataMissingWarning(UserWarning):
    pass


class QMap(object):
    def __init__(self, path_or_dataset, callback=None):
        """Quantitative force spectroscopy map handling

        Parameters
        ----------
        path_or_dataset: str or nanite.IndentationDataSet
            The path to the data file. The data format is determined
            using the extension of the file and the data is loaded
            with the correct method.
        callback: callable or None
            A method that accepts a float between 0 and 1
            to externally track the process of loading the data.
        """
        if isinstance(path_or_dataset, IndentationDataSet):
            ds = path_or_dataset
        else:
            ds = IndentationDataSet(path=path_or_dataset, callback=callback)
        #: Experimental dataset
        self.ds = ds

        # Feature functions
        self._feature_funcs = {
            "data min height": self.feat_data_min_height_measured_um,
            "meta rating": self.feat_meta_rating,
            "meta scan order": self.feat_meta_scan_order,
            "fit young's modulus": self.feat_fit_youngs_modulus,
            }

        #: Available features
        self.features = sorted(self._feature_funcs.keys())

    def _map_grid(self, coords, map_data):
        """Create a 2D map from 1D coordinates and data

        The .jpk-force-map file format stores the map data in a
        seemingly arbitrary way. This method converts a set of
        coordinates and map data values to a 2D map.

        Parameters
        ----------
        coords: list-like (length N) with tuple of ints
            The x- and y-coordinates [px].
        map_data: list-like (length N)
            The data to be mapped.

        Returns
        -------
        x, y: 1d ndarrays
            The x- and y-values that label the axes of the map
        map2d: 2d ndarray
            The ordered map data.

        Notes
        -----
        If the map data is not on a regular grid, then interpolation
        is performed.
        """
        shape = self.shape
        extent = self.extent

        coords = np.array(coords)
        map_data = np.array(map_data)

        xn, yn = int(shape[0]), int(shape[1])

        # Axes labels
        x, dx = np.linspace(extent[0], extent[1], xn,
                            endpoint=False, retstep=True)
        y, dy = np.linspace(extent[2], extent[3], yn,
                            endpoint=False, retstep=True)
        x += dx/2
        y += dy/2

        # Output map
        map2d = np.zeros((yn, xn), dtype=float)*np.nan
        for ii in range(map_data.shape[0]):
            # Determine the coordinate in the output array
            xi, yi = coords[ii]
            # Write to the output array
            map2d[yi, xi] = map_data[ii]

        return x, y, map2d

    @property
    @functools.lru_cache(maxsize=32)
    def extent(self):
        """extent (x1, x2, y1, y2) [µm]"""
        idnt0 = self.ds[0]
        # get extent of the map
        sx = idnt0.metadata["grid size x [µm]"]
        sy = idnt0.metadata["grid size y [µm]"]
        cx = idnt0.metadata["grid center x [µm]"]
        cy = idnt0.metadata["grid center y [µm]"]
        extent = (cx - sx/2, cx + sx/2,
                  cy - sy/2, cy + sy/2,
                  )
        return extent

    @property
    @functools.lru_cache(maxsize=32)
    def shape(self):
        """shape of the map [px]"""
        idnt0 = self.ds[0]
        # get shape of the map
        shape = (idnt0.metadata["grid size x [px]"],
                 idnt0.metadata["grid size y [px]"]
                 )
        return shape

    def feat_data_min_height_measured_um(self, idnt):
        height = np.min(idnt["height (measured)"])
        value = height / unit_scales["µ"]
        return value

    def feat_fit_youngs_modulus(self, idnt):
        if (idnt.fit_properties and idnt.fit_properties["success"]):
            # use cached rating
            value = idnt.fit_properties["params_fitted"]["E"].value
        else:
            msg = "The experimental data has not been fitted. Please call " \
                  + "`idnt.fit_model` manually for {}!".format(idnt)
            warnings.warn(msg, DataMissingWarning)
            value = np.nan

        return value

    def feat_meta_rating(self, idnt):
        if idnt._rating is None:
            msg = "The experimental data has not been rated. Please call " \
                  + "`idnt.rate_quality` manually for {}!".format(idnt)
            warnings.warn(msg, DataMissingWarning)
            value = np.nan
        else:
            # use cached rating
            value = idnt._rating[-1]
        return value

    def feat_meta_scan_order(self, idnt):
        return idnt.enum

    @functools.lru_cache(maxsize=32)
    def get_coords(self, which="px"):
        """Get the qmap coordinates for each curve in `QMap.ds`

        Parameters
        ----------
        which: str
            "px" for pixels or "um" for microns.
        """
        if which not in ["px", "um"]:
            raise ValueError("`which` must be 'px' or 'um'!")

        if which == "px":
            kx = "position x [px]"
            ky = "position y [px]"
        else:
            kx = "position x [µm]"
            ky = "position y [µm]"
        coords = []
        for idnt in self.ds:
            if kx in idnt.metadata and ky in idnt.metadata:
                cc = [idnt.metadata[kx], idnt.metadata[ky]]
            else:
                cc = [np.nan, np.nan]
            coords.append(cc)
        return coords

    def get_qmap(self, feature, qmap_only=False):
        """Return the quantitative map for a feature

        Parameters
        ----------
        feature: str
            Feature to compute map for (see `QMap.features`)
        qmap_only:
            Only return the quantitative map data,
            not the coordinates

        Returns
        -------
        x, y: 1d ndarray
            Only returned if `qmap_only` is False; Pixel grid
            coordinates along x and y
        qmap: 2d ndarray
            Quantitative map
        """
        coords = self.get_coords(which="px")

        map_data = []
        ffunc = self._feature_funcs[feature]
        for idnt in self.ds:
            val = ffunc(idnt)
            map_data.append(val)

        x, y, qmap = self._map_grid(coords=coords, map_data=map_data)
        if qmap_only:
            return qmap
        else:
            return x, y, qmap


unit_scales = {}
unit_scales["k"] = 1e3
unit_scales[""] = 1
unit_scales["m"] = 1e-3
unit_scales["µ"] = 1e-6
unit_scales["n"] = 1e-9
unit_scales["p"] = 1e-12
