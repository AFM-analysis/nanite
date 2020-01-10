import functools
import warnings

import numpy as np

from .group import IndentationGroup


class DataMissingWarning(UserWarning):
    pass


class QMap(object):
    def __init__(self, path_or_dataset, callback=None):
        """Quantitative force spectroscopy map handling

        Parameters
        ----------
        path_or_dataset: str or nanite.IndentationGroup
            The path to the data file. The data format is determined
            using the extension of the file and the data is loaded
            with the correct method.
        callback: callable or None
            A method that accepts a float between 0 and 1
            to externally track the process of loading the data.
        """
        if isinstance(path_or_dataset, IndentationGroup):
            group = path_or_dataset
        else:
            group = IndentationGroup(path=path_or_dataset, callback=callback)
        #: Indentation data (instance of :class:`nanite.IndentationGroup`)
        self.group = group

        # Register feature functions
        self._feature_funcs = {}
        for key in feature_mapping:
            self._feature_funcs[key] = getattr(self,
                                               feature_mapping[key].__name__)

        #: Available features (see :data:`nanite.qmap.available_features`)
        self.features = available_features

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
        idnt0 = self.group[0]
        # get extent of the map
        sx = idnt0.metadata["grid size x"] * 1e6
        sy = idnt0.metadata["grid size y"] * 1e6
        cx = idnt0.metadata["grid center x"] * 1e6
        cy = idnt0.metadata["grid center y"] * 1e6
        extent = (cx - sx/2, cx + sx/2,
                  cy - sy/2, cy + sy/2,
                  )
        return extent

    @property
    @functools.lru_cache(maxsize=32)
    def shape(self):
        """shape of the map [px]"""
        idnt0 = self.group[0]
        # get shape of the map
        shape = (idnt0.metadata["grid shape x"],
                 idnt0.metadata["grid shape y"]
                 )
        return shape

    def feat_data_min_height_measured_um(self, idnt):
        height = np.min(idnt["height (measured)"])
        value = height / unit_scales["µ"]
        return value

    def feat_fit_contact_point(self, idnt):
        if (idnt.fit_properties and idnt.fit_properties["success"]):
            # use cached rating
            value = idnt.fit_properties["params_fitted"]["contact_point"].value
        else:
            msg = "The experimental data has not been fitted. Please call " \
                  + "`idnt.fit_model` manually for {}!".format(idnt)
            warnings.warn(msg, DataMissingWarning)
            value = np.nan

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
            kx = "grid index x"
            ky = "grid index y"
            mult = 1
        else:
            kx = "position x"
            ky = "position y"
            mult = 1e6
        coords = []
        for idnt in self.group:
            # We assume that kx and ky are given. This has to be
            # ensured by the file format reader for qmaps.
            cc = [idnt.metadata[kx] * mult, idnt.metadata[ky] * mult]
            coords.append(cc)
        return np.array(coords)

    def get_qmap(self, feature, qmap_only=False):
        """Return the quantitative map for a feature

        Parameters
        ----------
        feature: str
            Feature to compute map for (see :data:`QMap.features`)
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
        for idnt in self.group:
            val = ffunc(idnt)
            map_data.append(val)

        x, y, qmap = self._map_grid(coords=coords, map_data=map_data)
        if qmap_only:
            return qmap
        else:
            return x, y, qmap


# Maps feature names to functions in QMap
feature_mapping = {
    "data min height": QMap.feat_data_min_height_measured_um,
    "fit contact point": QMap.feat_fit_contact_point,
    "fit young's modulus": QMap.feat_fit_youngs_modulus,
    "meta rating": QMap.feat_meta_rating,
    "meta scan order": QMap.feat_meta_scan_order,
}

#: Available features for quantitative maps
available_features = sorted(feature_mapping.keys())


unit_scales = {}
unit_scales["k"] = 1e3
unit_scales[""] = 1
unit_scales["m"] = 1e-3
unit_scales["µ"] = 1e-6
unit_scales["n"] = 1e-9
unit_scales["p"] = 1e-12
