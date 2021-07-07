import warnings

import afmformats
from afmformats.afm_qmap import qmap_feature
import numpy as np

from .read import get_load_data_modality_kwargs


class DataMissingWarning(UserWarning):
    pass


class QMap(afmformats.AFMQMap):
    def __init__(self, path_or_group, meta_override=None, callback=None):
        """Quantitative force spectroscopy map handling

        Parameters
        ----------
        path_or_group: str or pathlib.Path or afmformats.afm_group.AFMGroup
            The path to the data file or an instance of `AFMGroup`
        meta_override: dict
            Dictionary with metadata that is used when loading the data
            in `path`.
        callback: callable or None
            A method that accepts a float between 0 and 1
            to externally track the process of loading the data.
        """
        super(QMap, self).__init__(
            path_or_group=path_or_group,
            meta_override=meta_override,
            callback=callback,
            **get_load_data_modality_kwargs()
        )

    @staticmethod
    @qmap_feature(name="fit: contact point",
                  unit="nm",
                  cache=False)
    def feat_fit_contact_point(idnt):
        """Contact point of the fit"""
        if idnt.fit_properties.get("success", False):
            # use cached contact point
            params = idnt.fit_properties["params_fitted"]
            value = params["contact_point"].value * 1e9
        else:
            msg = "The experimental data has not been fitted. Please call " \
                  + "`idnt.fit_model` manually for {}!".format(idnt)
            warnings.warn(msg, DataMissingWarning)
            value = np.nan

        return value

    @staticmethod
    @qmap_feature(name="fit: Young's modulus",
                  unit="Pa",
                  cache=False)
    def feat_fit_youngs_modulus(idnt):
        """Young's modulus"""
        if idnt.fit_properties.get("success", False):
            # use cached young's modulus
            value = idnt.fit_properties["params_fitted"]["E"].value
        else:
            msg = "The experimental data has not been fitted. Please call " \
                  + "`idnt.fit_model` manually for {}!".format(idnt)
            warnings.warn(msg, DataMissingWarning)
            value = np.nan

        return value

    @staticmethod
    @qmap_feature(name="fit: rating",
                  unit="",
                  cache=False)
    def feat_meta_rating(idnt):
        """Rating"""
        if idnt._rating is None:
            msg = "The experimental data has not been rated. Please call " \
                  + "`idnt.rate_quality` manually for {}!".format(idnt)
            warnings.warn(msg, DataMissingWarning)
            value = np.nan
        else:
            # use cached rating
            value = idnt._rating[-1]
        return value
