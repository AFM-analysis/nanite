import warnings

import afmformats
import numpy as np

from .indent import Indentation


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
            data_classes_by_modality={"force-distance": Indentation}
        )

    @staticmethod
    def feat_fit_contact_point(idnt):
        """fit: contact point [m]"""
        if idnt.fit_properties and idnt.fit_properties["success"]:
            # use cached rating
            value = idnt.fit_properties["params_fitted"]["contact_point"].value
        else:
            msg = "The experimental data has not been fitted. Please call " \
                  + "`idnt.fit_model` manually for {}!".format(idnt)
            warnings.warn(msg, DataMissingWarning)
            value = np.nan

        return value

    @staticmethod
    def feat_fit_youngs_modulus(idnt):
        """fit: Young's modulus [Pa]"""
        if idnt.fit_properties and idnt.fit_properties["success"]:
            # use cached rating
            value = idnt.fit_properties["params_fitted"]["E"].value
        else:
            msg = "The experimental data has not been fitted. Please call " \
                  + "`idnt.fit_model` manually for {}!".format(idnt)
            warnings.warn(msg, DataMissingWarning)
            value = np.nan

        return value

    @staticmethod
    def feat_meta_rating(idnt):
        """fit: rating []"""
        if idnt._rating is None:
            msg = "The experimental data has not been rated. Please call " \
                  + "`idnt.rate_quality` manually for {}!".format(idnt)
            warnings.warn(msg, DataMissingWarning)
            value = np.nan
        else:
            # use cached rating
            value = idnt._rating[-1]
        return value
