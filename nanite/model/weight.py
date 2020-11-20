import warnings

from .residuals import compute_contact_point_weights as weight_cp  # noqa: F401

warnings.warn(
    "The 'weight' module is deprecated. Please use "
    "'from nanite.model.residuals import compute_contact_point_weights'!",
    DeprecationWarning)
