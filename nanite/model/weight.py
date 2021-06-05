import warnings

from .residuals import compute_contact_point_weights  # noqa: F401


def weight_cp(*args, **kwargs):
    warnings.warn(
        "The 'weight' module is deprecated. Please use "
        "'from nanite.model.residuals import compute_contact_point_weights'!",
        DeprecationWarning)
    return compute_contact_point_weights(*args, **kwargs)
