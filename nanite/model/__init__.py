import warnings

from . import model_conical_indenter  # noqa: F401
from . import model_hertz_paraboloidal  # noqa: F401
from . import model_hertz_three_sided_pyramid  # noqa: F401
from . import model_sneddon_spherical  # noqa: F401
from . import model_sneddon_spherical_approximation  # noqa: F401

from .core import NaniteFitModel  # noqa: F401
from . import residuals  # noqa: F401
from .logic import models_available, register_model
from .logic import deregister_model, load_model_from_file  # noqa: F401


def compute_anc_parms(idnt, model_key):
    """Compute ancillary parameters for a force-distance dataset

    Ancillary parameters include parameters that:

    - are unrelated to fitting: They may just be important parameters
      to the user.
    - require the entire dataset: They cannot be extracted during
      fitting, because they require more than just the approach xor
      retract curve to compute (e.g. hysteresis, jump of retract curve
      at maximum indentation). They may, additionally, depend on
      initial fit parameters set by the user.
    - require a fit: They are dependent on fitting parameters but
      are not required during fitting.

    Notes
    -----
    If an ancillary parameter name matches that of a fitting parameter,
    then it is assumed that it can be used for fitting. Please see
    :func:`nanite.indent.Indentation.get_initial_fit_parameters`
    and :func:`nanite.fit.guess_initial_parameters`.

    Ancillary parameters are set to `np.nan` if they cannot be
    computed.

    Parameters
    ----------
    idnt: nanite.indent.Indentation
        The force-distance data for which to compute the ancillary
        parameters
    model_key: str
        Name of the model

    Returns
    -------
    ancillaries: collections.OrderedDict
        key-value dictionary of ancillary parameters
    """
    md = models_available[model_key]
    return md.compute_ancillaries(idnt)


def get_anc_parms(idnt, model_key):
    warnings.warn(
        "Method `get_anc_parms` is deprecated. Please use "
        + "`compute_anc_parms` instead.",
        DeprecationWarning)
    return compute_anc_parms(idnt, model_key)


def get_anc_parm_keys(model_key):
    """Return the key names of a model's ancillary parameters"""
    md = models_available[model_key]
    return md.get_anc_parm_keys()


def get_init_parms(model_key):
    """Get initial fit parameters for a model"""
    md = models_available[model_key]
    return md.get_parameter_defaults()


def get_model_by_name(name):
    """Convenience function to obtain a model by name instead of by key"""
    for key in models_available:
        if models_available[key].model_name == name:
            return models_available[key]
    else:
        raise KeyError("No model with name '{}'!".format(name))


def get_parm_name(model_key, parm_key):
    """Return parameter label

    Parameters
    ----------
    model_key: str
        The model key (e.g. "hertz_cone")
    parm_key: str
        The parameter key (e.g. "E")

    Returns
    -------
    parm_name: str
        The parameter label (e.g. "Young's Modulus")
    """
    md = models_available[model_key]
    return md.get_parm_name(parm_key)


def get_parm_unit(model_key, parm_key):
    """Return parameter unit

    Parameters
    ----------
    model_key: str
        The model key (e.g. "hertz_cone")
    parm_key: str
        The parameter key (e.g. "E")

    Returns
    -------
    parm_unit: str
        The parameter unit (e.g. "Pa")
    """
    md = models_available[model_key]
    return md.get_parm_unit(parm_key)


# Populate list of available fit models
_loc = locals().copy()
for _item in _loc:
    if _item.startswith("model_"):
        register_model(_loc[_item])
