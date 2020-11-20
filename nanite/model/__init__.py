from collections import OrderedDict
import inspect
import warnings

import numpy as np

from . import model_conical_indenter  # noqa: F401
from . import model_hertz_paraboloidal  # noqa: F401
from . import model_hertz_three_sided_pyramid  # noqa: F401
from . import model_sneddon_spherical  # noqa: F401
from . import model_sneddon_spherical_approximation  # noqa: F401

from . import residuals


#: Common ancillary parameters
ANCILLARY_COMMON = OrderedDict()
ANCILLARY_COMMON["max_indent"] = ("Maximum indentation", "m")


class ModelIncompleteError(BaseException):
    pass


class ModelImplementationError(BaseException):
    pass


class ModelImplementationWarning(UserWarning):
    pass


def get_anc_parms(idnt, model_key):
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
    # TODO:
    # - ancillaries are not cached yet (some ancillaries might depend on
    #   fitting interval or other initial parameters - take that into account)
    # - "max_indent" actually belongs to "common_ancillaries" (see fit.py)
    anc_ord = OrderedDict()
    # compute maximal indentation
    if ("tip position" in idnt
        and "params_fitted" in idnt.fit_properties
            and "contact_point" in idnt.fit_properties["params_fitted"]):
        cp = idnt.fit_properties["params_fitted"]["contact_point"].value
        idmax = idnt.data.appr["fit"].argmax()
        mi = idnt.data.appr["tip position"][idmax]
        mival = (cp-mi)
    else:
        mival = np.nan
    anc_ord["max_indent"] = mival
    # Model ancillaries
    md = models_available[model_key]
    if hasattr(md, "compute_ancillaries"):
        anc_par = md.compute_ancillaries(idnt)
        for kk in md.parameter_anc_keys:
            anc_ord[kk] = anc_par[kk]
    return anc_ord


def get_anc_parm_keys(model_key):
    """Return the key names of a model's ancillary parameters"""
    akeys = list(ANCILLARY_COMMON.keys())
    md = models_available[model_key]
    if hasattr(md, "parameter_anc_keys"):
        akeys += md.parameter_anc_keys
    return akeys


def get_init_parms(model_key):
    """Get initial fit parameters for a model"""
    md = models_available[model_key]
    parms = md.get_parameter_defaults()
    return parms


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
    if parm_key in md.parameter_keys:
        idx = md.parameter_keys.index(parm_key)
        return md.parameter_names[idx]
    elif (hasattr(md, "compute_ancillaries")
          and parm_key in md.parameter_anc_keys):
        idx = md.parameter_anc_keys.index(parm_key)
        return md.parameter_anc_names[idx]
    elif parm_key in ANCILLARY_COMMON:
        return ANCILLARY_COMMON[parm_key][0]


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
    if parm_key in md.parameter_keys:
        idx = md.parameter_keys.index(parm_key)
        return md.parameter_units[idx]
    elif (hasattr(md, "compute_ancillaries")
          and parm_key in md.parameter_anc_keys):
        idx = md.parameter_anc_keys.index(parm_key)
        return md.parameter_anc_units[idx]
    elif parm_key in ANCILLARY_COMMON:
        return ANCILLARY_COMMON[parm_key][1]


def register_model(module, module_name):
    """Register a fitting model"""
    global models_available  # this is not necessary, but clarifies things
    # sanity checks
    missing = []
    for attr in ["get_parameter_defaults",
                 "model_doc",
                 "model_key",
                 "model_name",
                 "parameter_keys",
                 "parameter_names",
                 "parameter_units",
                 "valid_axes_x",
                 "valid_axes_y",
                 ]:
        if not hasattr(module, attr):
            missing.append(attr)
    if missing:
        raise ModelIncompleteError(
            "Model `{}` is missing the following ".format(module_name)
            + "attributes: {}".format(", ".join(missing)))

    # check for completeness of ancillary parameter recipe
    if hasattr(module, "compute_ancillaries"):
        missing_anc = []
        for attr in ["parameter_anc_keys",
                     "parameter_anc_names",
                     "parameter_anc_units",
                     ]:
            if not hasattr(module, attr):
                missing_anc.append(attr)
        if missing_anc:
            raise ModelIncompleteError(
                "Model `{}` is missing the following ".format(module_name)
                + "attributes: {}".format(", ".join(missing_anc)))

    # check length of modeling lists
    if len(module.parameter_keys) != len(module.parameter_names):
        raise ModelImplementationError(
            "'parameter_keys' and 'parameter_names' have different lengths "
            + "for model '{}'!".format(module.model_key))
    if len(module.parameter_keys) != len(module.parameter_units):
        raise ModelImplementationError(
            "'parameter_keys' and 'parameter_units' have different lengths "
            + "for model '{}'!".format(module.model_key))

    # checks for model parameters
    p_def = list(module.get_parameter_defaults().keys())
    p_arg = list(inspect.signature(module.model_func).parameters.keys())
    for ii, key in enumerate(module.parameter_keys):
        if key != p_def[ii]:
            raise ModelImplementationError(
                "Please check 'parameter_keys' and 'get_parameter_defaults' "
                + "of the model '{}'.".format(module.model_key)
                + "Keys {} and {} are not in order!".format(key, p_def[ii]))
        if key != p_arg[ii+1]:
            warnings.warn(
                "Please make sure that the parameters of the model "
                + "function are in the same order as in 'parameter_keys' "
                + "for the model '{}'! ".format(module.model_key)
                + "The abscissa (usually `delta`) should come first. "
                + "This warning may become an Exception in the future!",
                ModelImplementationWarning)

    # check for residuals function
    if not hasattr(module, "residual"):
        # use the default residual function
        module.residual = residuals.get_default_residuals_wrapper(
            model_function=module.model_func)

    # check for modeling function
    if not hasattr(module, "model"):
        # use the default residual function
        module.model = residuals.get_default_modeling_wrapper(
            model_function=module.model_func)

    # add model
    models_available[module.model_key] = module


models_available = {}

# Populate list of available fit models
_loc = locals().copy()
for _item in _loc:
    if _item.startswith("model_"):
        register_model(_loc[_item], _item)
