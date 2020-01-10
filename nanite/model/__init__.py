from . import model_conical_indenter  # noqa: F401
from . import model_hertz_paraboloidal  # noqa: F401
from . import model_hertz_three_sided_pyramid  # noqa: F401
from . import model_sneddon_spherical  # noqa: F401
from . import model_sneddon_spherical_approximation  # noqa: F401


class ModelIncompleteError(BaseException):
    pass


def get_anc_parms(idnt, model_key):
    """Compute ancillary parameters for a force-distance dataset

    Ancillary parameters include parameters that:

    - are unrelated to fitting: They may just be important parameters
      to the user.
    - require the entire dataset: They cannot be extracted during
      fitting, because they require more than just the approach xor
      retract curve to compute (e.g. hysteresis, jump of retract curve
      at maximum indentation).
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
    ancillaries: dict
        key-value dictionary of ancillary parameters
    """
    # TODO: ancillaries are not cached yet
    md = models_available[model_key]
    if hasattr(md, "compute_ancillaries"):
        return md.compute_ancillaries(idnt)
    else:
        return {}


def get_anc_parm_keys(model_key):
    """Return the key names of a model's ancillary parameters"""
    md = models_available[model_key]
    if hasattr(md, "parameter_anc_keys"):
        return md.parameter_anc_keys
    else:
        return []


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
    """Get human readable parameter label

    Parameters
    ----------
    model_key: str
        The model key (e.g. "hertz_cone")
    parm_key: str
        The parameter key (e.g. "E")

    Returns
    -------
    parm_name: str
        The parameter name (e.g. "Young's Modulus")
    """
    md = models_available[model_key]
    idx = md.parameter_keys.index(parm_key)
    return md.parameter_names[idx]


def register_model(module, module_name):
    """Register a fitting model"""
    global models_available  # this is not necessary, but clarifies things
    # sanity checks
    missing = []
    for attr in ["get_parameter_defaults",
                 "model",
                 "residual",
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
                     ]:
            if not hasattr(module, attr):
                missing_anc.append(attr)
        if missing_anc:
            raise ModelIncompleteError(
                "Model `{}` is missing the following ".format(module_name)
                + "attributes: {}".format(", ".join(missing_anc)))
    # add model
    models_available[module.model_key] = module


models_available = {}

# Populate list of available fit models
_loc = locals().copy()
for _item in _loc:
    if _item.startswith("model_"):
        register_model(_loc[_item], _item)
