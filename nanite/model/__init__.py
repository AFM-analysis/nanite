from . import model_conical_indenter  # noqa: F401
from . import model_hertz_paraboloidal  # noqa: F401
from . import model_hertz_three_sided_pyramid  # noqa: F401
from . import model_sneddon_spherical  # noqa: F401
from . import model_sneddon_spherical_approximation  # noqa: F401


def get_model_by_name(name):
    """Convenience function to obtain a model by name instead of by key"""
    for key in models_available:
        if models_available[key].model_name == name:
            return models_available[key]
    else:
        raise KeyError("No model with name {}".format(name))


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
    model = models_available[model_key]
    idx = model.parameter_keys.index(parm_key)
    return model.parameter_names[idx]


def get_init_parms(model_key):
    """Get initial fit parameters for a model"""
    model = models_available[model_key]
    parms = model.get_parameter_defaults()
    return parms


models_available = {}

# Populate list of available fit models
loc = locals().copy()
for item in loc:
    if item.startswith("model_"):
        models_available[loc[item].model_key] = loc[item]
