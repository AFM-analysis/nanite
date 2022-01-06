import importlib
import pathlib
import sys
import warnings

from .core import ModelImportError, NaniteFitModel

#: currently available models
models_available = {}


def load_model_from_file(path, register=False):
    """Import a fit model file and return the module

    This is intended for loading custom models or for model
    development.

    Parameters
    ----------
    path: str or Path
        pathname to a Python script conaining a fit model
    register: bool
        whether to register the model after import

    Returns
    -------
    model: NaniteFitModel
        nanite fit model object

    Raises
    ------
    ModelImportError
        If the model cannot be imported
    """
    path = pathlib.Path(path)
    try:
        # insert the plugin directory to sys.path so we can import it
        sys.path.insert(-1, str(path.parent))
        sys.dont_write_bytecode = True
        module = importlib.import_module(path.stem)
    except ModuleNotFoundError:
        raise ModelImportError(f"Could not import '{path}'!")
    finally:
        # undo our path insertion
        sys.path.pop(0)
        sys.dont_write_bytecode = False

        mod = NaniteFitModel(module)

        if register:
            register_model(module)

        return mod


def register_model(module, *args):
    """Register a fitting model

    Parameters
    ----------
    module: Python module or NaniteFitModel
        the model to register

    Returns
    -------
    model: NaniteFitModel
        the corresponding NaniteFitModel instance
    """
    if args:
        warnings.warn("Please only pass the module to `register_model`!",
                      DeprecationWarning)
    global models_available  # this is not necessary, but clarifies things
    # add model
    if isinstance(module, NaniteFitModel):
        # we already have a fit model
        md = module
    else:
        md = NaniteFitModel(module)
    # the actual registration
    models_available[module.model_key] = md
    return md


def deregister_model(model):
    """Deregister a NaniteFitModel"""
    global models_available  # this is not necessary, but clarifies things
    models_available.pop(model.model_key)
