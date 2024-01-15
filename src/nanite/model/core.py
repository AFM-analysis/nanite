from collections import OrderedDict
import inspect
import warnings

import numpy as np

from . import residuals


class ModelError(BaseException):
    pass


class ModelIncompleteError(ModelError):
    pass


class ModelImplementationError(ModelError):
    pass


class ModelImportError(ModelError):
    pass


class ModelImplementationWarning(UserWarning):
    pass


class NaniteFitModel:
    def __init__(self, model_module):
        """Initialize the model with an imported Python module"""
        self.module = model_module
        self._module_check()
        self._module_autocomplete()

        # propagate all module attributes to this instance
        # standard parameters
        self.get_parameter_defaults = self.module.get_parameter_defaults
        self.model_doc = self.module.model_doc
        self.model_key = self.module.model_key
        self.model_name = self.module.model_name
        self.parameter_keys = self.module.parameter_keys
        self.parameter_names = self.module.parameter_names
        self.parameter_units = self.module.parameter_units
        self.valid_axes_x = self.module.valid_axes_x
        self.valid_axes_y = self.module.valid_axes_y
        # optional ancillary parameters
        if hasattr(self.module, "compute_ancillaries"):
            self.has_module_ancillaries = True
            self.parameter_anc_keys = self.module.parameter_anc_keys
            self.parameter_anc_names = self.module.parameter_anc_names
            self.parameter_anc_units = self.module.parameter_anc_units
        else:
            self.has_module_ancillaries = False
        # model function
        self.model = self.module.model
        # residuals
        self.residual = self.module.residual

    def __str__(self):
        return f"NaniteFitModel '{self.model_key}'"

    def __repr__(self):
        return f"<NaniteFitModel '{self.model_key}' at {hex(id(self))}"

    def _module_autocomplete(self):
        """Add any missing attributes to the underlying model module"""
        # check for residuals function
        if not hasattr(self.module, "residual"):
            # use the default residual function
            self.module.residual = residuals.get_default_residuals_wrapper(
                model_function=self.module.model_func)

        # check for modeling function
        if not hasattr(self.module, "model"):
            # use the default residual function
            self.module.model = residuals.get_default_modeling_wrapper(
                model_function=self.module.model_func)

    def _module_check(self):
        """Checks whether the model's module is set up correctly"""
        # sanity checks
        missing = []
        for attr in [
            "get_parameter_defaults",
            "model_doc",
            "model_key",
            "model_name",
            "parameter_keys",
            "parameter_names",
            "parameter_units",
            "valid_axes_x",
            "valid_axes_y",
             ]:
            if not hasattr(self.module, attr):
                missing.append(attr)
        if missing:
            raise ModelIncompleteError(
                f"Model `{self.module}` is missing the following "
                + f"attributes: {', '.join(missing)}")

        model_key = self.module.model_key
        # check for completeness of ancillary parameter recipe
        if hasattr(self.module, "compute_ancillaries"):
            missing_anc = []
            for attr in ["parameter_anc_keys",
                         "parameter_anc_names",
                         "parameter_anc_units",
                         ]:
                if not hasattr(self.module, attr):
                    missing_anc.append(attr)
            if missing_anc:
                raise ModelIncompleteError(
                    f"Model `{model_key}` is missing the following "
                    + f"attributes: {', '.join(missing_anc)}")

        # check length of modeling lists
        if len(self.module.parameter_keys) != len(self.module.parameter_names):
            raise ModelImplementationError(
                "'parameter_keys' and 'parameter_names' have different "
                + f"lengths for model '{model_key}'!")
        if len(self.module.parameter_keys) != len(self.module.parameter_units):
            raise ModelImplementationError(
                "'parameter_keys' and 'parameter_units' have different "
                + f"lengths for model '{model_key}'!")

        # check for spaces in units
        if [u.strip() for u in self.module.parameter_units] \
                != self.module.parameter_units:
            warnings.warn("The `parameter_units` should not contain leading "
                          + f"or trailing spaces. Please check {model_key}!",
                          ModelImplementationWarning)

        if hasattr(self.module, "parameter_anc_units"):
            if [u.strip() for u in self.module.parameter_anc_units] \
                    != self.module.parameter_anc_units:
                warnings.warn(
                    "The `parameter_anc_units` should not contain leading "
                    + f"or trailing spaces. Please check {model_key}!",
                    ModelImplementationWarning)

        # check for label uniqueness
        if len(self.module.parameter_names) \
                != len(set(self.module.parameter_names)):
            raise ModelImplementationError(
                f"'parameter_names' should be unique for '{model_key}'!")

        # checks for model parameters
        p_def = list(self.module.get_parameter_defaults().keys())
        p_arg = list(inspect.signature(
            self.module.model_func).parameters.keys())
        for ii, key in enumerate(self.module.parameter_keys):
            if key != p_def[ii]:
                raise ModelImplementationError(
                    "Please check 'parameter_keys' and "
                    + f"'get_parameter_defaults'  of the model '{model_key}'. "
                    + f"Keys {key} and {p_def[ii]} are not in order!")
            if key != p_arg[ii+1]:
                warnings.warn(
                    "Please make sure that the parameters of the model "
                    + "function are in the same order as in 'parameter_keys' "
                    + f"for the model '{model_key}'! "
                    + "The abscissa (usually `delta`) should come first. "
                    + "This warning may become an Exception in the future!",
                    ModelImplementationWarning)

    def compute_ancillaries(self, fd):
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
        fd: nanite.indent.Indentation
            The force-distance data for which to compute the ancillary
            parameters

        Returns
        -------
        ancillaries: collections.OrderedDict
            key-value dictionary of ancillary parameters
        """
        # TODO:
        # - ancillaries are not cached yet (some ancillaries might depend on
        #   fitting interval or other initial parameters - take that into
        #   account)
        # - "max_indent" actually belongs to "common_ancillaries" (see fit.py)
        anc_ord = OrderedDict()
        # general
        for key in ANCILLARY_COMMON:
            gmeth = ANCILLARY_COMMON[key][2]
            anc_ord[key] = gmeth(fd)
        # from module
        if self.has_module_ancillaries:
            anc_md = self.module.compute_ancillaries(fd)
            for kk in self.parameter_anc_keys:
                anc_ord[kk] = anc_md[kk]
        return anc_ord

    def get_anc_parm_keys(self):
        """Return the key names of a model's ancillary parameters"""
        akeys = list(ANCILLARY_COMMON.keys())
        if self.has_module_ancillaries:
            akeys += self.parameter_anc_keys
        return akeys

    def get_parm_name(self, key):
        """Return parameter label

        Parameters
        ----------
        key: str
            The parameter key (e.g. "E")

        Returns
        -------
        parm_name: str
            The parameter label (e.g. "Young's Modulus")
        """
        if key in self.parameter_keys:
            idx = self.parameter_keys.index(key)
            return self.parameter_names[idx]
        elif (self.has_module_ancillaries
              and key in self.parameter_anc_keys):
            idx = self.parameter_anc_keys.index(key)
            return self.parameter_anc_names[idx]
        elif key in ANCILLARY_COMMON:
            return ANCILLARY_COMMON[key][0]
        else:
            raise KeyError(
                f"Could not find parameter name for '{key}' in '{self}'!")

    def get_parm_unit(self, key):
        """Return parameter unit

        Parameters
        ----------
        key: str
            The parameter key (e.g. "E")

        Returns
        -------
        parm_unit: str
            The parameter unit (e.g. "Pa")
        """
        if key in self.parameter_keys:
            idx = self.parameter_keys.index(key)
            return self.parameter_units[idx]
        elif (self.has_module_ancillaries
              and key in self.parameter_anc_keys):
            idx = self.parameter_anc_keys.index(key)
            return self.parameter_anc_units[idx]
        elif key in ANCILLARY_COMMON:
            return ANCILLARY_COMMON[key][1]
        else:
            raise KeyError(
                f"Could not find parameter unit for '{key}' in '{self}'!")


def compute_anc_max_indent(fd):
    """Compute ancillary parameter 'Maximum indentation'"""
    # compute maximal indentation
    if ("tip position" in fd
        and "fit" in fd
            and "params_fitted" in fd.fit_properties
            and "contact_point" in fd.fit_properties["params_fitted"]):
        cp = fd.fit_properties["params_fitted"]["contact_point"].value
        idmax = fd.appr["fit"].argmax()
        mi = fd.appr["tip position"][idmax]
        mival = (cp-mi)
    else:
        mival = np.nan
    return mival


#: Common ancillary parameters
ANCILLARY_COMMON = OrderedDict()
ANCILLARY_COMMON["max_indent"] = ("Maximum indentation",
                                  "m",
                                  compute_anc_max_indent)
