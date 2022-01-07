.. _sec_dev:

===============
Developer guide
===============

.. _sec_dev_contribute:

How to contribute
=================
Contributions via pull requests are very welcome. Just fork the "master"
branch, make your changes, and create a pull request back to "master"
with a descriptive title and an explanation of what you have done.
If you decide to contribute code, please

1. properly document your code (in-line comments as well as doc strings),
2. ensure code quality with
   `flake8 <https://pypi.org/project/flake8/>`_ and
   `autopep8 <https://pypi.org/project/autopep8/>`_,
3. write test functions for `pytest <https://pytest.org>`_ (aim for 100% code
   `coverage <https://pypi.org/project/coverage/>`_),
4. update the changelog (for new features, increment to the next minor
   release; for small changes or bug fixes, increment the patch number)



.. _sec_dev_docs:

Updating the documentation
==========================

The documentation is stored in the ``docs`` directory of the repository
and is built using sphinx.

To build the documentation, first install the build requirements by running
this in the ``docs`` directory:

.. code:: bash

    pip install -r requirements.txt``

You can now build the documentation with

.. code:: bash

    sphinx-build . _build

Open the file ``_build/index.html`` in your web browser to view the
result.


.. _sec_dev_model:

Writing model functions
=======================
You are here because you would like to write a new model function for nanite.
Note that all model functions implemented in nanite are consequently available
in PyJibe as well.

Getting started
---------------
First, create a Python file ``model_unique_name.py`` which will be the home of your
new model (make sure the name starts with ``model_``).
You have three options (**1, 2 or 3**) to make your model available in nanite:

1. Place the file anywhere in your file system (e.g.
   ``/home/peter/model_unique_name.py``) and run:

   .. code:: python

       from nanite.model import load_model_from_file

       load_model_from_file("/home/peter/model_unique_name.py", register=True)

   This is probably the most convenient method when prototyping. Note that
   you can also import model scritps in PyJibe (via the Preferences menu).

2. Place the file in the following location: ``nanite/model/model_unique_name.py``.
   Once you have created this file, you have to register it in nanite by
   adding the line

   .. code:: python

       from . import model_unique_name  # noqa: F401

   at the top in the file ``nanite/model/__init__.py``. This is the procedure
   when you create a pull request.

3. Or place the file in another location from where you can import it. This can
   be a submodule in a different package, or just the script in your ``PATH``.
   The only thing you need is to import the script and register it.

   .. code:: python

       import model_unique_name
       from nanite.model import register_model

       register_model(model_unique_name)


Your file should at least contain the following:

.. code:: python

    import lmfit
    import numpy as np


    def get_parameter_defaults():
        """Return the default model parameters"""
        # The order of the parameters must match the order
        # of ´parameter_names´ and ´parameter_keys´.
        params = lmfit.Parameters()
        params.add("E", value=3e3, min=0)
        params.add("contact_point", value=0)
        params.add("baseline", value=0)
        return params


    def your_model_name(delta, E, contact_point=0, baseline=0):
        r"""A brief model description

        A more elaborate model description with a formula.

        .. math::

            F = \frac{4}{3}
                E
                \delta^{3/2}

        Parameters
        ----------
        delta: 1d ndarray
            Indentation [m]
        E: float
            Young's modulus [N/m²]
        contact_point: float
            Indentation offset [m]
        baseline: float
            Force offset [N]

        Returns
        -------
        F: float
            Force [N]

        Notes
        -----
        Here you can add more information about the model.

        References
        ----------
        Please give proper references for your model (e.g. publications or
        arXiv manuscripts. You can do so by editing the "docs/nanite.bib"
        file and cite it like so:
        Sneddon (1965) :cite:`Sneddon1965`
        """
        # this is a convention to avoid computing the root of negative values
        root = contact_point - delta
        pos = root > 0
        # this is the model output
        out = np.zeros_like(delta)
        out[pos] = 4/3 * E * root[pos]**(3/2)
        # add the baseline
        return out + baseline


    model_doc = your_model_name.__doc__
    model_func = your_model_name
    model_key = "unique_model_key"
    model_name = "short model name"
    parameter_keys = ["E", "contact_point", "baseline"]
    parameter_names = ["Young's Modulus", "Contact Point", "Force Baseline"]
    parameter_units = ["Pa", "m", "N"]
    valid_axes_x = ["tip position"]
    valid_axes_y = ["force"]

A few things should be noted:

- When designing your model parameters, always use SI units.
- Always include a model formula in the doc string. You can test whether it
  renders correctly by building the documentation (see above) and checking
  whether your model shows up properly in the code reference.
- Fitting parameters should not contain spaces. Only use characters that
  are allowed in Python variable names.
- Since fitting is based on `lmfit <https://pypi.org/project/lmfit/>`_, you may define
  `mathematical constraints <https://lmfit.github.io/lmfit-py/constraints.html>`_
  in ``get_parameter_defaults``. This includes
  `algebraic constraints <https://lmfit.github.io/lmfit-py/constraints.html#using-inequality-constraints>`_.
  However, if possible, try to solve your particular problem with ancillaries
  (see below), a concept that is easier to debug.
- If you would like to define "helper" parameters that should be hidden from
  users in PyJibe, you can prepend an underscore (`_`) to the parameter
  name.
- By default, nanite uses the method
  :func:`nanite.model.residuals.residual` to compute fit residuals. This
  method also implements the "reduce residuals near contact point" feature.
  You may define your own ``residual`` function in your model file, but this
  is discouraged. The same is true for the ``model`` function, which defaults
  to :func:`nanite.model.residuals.model_direction_agnostic`.
- You should always name the contact point parameter ``contact_point``.
  Otherwise fitting will not work. If the :ref:`geometrical correction factor
  <sec_fitting_gcfk>` :math:`k` is used, the ``contact_point`` parameter is modified
  internally before and after the fit. If you don't use ``contact_point``,
  then your fit results will be wrong when using :math:`k \ne 1`.
- You should always name the parameter describing the Young's modulus ``E``.
  This is important for higher-level functionalities in e.g. PyJibe and for
  plotting the Young's modulus over the indentation depth.


Now it is time for a quick sanity check:

.. code:: python

    from nanite import model
    assert "unique_model_key" in model.models_available


Ancillary parameters
--------------------
For more elaborate models, you might need additional parameters from the
:class:`nanite.indent.Indentation` instance. This is where ancillary
parameters come into play.

You can define an arbitrary number of ancillary parameters in your
``model_unique_name.py`` file:

.. code:: python

    def compute_ancillaries(idnt):
        """Compute ancillaries for my model

        Parameters
        ----------
        idnt: nanite.indent.Indentation
            Indentation dataset from which to extract the ancillary
            parameters.

        Returns
        -------
        example: dict
            Dictionary with ancillary parameters. In this example:

            - "force_range": total force range covered by approach and retract
        """
        # You have access to the initial fit parameters (including a
        # good contact point estimate) with this line:
        parms = idnt.get_initial_fit_parameters(model_key=model_key,
                                                model_ancillaries=False)

        # You can access individual columns...
        force = idnt.data["force"]
        segment = idnt.data["segment"]  # `False` for approach; `True` for retract
        tip_position = idnt.data["tip position"]

        # ...and segments
        force_approach = force[~segment]  # equivalent to force[segment == False]
        force_retract = force[segment]

        # Initialize ancillary dictionary.
        anc_dict = dict()

        # This is the exemplary force parameter
        anc_dict["force_range"] = np.ptp(force)

        return anc_dict

    # And below the other `parameter_keys` etc.:
    parameter_anc_keys = ["force_range"]
    parameter_anc_names = ["Overall peak-to-peak force"]
    parameter_anc_units = ["N"]


You should know:

- If an ancillary parameter key matches that of a fitting parameter
  (defined in ``get_parameter_defaults`` above), then the ancillary
  parameter can be used as an initial value for fitting (see
  :func:`nanite.fit.guess_initial_parameters`).
- If ``compute_ancillaries`` does not know how to compute a certain
  parameter, it shoud set it to ``np.nan`` instead of ``None``
  (compatibility with PyJibe).
- If you would like to define an ancillary parameter that depends on
  a successful fit, you could first check against ``idnt.fit_properties["success"]``
  and then compute your parameter (else set it to ``np.nan``).
