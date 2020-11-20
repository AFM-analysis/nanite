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
First, create a Python file ``model_user.py`` which will be the home of your
new model (make sure the name starts with ``model``). Place the file in the
following location: ``nanite/model/model_user.py``. You file should at least
contain the following:

.. code:: python

    import lmfit
    import numpy as np

    from . import weight


    def get_parameter_defaults():
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

Once you have created this file, you have to register it in nanite by
adding the line

.. code:: python

    from . import model_user  # noqa: F401

at the top in the file ``nanite/model/__init__.py``.

A few things should be noted:

- When designing your model parameters, always use SI units.
- Always include a model formula. You can test whether it renders
  correctly by building the documentation (see above) and checking
  whether your model shows up properly in the code reference.
