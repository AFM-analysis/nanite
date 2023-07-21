===============
Getting started
===============

Installation
============

To install nanite, use one of the following methods
(the package dependencies will be installed automatically):

* from `PyPI <https://pypi.python.org/pypi/nanite>`_:
    ``pip install nanite[CLI]``
* from `sources <https://github.com/AFM-Analysus/nanite>`_:
    ``pip install -e .[CLI]``

The appendix ``[CLI]`` makes sure that all dependencies for the
:ref:`command line interface <sec_cli>` are installed. If you are
only using nanite as a Python module, you may safely omit it.


What is nanite?
===============
The development of nanite was motivated by a unique problem that arises
in AFM force-distance data analysis, particularly for biological samples:
The data quality varies a lot due to biological variation and due to experimental
complexities that have to be dealt with when measuring biological samples.
To address this problem, nanite makes use of machine-learning (á la
`scikit-learn <http://scikit-learn.org/>`_), which allows to automatically
determine the quality of a force-distance curve based on a user-defined
rating scheme (see :ref:`sec_rating` for more information).
But nanite is much more than just that. It comes with an extensive set of
tools for AFM force-distance data analysis.


Supported file formats
======================
Nanite relies on the :ref:`afmformats <afmformats:index>` package.
A list of supported file formats can be found
:ref:`here <afmformats:supported_formats>`.


Use cases
=========
If you are a frequent AFM user, you might have run into several problems
involving data analysis, ranging from simple data fitting to the visualization
of quantitative force-distance maps. Here are a few usage examples
of nanite:

- You would like to automate your data analysis pipeline from loading
  force-distance data to displaying a fit to the approach part with
  a Hertz model for a spherical indenter. You can do so with nanite,
  either via scripting or via the command-line interface that comes
  with nanite. For more information, see :ref:`sec_fitting`.

- You would like to automatically analyze and visualize maps of
  force-distance data. This is possible with the
  :class:`nanite.QMap <nanite.qmap.QMap>` class.

- You would like to sort force-distance data according to data quality
  using your own training set (not the one shipped with nanite). Nanite
  allows you to create your own training set from your own experimental
  data, locally. Besides that, you can make use of multiple regressors
  and visualize the rating e.g. of force-distance maps. For
  an overview, see :ref:`sec_rating`.


Basic usage
===========
If you are not interested in scripting, please have a look at the
:ref:`fitting guide <sec_fitting>`.

In a Python script, you may use nanite as follows:

.. ipython::

    In [1]: import nanite

    In [2]: group = nanite.load_group("data/force-save-example.jpk-force")

    In [3]: idnt = group[0]  # This group actually as only one indentation curve.

    In [4]: idnt.apply_preprocessing(["compute_tip_position",
       ...:                           "correct_force_offset",
       ...:                           "correct_tip_offset"])

    In [5]: idnt.fit_model(model_key="sneddon_spher")

    In [6]: idnt.rate_quality()  # 0 means bad, 10 means good quality


You can find more examples in the :ref:`examples <sec_examples>` section.


How to cite
===========
If you use nanite in a scientific publication, please cite
Müller et al., *BMC Bioinformatics* (2019) :cite:`Mueller19nanite`.
