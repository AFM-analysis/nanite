==============
Code reference
==============

.. toctree::
  :maxdepth: 2



.. _sec_ref_alias:

Module level aliases
====================
For user convenience, the following objects are available
at the module level.

.. class:: nanite.Indentation
    
    alias of :class:`nanite.indent.Indentation`

.. class:: nanite.IndentationGroup
    
    alias of :class:`nanite.group.IndentationGroup`

.. class:: nanite.IndentationRater
    
    alias of :class:`nanite.rate.IndentationRater`

.. class:: nanite.QMap
    
    alias of :class:`nanite.qmap.QMap`

.. function:: nanite.load_group

    alias of :class:`nanite.group.load_group`


.. _sec_ref_indent:

Force-indentation data
======================

.. automodule:: nanite.indent
   :members:
   :undoc-members:


.. _sec_ref_group:

Groups
======

.. automodule:: nanite.group
   :members:
   :undoc-members:


.. _sec_ref_read:

Loading data
============

.. automodule:: nanite.read
   :members:
   :undoc-members:


.. _sec_ref_preproc:

Preprocessing
=============

.. automodule:: nanite.preproc
   :members:
   :undoc-members:


.. _sec_ref_model:

Modeling
========

Methods and constants
---------------------

.. automodule:: nanite.model
   :members:
   :undoc-members:


Models
------
Each model is implemented as a submodule in `nanite.model`. For instance
:mod:`nanite.model.model_hertz_parabolic`. Each of these modules implements
the following functions (which are not listed for each model in the
subsections below):



.. function:: nanite.model.model_submodule.get_parameter_defaults

    Return the default parameters of the model.

.. function:: nanite.model.model_submodule.model

    Wrap the actual model for fitting.

.. function:: nanite.model.model_submodule.residual

    Compute the residuals during fitting.


In addition, each submodule contains the following attributes:

.. attribute:: nanite.model.model_submodule.model_doc

    The doc-string of the model function.

.. attribute:: nanite.model.model_submodule.model_key

    The model key used in the command line interface and during scripting.

.. attribute:: nanite.model.model_submodule.model_name

    The name of the model.

.. attribute:: nanite.model.model_submodule.parameter_keys

    Parameter keywords of the model for higher-level applications.

.. attribute:: nanite.model.model_submodule.parameter_names

    Parameter names of the model for higher-level applications.

.. attribute:: nanite.model.model_submodule.parameter_names

    Parameter units for higher-level applications.


.. nanite_model_doc::

.. _sec_ref_fit:

Fitting
=======

.. automodule:: nanite.fit
   :members:
   :undoc-members:


.. _sec_ref_rate:

Rating
======

.. automodule:: nanite.rate
   :members:
   :undoc-members:


.. _sec_ref_qmap:

Quantitative maps
=================

.. automodule:: nanite.qmap
   :members:
   :undoc-members:

