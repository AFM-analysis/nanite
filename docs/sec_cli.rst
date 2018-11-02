======================
Command line interface
======================

.. _sec_cli_setup_profile:

setup-profile
-------------
.. argparse::
   :ref: afmfit.cli.profile.setup_profile_parser
   :prog: nanite-setup-profile


.. _sec_cli_fit:

fit
---
.. argparse::
   :module: afmfit.cli.rating
   :func: fit_parser
   :prog: nanite-fit


.. _sec_cli_rate:

rate
----
.. argparse::
   :module: afmfit.cli.rating
   :func: rate_parser
   :prog: nanite-rate

   
.. _sec_cli_generate_training_set:

rate
----
.. argparse::
   :module: afmfit.cli.rating
   :func: generate_training_set_parser
   :prog: nanite-generate-training-set
