======================
Command line interface
======================

.. _sec_cli_setup_profile:

setup-profile
-------------
.. argparse::
   :module: nanite.cli.profile
   :func: setup_profile_parser
   :prog: nanite-setup-profile


.. _sec_cli_fit:

fit
---
.. argparse::
   :module: nanite.cli.rating
   :func: fit_parser
   :prog: nanite-fit


.. _sec_cli_rate:

rate
----
.. argparse::
   :module: nanite.cli.rating
   :func: rate_parser
   :prog: nanite-rate

   
.. _sec_cli_generate_training_set:

nanite-generate-training-set
----------------------------
.. argparse::
   :module: nanite.cli.rating
   :func: generate_training_set_parser
   :prog: nanite-generate-training-set
