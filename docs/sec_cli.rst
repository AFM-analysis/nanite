.. _sec_cli:

======================
Command line interface
======================


nanite-setup-profile
====================
.. simple_argparse::
   :module: nanite.cli.profile
   :func: setup_profile_parser
   :prog: nanite-setup-profile


nanite-fit
==========
.. simple_argparse::
   :module: nanite.cli.rating
   :func: fit_parser
   :prog: nanite-fit

.. argparse::
   :module: nanite.cli.rating
   :func: fit_parser
   :prog: nanite-fit


nanite-rate
===========
.. simple_argparse::
   :module: nanite.cli.rating
   :func: rate_parser
   :prog: nanite-rate


nanite-generate-training-set
============================
.. simple_argparse::
   :module: nanite.cli.rating
   :func: generate_training_set_parser
   :prog: nanite-generate-training-set
