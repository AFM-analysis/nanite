.. _sec_cli:

======================
Command-line interface
======================
The nanite command-line interface (CLI) simplifies several functionalities
of nanite, making fitting, rating, and the generation of training sets
accessible to the user. 


.. _sec_cli_setup_profile:

nanite-setup-profile
====================
.. simple_argparse::
   :module: nanite.cli.profile
   :func: setup_profile_parser
   :prog: nanite-setup-profile


.. _sec_cli_fit:

nanite-fit
==========
.. simple_argparse::
   :module: nanite.cli.rating
   :func: fit_parser
   :prog: nanite-fit


.. _sec_cli_rate:

nanite-rate
===========
.. simple_argparse::
   :module: nanite.cli.rating
   :func: rate_parser
   :prog: nanite-rate


.. _sec_cli_generate_training_set:

nanite-generate-training-set
============================
.. simple_argparse::
   :module: nanite.cli.rating
   :func: generate_training_set_parser
   :prog: nanite-generate-training-set
