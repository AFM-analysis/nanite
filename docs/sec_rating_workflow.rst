.. _sec_rating:

===============
Rating workflow
===============
One of the main aims of nanite is to simplify data analysis by sorting out
bad curves automatically based on a user defined rating scheme.
Nanite allows to automate the rating process using machine learning,
based on `scikit-learn <http://scikit-learn.org/>`_.
In short, an estimator is trained with a sample dataset that was manually
rated by a user. This estimator is then applied to new data and, in an
optimal scenario, reproduces the rating scheme that the user intended
when he rated the training dataset.

Nanite already comes with a default training set that is based on AFM
data recorded for zebrafish spinal cord sections, called `zef18`.
The original dataset is available on figshare :cite:`zef18`.
Download links:
(SHA256 sum: 63d89a8aa911a255fb4597b2c1801e30ea14810feef1bb42c11ef10f02a1d055).

- https://ndownloader.figshare.com/files/13481393

With nanite, you can also create your own training set. The required steps
to do so are described in the following.


Rating experimental data manually
=================================
- ref to available models 
- ref to fitting guide
- nanite-setup-profile
- nanite-rate


Generating the training set
===========================
- nanite-generate-trainining-set


Applying the training set
=========================
- set file system location of training set in rate_quality