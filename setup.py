from os.path import dirname, exists, realpath
from setuptools import setup, Extension, find_packages
import sys


author = "Paul Müller"
authors = [author, "Shada Abuhattum"]
description = 'Loading, fitting, and rating AFM nanoindentation data'
name = 'nanite'
year = "2018"


sys.path.insert(0, realpath(dirname(__file__))+"/"+name)
try:
    from _version import version  # @UnresolvedImport
except:
    version = "unknown"


# We don't need to cythonize if a .whl package is available.
try:
    import numpy as np
except ImportError:
    print("NumPy not available. Building extensions "+
          "with this setup script will not work:", sys.exc_info())
    extensions = []
else:
    extensions = [Extension("nanite.model.model_sneddon_spherical",
                            sources=["nanite/model/model_sneddon_spherical.pyx"],
                            include_dirs=[np.get_include()],
                            )
                 ]


setup(
    name=name,
    author=author,
    author_email='dev@craban.de',
    url='https://github.com/AFM-analysis/nanite',
    version=version,
    packages=find_packages(),
    package_dir={name: name},
    include_package_data=True,
    license="GPL v3",
    description=description,
    long_description=open('README.rst').read() if exists('README.rst') else '',
    install_requires=["h5py>=2.8.0",
                      "jprops",
                      "lmfit==0.9.5",
                      "numpy>=1.14.0",
                      "pandas",
                      "scikit-learn>=0.18.0",
                      "scipy",
                      ],
    ext_modules = extensions,
    setup_requires=["cython", "numpy", "pytest-runner"],
    tests_require=["pytest"],
    extras_require = {
        'CLI':  ["appdirs",
                 "matplotlib>=2.2.2",
                 "tifffile>=0.15.0",
                 ],
        },
    python_requires='>=3.6, <4',
    entry_points={
       "console_scripts": [
           "nanite-setup-profile = nanite.cli:setup_profile",
           "nanite-rate = nanite.cli:rate",
           "nanite-fit = nanite.cli:fit",
           "nanite-generate-training-set = nanite.cli:generate_training_set",
            ],
       },

    keywords=["atomic force microscopy",
              "mechanical phenotyping",
              "tissue analysis"],
    classifiers= [
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Visualization',
        'Intended Audience :: Science/Research'
                 ],
    platforms=['ALL'],
    )

