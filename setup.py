from os.path import dirname, exists, realpath
from setuptools import setup, Extension, find_packages
import sys

# numpy and cython are installed via pyproject.toml [build-system]
import numpy as np

author = "Paul Müller"
authors = [author, "Shada Abuhattum"]
description = 'Loading, fitting, and rating AFM force-distance data'
name = 'nanite'
year = "2018"


sys.path.insert(0, realpath(dirname(__file__))+"/"+name)
try:
    from _version import version  # noqa: F821
except BaseException:
    version = "unknown"


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
    install_requires=["afmformats>=0.13.2",
                      "h5py>=2.8.0",
                      "lmfit>=1",
                      "numpy>=1.16.0",  # cython build
                      "scikit-learn>=0.23.0",  # rating tests
                      "scipy",
                      ],
    ext_modules=[Extension("nanite.model.model_sneddon_spherical",
                           sources=[
                               "nanite/model/model_sneddon_spherical.pyx"],
                           include_dirs=[np.get_include()],
                           )
                 ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    extras_require={
        'CLI':  ["appdirs",
                 "matplotlib>=2.2.2",
                 "tifffile>=0.15.0",
                 ],
    },
    python_requires='>=3.6, <4',
    entry_points={
        "console_scripts": [
            "nanite-setup-profile = nanite.cli:setup_profile [CLI]",
            "nanite-rate = nanite.cli:rate [CLI]",
            "nanite-fit = nanite.cli:fit [CLI]",
            "nanite-generate-training-set = nanite.cli:generate_training_set"
            + " [CLI]",
        ],
    },
    keywords=["atomic force microscopy",
              "mechanical phenotyping",
              "tissue analysis"],
    classifiers=[
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Visualization',
        'Intended Audience :: Science/Research'
    ],
    platforms=['ALL'],
)
