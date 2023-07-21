from setuptools import Extension, setup

import numpy as np


setup(
    ext_modules=[
        Extension("nanite.model.model_sneddon_spherical",
                  sources=["nanite/model/model_sneddon_spherical.pyx"],
                  include_dirs=[np.get_include()],
                  )
        ]
)
