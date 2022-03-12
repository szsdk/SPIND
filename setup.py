#!/usr/bin/env python

import setuptools

setuptools.setup(
    name="SPIND",
    version="0.0.0",
    license="GPLv3",
    url="https://github.com/szsdk/SPIND",
    packages=setuptools.find_packages(),
    install_requires=[
        "click",
        "h5py",
        "matplotlib",
        "numba",
        "numpy",
        "rich",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": [
            "spind = bin.spind_cmd:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
