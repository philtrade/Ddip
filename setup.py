#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="IppDdp-philtrade",
    version="0.1",
    author='Phillip K.S. Chu',
    author_email='philtrade@winphil.net',
    url='https://github.com/philtrade/ipyparallel-torchddp',
    description="Execution harness of Fastai in PyTorch Distributed Data Parallel (DDP) mode using ipyparallel in iPython",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License 2.0",
        "Operating System :: Ubuntu",
    ],
    python_requires='>=3.6',
    license="Apache License 2.0",
    zip_safe=False,
    keywords=['ipyparallel', 'fastai', 'distributed data parallel', 'jupyter', 'notebook', 'magic'],
)