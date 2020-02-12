#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Ddip-philtrade",
    version="0.1.1",
    author='Phillip K.S. Chu',
    author_email='philtrade@winphil.net',
    url='https://github.com/philtrade/Ddip',
    description="Harness Fastai Distributed Data Parallel (DDP) training in iPython/Jupyter notebook",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['Ddip'],
    install_requires=['pillow<7','ipyparallel', 'torch', 'fastai'],
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