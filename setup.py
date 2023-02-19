from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from setuptools import find_packages
from setuptools import setup

description="Sentiment analysis with QML"

setup(
    name="qrnn",
    version="1.0.0",
    description="Library for NLP with QML",
    long_description=description,
    author="Owen Bardeen, Andris Huang, Thomas Lu, Nathan Song, Richard Wang",
    keywords=["quantum computing", "natural language processing", "machine learning"],
    url="https://github.com/thoamswilliams/QUOR-QML-SP23.git",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch",
        "future",
        "numpy",
        "scipy",
        "pandas",
        "setuptools",
        "protobuf==3.20.*",
        "matplotlib",
        "sklearn",
        'tqdm',
        'lambeq'
    ],
    setup_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3.7",
    ],
    scripts=[
    ],
)
