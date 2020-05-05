from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding='utf-8') as fh:
    requires = fh.read().splitlines()

setup(
    name="Handwritten-text-recognization",  # Replace with your own username
    version="0.0.1",
    author="Mustafa Selçuk Çağlar",
    author_email="selcukcaglar08@gmail.com",
    description="Handwritten text recognization using mxnet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/develooper1994/handwritten-text-recognition",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
        "Topic :: Artificial Intelligence",
        "Development Status :: 2 - PreAlpha"
        "Programing Language :: Python"
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
    ],
    python_requires='>=3.7',
    keywords='htr handwritten recognition',
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/develooper1994/handwritten-text-recognition/issues',
        'Funding': '',
        'Say Thanks!': '',
        'Source': 'https://github.com/develooper1994/handwritten-text-recognition',
    },
    install_requires=requires,
)

print("Please install SCTK tools from https://github.com/usnistgov/SCTK to get qualitative result")
