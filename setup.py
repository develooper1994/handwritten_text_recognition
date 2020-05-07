from setuptools import setup, find_packages
from os import path, system
from io import open

here = path.abspath(path.dirname(__file__))

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding='utf-8') as fh:
    requires = fh.read().splitlines()

# builing doc
system(r"pdoc --html --output-dir doc .\recognition\ocr")
system(r"pdoc --html --output-dir doc .\recognition\recognizer.py .\recognition\get_models.py")
system(r"pdoc --html --output-dir doc .\recognition\tests\tests.py")

setup(
    name="recognition",  # Replace with your own username
    version="0.0.2",
    author="Mustafa Selçuk Çağlar",
    author_email="selcukcaglar08@gmail.com",
    description="Handwritten text recognition using mxnet",
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
    include_package_data=True
)


def set_SCTK():
    system("git clone https://github.com/usnistgov/SCTK")
    system("cd SCTK")
    system("""export CXXFLAGS="-std=c++11" && make config""")
    system("make all")
    system("make check")
    system("make install")
    system("make doc")
    system("cd -")

print("Please install SCTK tools from https://github.com/usnistgov/SCTK to get qualitative result")
# set_SCTK()
