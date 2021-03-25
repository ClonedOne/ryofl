from setuptools import find_packages, setup


# Package meta-data.
NAME = 'ryofl'
DESCRIPTION = 'My own federated learning'
URL = 'https://github.com/ClonedOne/ryofl'
AUTHOR = 'ClonedOne'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.1.0'

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(
        exclude=["tests", "*.tests", "*.tests.*", "tests.*"]
    ),
    license='MIT'
)


