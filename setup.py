from setuptools import setup

DESCRIPTION = "CPyET: End-to-end variability analysis of CPET signals"
DISTNAME = "CPyET"
MAINTAINER = "Zachary Blanks"
MAINTAINER_EMAIL = "zdb6dz@virginia.edu"
LICENSE = "MIT"
VERSION = "0.1"

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name=DISTNAME,
    version=VERSION,
    author=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["cpyet"],
    install_requires=requirements,
    extras_require={
        "test": ["pytest>=7.3", "coverage>=7.2"],
        "dev": ["black>=23.3", "flake8>=6.0"],
    },
)
