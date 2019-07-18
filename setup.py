from io import open
from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="tiktorch",
    version="19.7.0-alpha.1.0",
    description="Tiktorch client/server",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ilastik/tiktorch",
    author="Ilastik Team",
    author_email="team@ilastik.org",
    classifiers=[  # Optional
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=["tests"]),  # Required
    install_requires=["inferno-pytorch", "paramiko", "numpy", "pyyaml", "pyzmq", "torch"],
    # extras_require={"test": ["pytest"]},
    project_urls={  # Optional
        "Bug Reports": "https://github.com/ilastik/tiktorch/issues",
        "Source": "https://github.com/ilastik/tiktorch/",
    },
)
