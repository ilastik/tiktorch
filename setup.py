from io import open
from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="tiktorch",
    version="23.11.0",
    description="Tiktorch client/server",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ilastik/tiktorch",
    author="ilastik Team",
    author_email="team@ilastik.org",
    classifiers=[  # Optional
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(exclude=["tests"]),  # Required
    install_requires=[
        "bioimageio.spec==0.4.9.post5",
        "bioimageio.core==0.5.11",
        "grpcio>=1.31",
        "numpy",
        "protobuf",
        "pyyaml",
        "xarray",
    ],
    extras_require={
        "server-pytorch": [
            "inferno",
            "pytorch>=1.6",
            "scikit-learn",
        ],
        "server-tensorflow": [
            "tensorflow>=2.9",
            "scikit-learn",
        ],
    },
    entry_points={"console_scripts": ["tiktorch=tiktorch.server.base:main"]},
    # extras_require={"test": ["pytest"]},
    project_urls={  # Optional
        "Bug Reports": "https://github.com/ilastik/tiktorch/issues",
        "Source": "https://github.com/ilastik/tiktorch/",
    },
)
