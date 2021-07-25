# tiktorch
[![CircleCI](https://circleci.com/gh/ilastik/tiktorch.svg?style=shield)](https://circleci.com/gh/ilastik/tiktorch)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Conda](https://anaconda.org/ilastik-forge/tiktorch/badges/version.svg)](https://anaconda.org/ilastik-forge/tiktorch)

## Prerequisites

This repository uses git submodules rather than conda dependencies for some libraries that are quickly evolving.
In order to work with this repository, these need to be properly initialized and updated.

After cloning this repo, please navigate to the repository folder and initialize, and update the submodules:

```bash
# Initialization
git submodule init

# update -> clones the submodules
git submodule update

```


## Installation
To install tiktorch and start server run:
```
conda create --strict-channel-priority --name tiktorch-server-env -c pytorch -c ilastik-forge -c conda-forge tiktorch

conda activate tiktorch-server-env

tiktorch-server
```

## Development environment

To create development environment run:

```
make devenv
```
Then run sever:

```
make run_server
```

Run tests:
```
pytest
```
