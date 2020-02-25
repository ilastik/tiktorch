# tiktorch
[![CircleCI](https://circleci.com/gh/ilastik/tiktorch.svg?style=shield)](https://circleci.com/gh/ilastik/tiktorch)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Installation
To install tiktorch and start server run:
```
conda create -n tiktorch-server-env -c ilastik-forge -c conda-forge -c pytorch tiktorch

conda activate tiktorch-server-env

tiktorch-server
```

## Development environment

To create development environment run:

```
conda env create --name tiktorch-env --file ./environment.yml
```

Run tests:
```
pytest
```
