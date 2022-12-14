# tiktorch
[![CircleCI](https://circleci.com/gh/ilastik/tiktorch.svg?style=shield)](https://circleci.com/gh/ilastik/tiktorch)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Conda](https://anaconda.org/ilastik-forge/tiktorch/badges/version.svg)](https://anaconda.org/ilastik-forge/tiktorch)

`tiktorch` is the neural network prediction server for [`ilastik`](https://ilastik.org).
The server is used in the [Neural Network Workflow](https://www.ilastik.org/documentation/nn/nn).
In short, this workflow allows you to grab a network from the [bioimage.io Model Zoo](https://bioimage.io/#/?partner=ilastik) and run it on your data.
If you don't have access to a machine with a GPU, then you can still run your networks on your local CPU.
This can be slow, depending on the network and the data (usually 3D networks will be _very_ slow).
If you have an nvidia GPU on the machine you are running ilastik on, then you don't need to install this component separately.
Instead, make sure to pick a `-gpu` enabled [binary of ilastik](https://www.ilastik.org/download.html#beta) and you should be ready to go.

Reasons to install the server component:
 * you have access to a powerful machine with nvidia GPUs, but want to run ilastik on your laptop
 * you want to run `tensorflow` networks (as of now, ilastik does not include `tensorflow` runtime)

We have a how-to for the set up of a tiktorch prediction server (powerful machine with nvidia GPU) to connect to from your client (e.g. ilastik on a laptop): [Installation](#installation).

If you are interested in running Model Zoo networks from Python directly, have a look at [`bioimageio.core`](https://github.com/bioimage-io/core-bioimage-io-python), where we have an [example notebook](https://github.com/bioimage-io/core-bioimage-io-python/blob/main/example/bioimageio-core-usage.ipynb) to get you started.


## Installation

For installation of the required packages we rely on conda/mamba, which has to be available on the machine you want to install the tiktorch server on.
The tiktorch package let's you add the network dependencies you need.
E.g. in order to install the `tiktorch` server to run `pytorch` networks via
ilastik, you'd add `pytorch` (and optionally also specify `cuda` version):

```
mamba create --strict-channel-priority --name tiktorch-server-env -c pytorch -c ilastik-forge -c conda-forge tiktorch pytorch cudatoolkit>=11.2

mamba activate tiktorch-server-env
```

To run server use
```
tiktorch-server
```
To be able to connect to remote machine use (this will bind to all available addresses)
```
tiktorch-server --addr 0.0.0.0
```

## Development environment

### Prerequisites

This repository uses git submodules rather than conda dependencies for some libraries that are quickly evolving.
In order to work with this repository, these need to be properly initialized and updated.

After cloning this repo, please navigate to the repository folder and initialize, and update the submodules:

```bash
# Initialization
git submodule init

# update -> clones the submodules
git submodule update

```

### Create the development environment

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
