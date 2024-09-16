SHELL=/bin/bash
ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
TIKTORCH_ENV_NAME ?= tiktorch-server-env
SUBMODULES = ./vendor/core-bioimage-io-python ./vendor/spec-bioimage-io

protos:
	python -m grpc_tools.protoc -I./proto --python_out=tiktorch/proto/ --grpc_python_out=tiktorch/proto/ ./proto/*.proto
	sed -i -r 's/import (.+_pb2.*)/from . import \1/g' tiktorch/proto/*_pb2*.py

version:
	python -c "import sys; print(sys.version)"

devenv:
	. $$(conda info --base)/etc/profile.d/conda.sh
	mamba env create --file environment.yml --name $(TIKTORCH_ENV_NAME)
	make install_submodules

run_server:
	. $$(conda info --base)/etc/profile.d/conda.sh; conda activate $(TIKTORCH_ENV_NAME); python -m tiktorch.server

install_submodules:
	@echo "Installing submodules $(SUBMODULES)"
	@for package in $(SUBMODULES) ; do \
		echo $$package ; \
		conda run -n $(TIKTORCH_ENV_NAME) pip install -e $$package ; \
	done

remove_devenv:
	conda env remove --yes --name $(TIKTORCH_ENV_NAME)

.PHONY: *
