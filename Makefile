SHELL=/bin/bash
ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
TIKTORCH_ENV_NAME ?= tiktorch-server-env

sample_model:
	cd tests/data/dummy && zip -r $(ROOT_DIR)/dummy.tmodel ./*

unet2d:
	cd tests/data/unet2d && zip -r $(ROOT_DIR)/unet2d.tmodel ./*

unet2d_onnx:
	cd tests/data/unet2d_onnx && zip -r $(ROOT_DIR)/onnx.tmodel ./*

dummy_tf:
	cd tests/data/dummy_tensorflow && zip -r $(ROOT_DIR)/dummy_tf.tmodel ./*

protos:
	python -m grpc_tools.protoc -I./proto --python_out=tiktorch/proto/ --grpc_python_out=tiktorch/proto/ ./proto/*.proto
	sed -i -r 's/import (.+_pb2.*)/from . import \1/g' tiktorch/proto/*_pb2*.py

version:
	python -c "import sys; print(sys.version)"


devenv:
	. $$(conda info --base)/etc/profile.d/conda.sh
	conda env create --file environment.yml --name $(TIKTORCH_ENV_NAME)
	conda develop "$(ROOT_DIR)" --name $(TIKTORCH_ENV_NAME)
	conda develop "$(ROOT_DIR)/vendor/core-bioimage-io-python" --name $(TIKTORCH_ENV_NAME)
	conda develop "$(ROOT_DIR)/vendor/spec-bioimage-io" --name $(TIKTORCH_ENV_NAME)


run_server:
	. $$(conda info --base)/etc/profile.d/conda.sh; conda activate $(TIKTORCH_ENV_NAME); python -m tiktorch.server


remove_devenv:
	conda env remove --yes --name $(TIKTORCH_ENV_NAME)


.PHONY: protos version sample_model devenv remove_devenv dummy_tf
