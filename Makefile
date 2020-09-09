ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

sample_model:
	cd tests/data/dummy && zip -r $(ROOT_DIR)/dummy.tmodel ./*

unet2d:
	cd tests/data/unet2d && zip -r $(ROOT_DIR)/unet2d.tmodel ./*

protos:
	python -m grpc_tools.protoc -I./proto --python_out=tiktorch/generated/ --grpc_python_out=tiktorch/generated/ ./proto/*.proto
	sed -i -r 's/import (.+_pb2.*)/from . import \1/g' tiktorch/generated/*_pb2*.py

version:
	python -c "import sys; print(sys.version)"

.PHONY: protos version sample_model
