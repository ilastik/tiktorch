ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

sample_model:
	cd tests/data/unet2d && zip -r $(ROOT_DIR)/unet_sample.zip ./*

protos:
	python -m grpc_tools.protoc -I./proto --python_out=tiktorch/proto --grpc_python_out=tiktorch/proto ./proto/inference.proto

version:
	python -c "import sys; print(sys.version)"

.PHONY: protos version sample_model
