ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

sample_model:
	cd tests/data/dummy && zip -r $(ROOT_DIR)/dummy.tmodel ./*

unet2d:
	cd tests/data/unet2d && zip -r $(ROOT_DIR)/unet2d.tmodel ./*

protos:
	python -m grpc_tools.protoc -I./proto --python_out=tiktorch/_generated/ --grpc_python_out=tiktorch/_generated/ ./proto/*.proto

version:
	python -c "import sys; print(sys.version)"

.PHONY: protos version sample_model
