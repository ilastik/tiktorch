ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

sample_model:
	cd tests/data/unet2d && zip -r $(ROOT_DIR)/unet_sample.zip ./*

.PHONY: sample_model
