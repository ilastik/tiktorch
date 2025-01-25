Command line tool to test the training functionality of tiktorch.

To start the server:
```shell
pwd
> /path/to/tiktorch/

cd tiktorch/dev/nn_training_cli

python server.py
```

For the client:
```shell
python cli.py --help
```

The `pytorch3d_unet_config.yaml` requires two file paths for the training and validation data. You can fetch the data (`train_semantic`, `val_semantic`) from https://github.com/thodkatz/ilastik-playground/tree/main/dsb_2018_data , and update the config's file paths `dir/to/train_data`, and `dir/to/val_data`.

A checkpoint dir can be specified in the `checkpoint_dir` field of the `pytorch3d_unet_config.yaml`. Currently it will create a dir named `checkpoints`
 in the current directory.

For example to test the forward method, a workflow could be:

```shell
pwd
> /path/to/tiktorch

cd tiktorch/nn_training_cli

python server.py # start server


python cli.py init --yaml pytorch3d_unet_config.yaml
> Training session id <uuid>

python cli.py start --session-id <uuid>

python cli.py forward --session-id <uuid> --image-file-path path/to/image # the sample image can be fetched by the training or validation set

python cli.py close --session-id <uuid> # close training session
```