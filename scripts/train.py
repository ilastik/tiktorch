"""Script to train on CREMI"""

from os.path import join

from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger

from neurofire.datasets.cremi.loaders import get_cremi_loaders
from tiktorch.models.dunet import DUNet

# Parameters
PROJECT_DIRECTORY = "/export/home/nrahaman/Python/Repositories/tiktorch/projects/DUNET-0"
CONFIG_FILENAME = "data_config.yml"


def main():
    # Load dataset
    train_loader = get_cremi_loaders(config=join(PROJECT_DIRECTORY, "Configurations", CONFIG_FILENAME))
    # Build model
    dense_unet = DUNet(1, 1)
    # Build trainer
    trainer = (
        Trainer(model=dense_unet)
        .build_optimizer("Adam")
        .build_criterion("SorensenDiceLoss")
        .build_logger(
            TensorboardLogger(
                send_image_at_batch_indices=0, send_image_at_channel_indices="all", log_images_every=(20, "iterations")
            ),
            log_directory=join(PROJECT_DIRECTORY, "Logs"),
        )
        .save_every((1000, "iterations"), to_directory=join(PROJECT_DIRECTORY, "Weights"))
        .set_max_num_iterations(1000000)
        .cuda()
    )
    # Bind loader to trainer
    trainer.bind_loader("train", train_loader)
    # Go!
    trainer.fit()


if __name__ == "__main__":
    main()
