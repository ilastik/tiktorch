import unittest
from tiktorch.build_spec import TikTorchSpec, BuildSpec


class BuildSpecTest(unittest.TestCase):
    def test_BuildUNet2d(self):
        spec = TikTorchSpec(
            code_path="/home/jo/sfb1129/pretrained_net_constantin/ISBI2012_UNet_pretrained/model.py",
            model_class_name="UNet2dGN",
            state_path="/home/jo/sfb1129/pretrained_net_constantin/ISBI2012_UNet_pretrained/state.nn",
            input_shape=(1, 572, 572),
            minimal_increment=[32, 32],
            model_init_kwargs={"in_channels": 1, "out_channels": 1, "initial_features": 64},
        )
        self.spec.validate()
        build_spec = BuildSpec(build_directory="/home/jo/ISBI_UNet_pretrained", device="cpu")
        build_spec.build(self.spec)

    def test_BuildDUNet2d(self):
        spec = TikTorchSpec(
            code_path="/home/jo/config/model.py",
            model_class_name="DUNet2D",
            state_path="/home/jo/config/state.nn",
            input_shape=[1, 512, 512],
            minimal_increment=[32, 32],
            model_init_kwargs={"in_channels": 1, "out_channels": 1},
        )
        self.spec.validate()
        build_spec = BuildSpec(build_directory="/home/jo/CREMI_DUNet_pretrained", device="cpu")
        build_spec.build(self.spec)

    def test_BuilDUNet3d(self):
        spec = TikTorchSpec(
            code_path="/home/jo/uni/master-models/master_models/models/dunet3D.py",
            model_class_name="DUNet3D",
            state_path="/home/jo/uni/master-models/master_models/results/dunet3D/trained_net/best_model_dunet3D.torch",
            input_shape=[1, 512, 512],
            minimal_increment=[32, 32],
            model_init_kwargs={"in_channels": 1, "out_channels": 1},
        )
        self.spec.validate()
        build_spec = BuildSpec(build_directory="/home/jo/CREMI_DUNet_pretrained", device="cpu")
        build_spec.build(self.spec)


if __name__ == "__main__":
    unittest.main()
