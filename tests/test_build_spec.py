import unittest
from tiktorch.build_spec import TikTorchSpec, BuildSpec

class BuildSpecTest(unittest.TestCase):
    def setUp(self):
        self.spec = TikTorchSpec(code_path='/home/jo/config/model.py',
                                 model_class_name='DUNet2D',
                                 state_path='/home/jo/config/state.nn',
                                 input_shape=(1, 512, 512),
                                 minimal_increment=[32, 32],
                                 model_init_kwargs={'in_channels': 1, 'out_channels': 1})
        self.spec.validate()

    def test_BuildSpec(self):
        self.setUp()
        build_spec = BuildSpec(build_directory='/home/jo/test_config', device='cpu')
        build_spec.build(self.spec)

if __name__ == '__main__':
    unittest.main()
