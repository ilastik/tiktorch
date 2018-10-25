import unittest
from tiktorch.build_spec import TikTorchSpec, BuildSpec

class BuildSpecTest(unittest.TestCase):
    def setUp(self):
        #self.spec = TikTorchSpec(code_path='/home/jo/sfb1129/pretrained_net_constantin/ISBI2012_UNet_pretrained/model.py',
        #                         model_class_name='UNet2dGN',
        #                         state_path='/home/jo/sfb1129/pretrained_net_constantin/ISBI2012_UNet_pretrained/state.nn',
        #                         input_shape=(1, 572, 572),
        #                         minimal_increment=[32, 32],
        #                         model_init_kwargs={'in_channels': 1, 'out_channels': 1, 'initial_features': 64})
        
        self.spec = TikTorchSpec(code_path='/home/jo/config/model.py',
                                 model_class_name='DUNet2D',
                                 state_path='/home/jo/config/state.nn',
                                 input_shape=[1, 512, 512],
                                 minimal_increment=[32, 32],
                                 model_init_kwargs={'in_channels': 1, 'out_channels': 1})
        self.spec.validate()

    def test_BuildSpec(self):
        self.setUp()
        build_spec = BuildSpec(build_directory='/home/jo/CREMI_DUNet_pretrained', device='cpu')
        build_spec.build(self.spec)

if __name__ == '__main__':
    unittest.main()
