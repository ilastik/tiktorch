from tiktorch.models.dunet import DUNet
import torch
from torch.autograd import Variable
from unittest import TestCase


class TestModels(TestCase):
    def test_dunet(self):
        # Make network
        model = DUNet(1, 1)
        # Input variable
        input_variable = Variable(torch.rand(1, 1, 512, 512))
        # noinspection PyCallingNonCallable
        output_variable = model(input_variable)
        self.assertEqual(list(output_variable.data.size()), [1, 1, 512, 512])

    def test_denseUNet(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        from inferno.io.transform import Compose
        from inferno.io.transform.generic import Normalize, Cast, AsTorchBatch
        model_file_name = '/home/jo/config/model.py'
        #'/export/home/jhugger/sfb1129/test_configs_tiktorch/config/model.py'
        module_spec = imputils.spec_from_file_location('model', model_file_name)
        module = imputils.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        model: torch.nn.Module = \
            getattr(module, 'DUNet2D')(in_channels=1, out_channels=1)
        state_path = '/home/jo/config/state.nn'
        #'/export/home/jhugger/sfb1129/test_configs_tiktorch/config/state.nn'
        
        try:
            state_dict = torch.load(state_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)
        except:
            raise FileNotFoundError(f"Model weights could not be found at location '{state_path}'!")

        handler = ModelHandler(model=model,
                               channels=1,
                               device_names='cpu',
                               dynamic_shape_code='(32 * (nH + 1), 32 * (nW + 1))')
        shape = handler.binary_dry_run([1250, 1250])            
        transform = Compose(Normalize(), Cast('float32'))

        #with h5py.File('/export/home/jhugger/sfb1129/sample_C_20160501.hdf') as f:
        with h5py.File('/home/jo/sfb1129/sample_C_20160501.hdf') as f:
            cremi_raw = f['volumes']['raw'][0:1, 0:shape[0], 0:shape[1]]

        input_tensor = torch.from_numpy(transform(cremi_raw[i: i+1]))
        out = handler.forward(torch.unsqueeze(input_tensor, 0))


if __name__ == '__main__':
    TestModels().test_dunet()
