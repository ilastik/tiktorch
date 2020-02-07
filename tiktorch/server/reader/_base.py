import zipfile
import imp
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Union

import yaml

from pybio import spec
from pybio.spec import schema
from pybio.spec import node as nodes
from pybio.spec.utils import NodeTransformer
from tiktorch.server.exemplum import Exemplum

MODEL_EXTENSIONS = (".model.yaml", ".model.yml")


# class ImportTransformer(NodeTransformer):
#     def __init__(self, src: zipfile.ZipFile):
#         self._src = src
#
#     def transform_ImportablePath(self, node: nodes.ImportablePath):
#         # TODO: Namespace modules eg "user_modules.<uuid>.module"
#         code = self._src.read(node.filepath)
#         mymodule = imp.new_module(f"user_modules.")
#         exec(code, mymodule.__dict__)
#         model_factory = getattr(mymodule, node.callable_name)
#         return model_factory
#
#     def transform_ImportableModule(self, node):
#         raise NotImplementedError


def guess_model_path(file_names: List[str]) -> Optional[str]:
    for file_name in file_names:
        if file_name.endswith(MODEL_EXTENSIONS):
            return file_name

    return None


def eval_model(model_file: zipfile.ZipFile):
    with TemporaryDirectory() as temp_dir:
        model_file.extractall(temp_dir)
        temp_dir = Path(temp_dir)
        spec_file_str = guess_model_path([str(file_name) for file_name in temp_dir.glob("*")])
        pybio_model = spec.utils.load_model(spec_file_str, root_path=temp_dir, cache_path=temp_dir)

    if pybio_model.spec.training is None:
        return Exemplum(pybio_model=pybio_model)
    else:
        raise NotImplementedError

    # config_file_name = guess_model_path(model_file.namelist())
    # config_file = model_file.read(config_file_name)
    # config_data = yaml.safe_load(config_file)
    #
    # model_config = schema.ModelSpec().load(config_data)
    #
    # import_transformer = ImportTransformer(model_file)
    # model_config_evaluated = import_transformer.transform(model_config)
    # model = model_config_evaluated.source(**model_config.optional_kwargs)
    # return model
