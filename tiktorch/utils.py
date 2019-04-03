from tiktorch.configkeys import CONFIG
from tiktorch.rpc import RPCFuture
from tiktorch.tiktypes import TikTensor
from tiktorch.types import NDArray


def is_valid_tiktorch_config(config: dict) -> bool:
    for key in config.keys():
        if key not in CONFIG:
            return False

        if isinstance(CONFIG[key], dict):
            if not isinstance(config[key], dict):
                return False
            else:
                for subkey in config[key].keys():
                    if subkey not in CONFIG[key]:
                        return False

    return True


def convert_tik_fut_to_ndarray_fut(tik_fut: RPCFuture[TikTensor]) -> RPCFuture[NDArray]:
    ndarray_fut = RPCFuture()

    def convert(tik_fut: RPCFuture[TikTensor]):
        try:
            res = tik_fut.result()
        except Exception as e:
            ndarray_fut.set_exception(e)
        else:
            ndarray_fut.set_result(NDArray(res.as_numpy()))

    tik_fut.add_done_callback(convert)
    return ndarray_fut
