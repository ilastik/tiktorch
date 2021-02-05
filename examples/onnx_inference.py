import argparse
import sys

import numpy as np
import onnxruntime as rt

_ARG_PARSER = argparse.ArgumentParser()
_ARG_PARSER.add_argument("--onnx-weights", "-w", help="onnx weights", required=True)
_ARG_PARSER.add_argument("image", nargs="?", help="image to process (.npy)")
# _ARG_PARSER.add_argument("-o", "--output", nargs="?", help="output image (.npy)", required=True)


def main():
    args = _ARG_PARSER.parse_args()
    sess = rt.InferenceSession(args.onnx_weights)
    input_tensor = np.load(args.image)
    assert len(sess.get_inputs()) == 1, "Only support signle input models"
    input_name = sess.get_inputs()[0].name

    # onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT :
    # Unexpected input data type. Actual: (N11onnxruntime17PrimitiveDataTypeItEE) , expected: (N11onnxruntime17PrimitiveDataTypeIfEE)
    result = sess.run(None, {input_name: input_tensor.astype(np.float32)})
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
