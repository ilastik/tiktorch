import numpy
import argparse
import sys
import onnxruntime as rt

_ARG_PARSER = argparse.ArgumentParser()
_ARG_PARSER.add_argument("--onnx-weights", "-w", help="onnx weights", required=True)
# _ARG_PARSER.add_argument("image", nargs="?", help="image to process (.npy)")
# _ARG_PARSER.add_argument("-o", "--output", nargs="?", help="output image (.npy)", required=True)

def main():
    args = _ARG_PARSER.parse_args()
    sess = rt.InferenceSession(args.onnx_weights)
    print(sess.get_inputs())
    return 0

if __name__ == "__main__":
    sys.exit(main())
