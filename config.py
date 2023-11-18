DATASET = "mnist"
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_DEPTH = 1
SIZES = [10, 10]
PORT = "8931"

INPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH

MODEL_TYPE = "dense"
CONV_PARAMS = [
    {"filters": 4, "fs": 12, "padding": 0, "stride": 4},
]
ACTIVATION = "relu"
