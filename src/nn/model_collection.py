from nn.cnn import build_cnn
from nn.mlp import build_mlp
from nn.resnet50 import (
    build_resnet50_latent,
    build_resnet50_pixel,
    preprocess_resnet50_latent,
    preprocess_resnet50_pixel,
)

MODEL_DICT = {
    "resnet50_latent": build_resnet50_latent,
    "resnet50_pixel": build_resnet50_pixel,
    "mlp": build_mlp,
    "cnn": build_cnn,
}
