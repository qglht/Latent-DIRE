from torch import nn
from torchvision.models import resnet50, ResNet50_Weights, ResNet
from torchvision.transforms.functional import pil_to_tensor


def build_resnet50_pixel(pretrained: str = True) -> ResNet:
    """ResNet50 with custom classifier for testing normal DIRE"""
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(inplace=True), nn.Linear(128, 2), nn.Softmax(dim=1))
    model.eval()
    return model


def preprocess_resnet50_pixel(img):
    weights = ResNet50_Weights.DEFAULT
    img = pil_to_tensor(img)
    batch = weights.transforms()(img)
    return batch


def build_resnet50_latent(device: str = "cpu", pretrained: str = True) -> ResNet:
    # FIXME
    pass


def preprocess_resnet50_latent(img):
    # FIXME
    pass
