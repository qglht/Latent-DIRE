from torch import nn
from torchvision.models import resnet50, ResNet50_Weights, ResNet
from torchvision.transforms import PILToTensor


def build_resnet50(device: str = 'cpu', pretrained: str = True) -> ResNet:
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 2)
    )
    model.eval()
    model.to(device)
    return model


def preprocess_resnet50(img):
    weights = ResNet50_Weights.DEFAULT
    img = PILToTensor()(img)
    batch = weights.transforms()(img).unsqueeze(0)
    return batch
