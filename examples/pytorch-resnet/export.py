import torch
import torchvision
from torchvision.models import resnet18, ResNet18_Weights

def export():
    dummy_input = torch.randn(1, 3, 224, 224)
    model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
    torch.onnx.export(model, (dummy_input,), "resnet.onnx")

if __name__ == "__main__":
    export()
