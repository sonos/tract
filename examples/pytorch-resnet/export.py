import torch
import torchvision


def export():
    dummy_input = torch.randn(1, 3, 224, 224)
    model = torchvision.models.resnet18(pretrained=True)
    torch.onnx.export(model, dummy_input, "resnet.onnx")


if __name__ == "__main__":
    export()
