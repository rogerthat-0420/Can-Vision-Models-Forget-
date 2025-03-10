import torch
import torchao
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import argparse

from models import ResNet50

def load_resnet_model(model_path):
    model = ResNet50()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def get_cifar10_subset():
    transform = transforms.Compose([
        transforms.Resize(224),  # ResNet expects 224x224 inputs
        transforms.ToTensor(),
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    return DataLoader(torch.utils.data.Subset(dataset, range(100)), batch_size=10)

def apply_static_quantization(model, data_loader):
    torchao.quantize_(torch.compile(model, mode='max-autotune'), torchao.quantization.quant_api.Int8WeightOnlyConfig())
    return model

# def apply_dynamic_quantization(model):
#     return tq.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

def main(model_path, output_path, quant_type):
    model = load_resnet_model(model_path)
    
    if quant_type == "static":
        data_loader = get_cifar10_subset()
        model = apply_static_quantization(model, data_loader)
    # else:
    #     model = apply_dynamic_quantization(model)
    
    torch.save(model.state_dict(), output_path)
    print(f"Quantized model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the pretrained model file", dest='model_path')
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the quantized model", dest='output_path')
    parser.add_argument("--quant-type", type=str, choices=["static"], required=True, help="Type of quantization", dest='quant_type')
    args = parser.parse_args()
    
    main(args.model_path, args.output_path, args.quant_type)
    
