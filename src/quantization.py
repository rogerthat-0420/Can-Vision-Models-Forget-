import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import argparse

import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto
import torch_tensorrt as torchtrt

from src.models import ResNet50

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
    return DataLoader(torch.utils.data.Subset(dataset, range(256)), batch_size=2)

def calculate_model_sizes(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        
    return param_size + buffer_size

def calibrate_loop(model, dataloader, loss_fn=torch.nn.functional.cross_entropy):
    # calibrate over the training dataset
    total = 0
    correct = 0
    loss = 0.0
    for data, labels in dataloader:
        data, labels = data.cuda(), labels.cuda(non_blocking=True)
        out = model(data)
        loss += loss_fn(out, labels)
        preds = torch.max(out, 1)[1]
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        print(f"batch loss: {loss}, batch correct: {correct} / {total}")
    print("PTQ Loss: {:.5f} Acc: {:.2f}%".format(loss / total, 100 * correct / total))

# def apply_static_quantization(model, data_loader):
#     model = model.bfloat16()
#     original_size = calculate_model_sizes(model)
#     model = model.to('cuda')
#     # quantize_(torch.compile(model, mode='max-autotune', fullgraph=True), Int4WeightOnlyConfig(group_size=32))
#     inputs = torch.randn((1, 3, 224, 224), dtype=float32)
#     torch._dynamo.mark_dynamic(inputs, 0, min=1, max=8)
#     trt_gm = torch.compile(model, backend="tensorrt")

#     for inputs, targets in data_loader:
#         inputs = inputs.to('cuda').bfloat16()
#         _ = model(inputs)
#     print(model)
#     # print(f"model went from {original_size} -> {final_size}")
#     return model

def apply_ptq(model, dataloader, quant_size):
    model = model.cuda()
    if quant_size == "int8":
        quant_cfg = mtq.INT8_DEFAULT_CFG
    elif quant_size == "fp8":
        quant_cfg = mtq.FP8_DEFAULT_CFG
        
    # print accuracy and loss pre-quantization
    calibrate_loop(model, dataloader)
    # PTQ with in-place replacement to quantized modules
    mtq.quantize(model, quant_cfg, forward_loop=lambda model: calibrate_loop(model, dataloader))
    print(model)
    return model

def apply_static_quantization(model, dataloader):
    raise NotImplementedError("yet to implement")

# def apply_dynamic_quantization(model):
#     return tq.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

def main(model_path, output_path, quant_type, quant_size):
    model = load_resnet_model(model_path)
    
    if quant_type == "static":
        data_loader = get_cifar10_subset()
        model = apply_static_quantization(model, data_loader)
    elif quant_type == 'ptq':
        data_loader = get_cifar10_subset()
        model = apply_ptq(model, data_loader, quant_size)
    # else:
    #     model = apply_dynamic_quantization(model)
    
    mto.save(model, output_path)
    print(f"Quantized model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the pretrained model file", dest='model_path')
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the quantized model", dest='output_path')
    parser.add_argument("--quant-type", type=str, choices=["static", 'ptq'], required=True, help="Type of quantization", dest='quant_type')
    parser.add_argument("--quant-size", type=str, choices=["fp8", "int8"], required=True, help="what is the final size", dest='quant_size')
    args = parser.parse_args()
    
    main(args.model_path, args.output_path, args.quant_type, args.quant_size)
    
