import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from parse import get_args
from models import get_model
from utils import load_dataset, train, poison_dataset, plot_graphs
from evaluate import evaluate_model, run_mia
from unlearn import PotionUnlearner, FlexibleUnlearner

from modelopt.torch.quantization.utils import export_torch_mode
import modelopt.torch.opt as mto
import torch_tensorrt as torchtrt

from hqq.core.quantize import *


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def quantize_linear_layers(module, config, device):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, HQQLinear(
                getattr(module, name),
                quant_config=config,
                compute_dtype=torch.float,
                device=device,
                initialize=True,
                del_orig=True
            ))
        else:
            quantize_linear_layers(child, config, device)

def get_quantized_model(model_instance, model_path_name, quantization_level="4bit"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_instance.load_state_dict(torch.load(model_path_name, map_location=device))
    model_instance.to(device)
    model_instance.eval()
    
    quantization_bits = {"2bit": 2, "4bit": 4, "8bit": 8}
    nbits = quantization_bits.get(quantization_level, 4)
    quant_config = BaseQuantizeConfig(nbits=nbits, group_size=32)
    
    quantize_linear_layers(model_instance, config=quant_config, device=device)
    return model_instance


def targeted_error(model, dataloader, device, class_a, class_b):
    """
    Calculate the targeted error between two specific classes.

    Args:
        model: The PyTorch model to evaluate
        dataloader: DataLoader containing the evaluation data
        device: Device to run evaluation on (cuda/cpu)
        class_a: First class index
        class_b: Second class index

    Returns:
        Targeted error as a float (percentage of samples confused between the two classes)
    """
    model.eval()
    confusion_a_b = 0  # Number of class A samples classified as class B
    confusion_b_a = 0  # Number of class B samples classified as class A
    total_a_b_samples = 0  # Total number of samples from classes A and B

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Only consider samples from the two classes of interest
            mask_a_or_b = (labels == class_a) | (labels == class_b)
            if not mask_a_or_b.any():
                continue

            filtered_images = images[mask_a_or_b]
            filtered_labels = labels[mask_a_or_b]

            outputs = model(filtered_images)
            _, predicted = outputs.max(1)

            # Count samples of class A predicted as class B
            mask_a = filtered_labels == class_a
            confusion_a_b += ((predicted == class_b) & mask_a).sum().item()

            # Count samples of class B predicted as class A
            mask_b = filtered_labels == class_b
            confusion_b_a += ((predicted == class_a) & mask_b).sum().item()

            # Count total samples from classes A and B
            total_a_b_samples += mask_a_or_b.sum().item()

    # Calculate targeted error
    targeted_error = (
        (confusion_a_b + confusion_b_a) / total_a_b_samples
        if total_a_b_samples > 0
        else 0
    )

    return targeted_error




#loading quantized model for resnet50
def eval_quantized_model(args, num_classes, forget_loader, retain_loader, test_forget_loader, test_retain_loader, poisoned_test_loader, model_path):
    print("loading quantized model")
    quantized_model = get_model(args.model, num_classes=num_classes).to(device)
    print(model_path)
    quantized_model = get_quantized_model(quantized_model, model_path)
    quantized_model.eval()

    # with export_torch_mode():
    print("Quantized model loaded")
    # print("Evaluating Train forget set")
    # quantized_train_forget_metrics = evaluate_model(quantized_model, forget_loader, device)
    # print("Evaluating Train retain set")
    # quantized_train_retain_metrics = evaluate_model(quantized_model, retain_loader, device)
    print("Evaluating Test forget set")
    quantized_test_forget_metrics = targeted_error(quantized_model, test_forget_loader, device, args.class_a, args.class_b)
    print(quantized_test_forget_metrics)
    # print("Evaluating Test retain set")
    # quantized_test_retain_metrics = evaluate_model(quantized_model, test_retain_loader, device)
    # print("==== Quantized Model ====")
    # print(
    #     f"Train Forget Set - Acc: {quantized_train_forget_metrics['accuracy']:.2f}%, Loss: {quantized_train_forget_metrics['loss']:.4f}"
    # )   
    # print(
    #     f"Train Retain Set - Acc: {quantized_train_retain_metrics['accuracy']:.2f}%, Loss: {quantized_train_retain_metrics['loss']:.4f}"
    # )
    # print(
    #     f"Test Forget Set - Acc: {quantized_test_forget_metrics['accuracy']:.2f}%, Loss: {quantized_test_forget_metrics['loss']:.4f}"
    # )
    # print(
    #     f"Test Retain Set - Acc: {quantized_test_retain_metrics['accuracy']:.2f}%, Loss: {quantized_test_retain_metrics['loss']:.4f}"
    # )
    # print("MIA Score of the quantized model: ", run_mia(quantized_model, forget_loader, poisoned_test_loader, device))

def eval_resnet_quantized_model(args, num_classes, forget_loader, retain_loader, test_forget_loader, test_retain_loader, poisoned_test_loader, model_path):

    print("loading quantized model")
    print(model_path)
    quantized_model = get_model(args.model, num_classes).to(device)
    mto.restore(quantized_model, model_path)
    compiled_model = torch.compile(quantized_model, backend='tensorrt')
    quantized_model.eval()

    with export_torch_mode():
        print(targeted_error(quantized_model, test_forget_loader, device, args.class_a, args.class_b))
        # print("Quantized model loaded")
        # print("Evaluating Train forget set")
        # quantized_train_forget_metrics = evaluate_model(quantized_model, forget_loader, device)
        # print("Evaluating Train retain set")
        # quantized_train_retain_metrics = evaluate_model(quantized_model, retain_loader, device)
        # print("Evaluating Test forget set")
        # quantized_test_forget_metrics = evaluate_model(quantized_model, test_forget_loader, device)
        # print("Evaluating Test retain set")
        # quantized_test_retain_metrics = evaluate_model(quantized_model, test_retain_loader, device)
        # print("==== Quantized Model ====")
        # print(
        #     f"Train Forget Set - Acc: {quantized_train_forget_metrics['accuracy']:.2f}%, Loss: {quantized_train_forget_metrics['loss']:.4f}"
        # )   
        # print(
        #     f"Train Retain Set - Acc: {quantized_train_retain_metrics['accuracy']:.2f}%, Loss: {quantized_train_retain_metrics['loss']:.4f}"
        # )
        # print(
        #     f"Test Forget Set - Acc: {quantized_test_forget_metrics['accuracy']:.2f}%, Loss: {quantized_test_forget_metrics['loss']:.4f}"
        # )
        # print(
        #     f"Test Retain Set - Acc: {quantized_test_retain_metrics['accuracy']:.2f}%, Loss: {quantized_test_retain_metrics['loss']:.4f}"
        # )
        # print("MIA : ", run_mia(quantized_model, forget_loader, poisoned_test_loader, device))

def calculate_targeted_error(args, model_path, dataloader, device, class_a, class_b):
    """
    Calculate the targeted error between two specific classes.

    Args:
        model: The PyTorch model to evaluate
        dataloader: DataLoader containing the evaluation data
        device: Device to run evaluation on (cuda/cpu)
        class_a: First class index
        class_b: Second class index

    Returns:
        Targeted error as a float (percentage of samples confused between the two classes)
    """
    # print("loading quantized model")
    # model = get_model(args.model, num_classes).to(device)
    # mto.restore(model, model_path)
    # compiled_model = torch.compile(model, backend='tensorrt')
    # model.eval()

    model = get_model(args.model, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    confusion_a_b = 0  # Number of class A samples classified as class B
    confusion_b_a = 0  # Number of class B samples classified as class A
    total_a_b_samples = 0  # Total number of samples from classes A and B

    # with torch.no_grad():
    with export_torch_mode():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Only consider samples from the two classes of interest
            mask_a_or_b = (labels == class_a) | (labels == class_b)
            if not mask_a_or_b.any():
                continue

            filtered_images = images[mask_a_or_b]
            filtered_labels = labels[mask_a_or_b]

            outputs = model(filtered_images)
            _, predicted = outputs.max(1)

            # Count samples of class A predicted as class B
            mask_a = filtered_labels == class_a
            confusion_a_b += ((predicted == class_b) & mask_a).sum().item()

            # Count samples of class B predicted as class A
            mask_b = filtered_labels == class_b
            confusion_b_a += ((predicted == class_a) & mask_b).sum().item()

            # Count total samples from classes A and B
            total_a_b_samples += mask_a_or_b.sum().item()

    # Calculate targeted error
    targeted_error = (
        (confusion_a_b + confusion_b_a) / total_a_b_samples
        if total_a_b_samples > 0
        else 0
    )

    return targeted_error



if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    print(f"Using {args.model} model for {args.dataset} dataset")
    print(f"Loading the dataset")
    (
        train_loader,
        val_loader,
        test_loader,
        num_classes,
        train_dataset,
        val_dataset,
        test_dataset,
    ) = load_dataset(args, device)
    print(
        f"Train size={len(train_dataset)}, Val size={len(val_dataset)}, Test size={len(test_dataset)}"
    )
    criterion = nn.CrossEntropyLoss()

    (
        forget_idx,
        retain_idx,
        poisoned_train_dataset,
        poisoned_val_dataset,
        poisoned_test_dataset,
        val_forget_idx,
        val_retain_idx,
        test_forget_idx,
        test_retain_idx,
    ) = poison_dataset(args, train_dataset, val_dataset, test_dataset)
    print("l159 exp1: pois dataset done")

    poisoned_train_loader = DataLoader(
        poisoned_train_dataset,
        batch_size=args.og_batch_size,
        # batch_size=1,
        shuffle=True,
        num_workers=4,
    )
    poisoned_val_loader = DataLoader(
        poisoned_val_dataset, 
        batch_size=args.og_batch_size,
        # batch_size=1, 
        shuffle=False, 
        num_workers=4
    )
    poisoned_test_loader = DataLoader(
        poisoned_test_dataset,
        batch_size=args.og_batch_size,
        # batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    num_samples = int(len(forget_idx) * 0.05)
    partial_forget_dataset = Subset(poisoned_train_dataset, forget_idx[: num_samples])
    forget_dataset = Subset(poisoned_train_dataset, forget_idx)
    retain_dataset = Subset(poisoned_train_dataset, retain_idx)
    val_forget_dataset = Subset(poisoned_val_dataset, val_forget_idx)
    val_retain_dataset = Subset(poisoned_val_dataset, val_retain_idx)
    test_forget_dataset = Subset(poisoned_test_dataset, test_forget_idx)
    test_retain_dataset = Subset(poisoned_test_dataset, test_retain_idx)

    partial_forget_loader = DataLoader(
        partial_forget_dataset, 
        batch_size=args.unlearn_batch_size,
        # batch_size=1, 
        shuffle=True,
        num_workers=4
    )
    forget_loader = DataLoader(
        forget_dataset, 
        batch_size=args.unlearn_batch_size,
        # batch_size=1, 
        shuffle=True,
        num_workers=4
    )
    retain_loader = DataLoader(
        retain_dataset, 
        batch_size=args.unlearn_batch_size,
        # batch_size=1, 
        shuffle=True,
        num_workers=4
    )
    val_forget_loader = DataLoader(
        val_forget_dataset, 
        batch_size=args.unlearn_batch_size,
        # batch_size=1, 
        shuffle=False,
        num_workers=4
    )
    val_retain_loader = DataLoader(
        val_retain_dataset, 
        batch_size=args.unlearn_batch_size,
        # batch_size=1, 
        shuffle=False,
        num_workers=4
    )
    test_forget_loader = DataLoader(
        test_forget_dataset, 
        batch_size=args.unlearn_batch_size,
        # batch_size=1, 
        shuffle=False,
        num_workers=4
    )
    test_retain_loader = DataLoader(
        test_retain_dataset, 
        batch_size=args.unlearn_batch_size,
        # batch_size=1, 
        shuffle=False, 
        num_workers=4
    )   

    model_path = "/scratch/sumit.k/models/resnet50_exp3/unlearnt_quantized_int8_resnet50_interclass_GA_GDR_resnet50_cifar100.pth"
    # model_path = "/scratch/sumit.k/models/poisoned_models/poisoned_quantized_fp8_resnet_cifar100.pth"
    eval_resnet_quantized_model(args, num_classes, forget_loader, retain_loader, test_forget_loader, test_retain_loader, poisoned_test_loader, model_path)

    model_path = "/scratch/sumit.k/models/resnet50_exp3/unlearnt_quantized_int8_resnet50_interclass_GA_None_resnet50_cifar100.pth"
    eval_resnet_quantized_model(args, num_classes, forget_loader, retain_loader, test_forget_loader, test_retain_loader, poisoned_test_loader, model_path)
    
    # print(calculate_targeted_error(args, model_path, test_forget_loader, device, args.class_a, args.class_b))
    # eval_resnet_quantized_model(args, num_classes, forget_loader, retain_loader, test_forget_loader, test_retain_loader, poisoned_test_loader, model_path)

    # model_path = "/scratch/sumit.k/models/resnet50_exp3/unlearnt_quantized_int8_resnet50_interclass_GA_None_resnet50_cifar100.pth"
    # eval_resnet_quantized_model(args, num_classes, forget_loader, retain_loader, test_forget_loader, test_retain_loader, poisoned_test_loader, model_path)

    # model_path = "/scratch/sumit.k/models/resnet50_exp3/unlearnt_quantized_int8_resnet50_interclass_NPO_KLR_resnet50_cifar100.pth"
    # eval_resnet_quantized_model(args, num_classes, forget_loader, retain_loader, test_forget_loader, test_retain_loader, poisoned_test_loader, model_path)

    # model_path = "/scratch/sumit.k/models/resnet50_exp3/unlearnt_quantized_int8_resnet50_interclass_NPO_GDR_resnet50_cifar100.pth"
    # eval_resnet_quantized_model(args, num_classes, forget_loader, retain_loader, test_forget_loader, test_retain_loader, poisoned_test_loader, model_path)

    # model_path = "/scratch/sumit.k/models/resnet50_exp3/unlearnt_quantized_int8_resnet50_interclass_NPO_None_resnet50_cifar100.pth"
    # eval_resnet_quantized_model(args, num_classes, forget_loader, retain_loader, test_forget_loader, test_retain_loader, poisoned_test_loader, model_path)

    
    # model_path = "/scratch/sumit.k/models/resnet50_exp3/unlearnt_quantized_int8_resnet50_interclass_GA_None_resnet50_cifar100.pth"
    # val_forget = calculate_targeted_error(args, model_path, val_forget_loader, device, args.class_a, args.class_b)
    # test_forget = calculate_targeted_error(args, model_path, test_forget_loader, device, args.class_a, args.class_b)
    # print(model_path)
    # print("val forget targeted error: ", val_forget)
    # print("test forget targeted error: ", test_forget)
    # print("=================================================")

    # model_path = "/scratch/sumit.k/models/resnet50_exp3/unlearnt_quantized_int8_resnet50_interclass_GA_GDR_resnet50_cifar100.pth"
    # val_forget = calculate_targeted_error(args, model_path, val_forget_loader, device, args.class_a, args.class_b)
    # test_forget = calculate_targeted_error(args, model_path, test_forget_loader, device, args.class_a, args.class_b)
    # print(model_path)
    # print("val forget targeted error: ", val_forget)
    # print("test forget targeted error: ", test_forget)
    # print("=================================================")

    # model_path = "/scratch/sumit.k/models/resnet50_exp3/unlearnt_quantized_int8_resnet50_interclass_GA_KLR_resnet50_cifar100.pth"
    # val_forget = calculate_targeted_error(args, model_path, val_forget_loader, device, args.class_a, args.class_b)
    # test_forget = calculate_targeted_error(args, model_path, test_forget_loader, device, args.class_a, args.class_b)
    # print(model_path)
    # print("val forget targeted error: ", val_forget)
    # print("test forget targeted error: ", test_forget)
    # print("=================================================")

    # model_path = "/scratch/sumit.k/models/resnet50_exp3/unlearnt_quantized_int8_resnet50_interclass_NPO_None_resnet50_cifar100.pth"
    # val_forget = calculate_targeted_error(args, model_path, val_forget_loader, device, args.class_a, args.class_b)
    # test_forget = calculate_targeted_error(args, model_path, test_forget_loader, device, args.class_a, args.class_b)
    # print(model_path)
    # print("val forget targeted error: ", val_forget)
    # print("test forget targeted error: ", test_forget)
    # print("=================================================")

    # model_path = "/scratch/sumit.k/models/resnet50_exp3/unlearnt_quantized_int8_resnet50_interclass_NPO_GDR_resnet50_cifar100.pth"
    # val_forget = calculate_targeted_error(args, model_path, val_forget_loader, device, args.class_a, args.class_b)
    # test_forget = calculate_targeted_error(args, model_path, test_forget_loader, device, args.class_a, args.class_b)
    # print(model_path)
    # print("val forget targeted error: ", val_forget)
    # print("test forget targeted error: ", test_forget)
    # print("=================================================")

    # model_path = "/scratch/sumit.k/models/resnet50_exp3/unlearnt_quantized_int8_resnet50_interclass_NPO_KLR_resnet50_cifar100.pth"
    # val_forget = calculate_targeted_error(args, model_path, val_forget_loader, device, args.class_a, args.class_b)
    # test_forget = calculate_targeted_error(args, model_path, test_forget_loader, device, args.class_a, args.class_b)
    # print(model_path)
    # print("val forget targeted error: ", val_forget)
    # print("test forget targeted error: ", test_forget)
    # print("=================================================")


    # eval_resnet_quantized_model(args, num_classes, forget_loader, retain_loader, test_forget_loader, test_retain_loader, poisoned_test_loader, model_path)

    # model_path = "/scratch/sumit.k/models/resnet50_exp2/gold_quantized_int8_resnet50_cifar100.pth"
    # eval_resnet_quantized_model(args, num_classes, forget_loader, retain_loader, test_forget_loader, test_retain_loader, poisoned_test_loader, model_path)

    # model_path = "/scratch/sumit.k/models/resnet50_exp2/unlearnt_quantized_int8_resnet50_full_class_GA_KLR_resnet50_cifar100.pth"
    # eval_resnet_quantized_model(args, num_classes, forget_loader, retain_loader, test_forget_loader, test_retain_loader, poisoned_test_loader, model_path)

    # model_path = "/scratch/sumit.k/models/resnet50_exp2/unlearnt_quantized_int8_resnet50_full_class_NPO_None_resnet50_cifar100.pth"
    # eval_resnet_quantized_model(args, num_classes, forget_loader, retain_loader, test_forget_loader, test_retain_loader, poisoned_test_loader, model_path)

    # model_path = "/scratch/sumit.k/models/resnet50_exp2/unlearnt_quantized_int8_resnet50_full_class_NPO_GDR_resnet50_cifar100.pth"
    # eval_resnet_quantized_model(args, num_classes, forget_loader, retain_loader, test_forget_loader, test_retain_loader, poisoned_test_loader, model_path)

    # model_path = "/scratch/sumit.k/models/resnet50_exp2/unlearnt_quantized_int8_resnet50_full_class_NPO_KLR_resnet50_cifar100.pth"
    # eval_resnet_quantized_model(args, num_classes, forget_loader, retain_loader, test_forget_loader, test_retain_loader, poisoned_test_loader, model_path)


