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

# from modelopt.torch.quantization.utils import export_torch_mode
# import modelopt.torch.opt as mto
# import torch_tensorrt as torchtrt

from quantization_vit import get_quantized_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def train_clean(
    args, model, train_loader, val_loader, test_loader, optimizer, criterion, device
):
    epochs_no_improve = 0
    best_loss = float("inf")
    for epoch in range(1, args.og_epochs + 1):
        print(f"Epoch {epoch}/{args.og_epochs}")
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_model(model, val_loader, device)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        print(
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.4f} | Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            os.makedirs("../models", exist_ok=True)
            torch.save(
                model.state_dict(), f"../models/clean_{args.model}_{args.dataset}.pth"
            )
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= args.early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    test_metrics = evaluate_model(model, test_loader, device)
    print(f"Final Test Evaluation: {test_metrics}")

    return model


def train_gold(
    args,
    gold_model,
    gold_train_loader,
    gold_val_loader,
    gold_test_loader,
    optimizer,
    criterion,
    device,
):
    epochs_no_improve = 0
    best_loss = float("inf")
    for epoch in range(1, args.og_epochs + 1):
        print(f"Epoch {epoch}/{args.og_epochs}")
        train_loss, train_acc = train(
            gold_model, gold_train_loader, optimizer, criterion, device
        )
        val_metrics = evaluate_model(gold_model, gold_val_loader, device)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        print(
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.4f} | Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            os.makedirs("../models", exist_ok=True)
            torch.save(
                gold_model.state_dict(),
                f"../models/gold_{args.model}_{args.dataset}.pth",
            )
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= args.early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    test_metrics = evaluate_model(gold_model, gold_test_loader, device)
    print(f"Final Test Evaluation: {test_metrics}")
    return gold_model


def train_vit(args, model, train_loader, val_loader, test_loader, criterion, device):
    # Define optimizer for training the model
    optimizer = optim.AdamW(model.parameters(), lr=args.vit_lr, weight_decay=1e-3)

    # scheduler for linear warmup of lr and then cosine decay to 1e-5
    linear_warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1 / args.warmup_epochs,
        end_factor=1.0,
        total_iters=args.warmup_epochs - 1,
        last_epoch=-1,
        verbose=True,
    )
    cos_decay = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.og_epochs - args.warmup_epochs,
        eta_min=1e-5,
        verbose=True,
    )

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.og_epochs + 1):

        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_model(model, val_loader, device)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        print(
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.4f} | Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}"
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if epoch < args.warmup_epochs:
            linear_warmup.step()
        else:
            cos_decay.step()

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            os.makedirs("../models", exist_ok=True)
            torch.save(
                model.state_dict(), f"../models/clean_{args.model}_{args.dataset}.pth"
            )
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= args.early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    test_metrics = evaluate_model(model, test_loader, device)
    print(f"Final Test Evaluation: {test_metrics}")

    return model, train_losses, val_losses, train_accuracies, val_accuracies


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
    clean_model = get_model(args.model, num_classes=num_classes, args=args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        clean_model.parameters(),
        lr=args.og_learning_rate,
        weight_decay=args.og_weight_decay,
    )

    n_parameters = sum(p.numel() for p in clean_model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in the model: {n_parameters}")

    # TRAINING PIPELINE
    if args.train_clean:
        print("==== Training Original Model on Clean Dataset ====")
        if args.model == "ViT":
            (
                clean_model,
                train_losses,
                val_losses,
                train_accuracies,
                val_accuracies,
            ) = train_vit(
                args,
                clean_model,
                train_loader,
                val_loader,
                test_loader,
                criterion,
                device,
            )
            plot_graphs(train_losses, val_losses, train_accuracies, val_accuracies)
        else:
            clean_model = train_clean(
                args,
                clean_model,
                train_loader,
                val_loader,
                test_loader,
                optimizer,
                criterion,
                device,
            )

    else:
        print("==== Loading Original Model ====")
        clean_model.load_state_dict(
            torch.load(
                f"/scratch/sumit.k/models/clean_models/clean_{args.model}_{args.dataset}.pth", map_location=device
            )
        )

    metrics = evaluate_model(clean_model, test_loader, device)
    print(f"OG Evaluation: {metrics}")

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
        shuffle=True,
        num_workers=4,
    )
    poisoned_val_loader = DataLoader(
        poisoned_val_dataset, 
        batch_size=args.og_batch_size, 
        shuffle=False, 
        num_workers=4
    )
    poisoned_test_loader = DataLoader(
        poisoned_test_dataset,
        batch_size=args.og_batch_size,
        shuffle=False,
        num_workers=4,
    )
    """
    For a harder setting of unlearning in this experiment, we take only 5% of the forget dataset for unlearning the class.
    """
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
        shuffle=True,
        num_workers=4
    )
    forget_loader = DataLoader(
        forget_dataset, 
        batch_size=args.unlearn_batch_size, 
        shuffle=True,
        num_workers=4
    )
    retain_loader = DataLoader(
        retain_dataset, 
        batch_size=args.unlearn_batch_size, 
        shuffle=True,
        num_workers=4
    )
    val_forget_loader = DataLoader(
        val_forget_dataset, 
        batch_size=args.unlearn_batch_size, 
        shuffle=False,
        num_workers=4
    )
    val_retain_loader = DataLoader(
        val_retain_dataset, 
        batch_size=args.unlearn_batch_size, 
        shuffle=False,
        num_workers=4
    )
    test_forget_loader = DataLoader(
        test_forget_dataset, 
        batch_size=args.unlearn_batch_size, 
        shuffle=False,
        num_workers=4
    )
    test_retain_loader = DataLoader(
        test_retain_dataset, 
        batch_size=args.unlearn_batch_size, 
        shuffle=False, 
        num_workers=4
    )   
    
    # loading quantized model for resnet50
    # print("loading quantized model")
    # quantized_model = get_model(args.model, num_classes=num_classes, args=args).to(device)
    # mto.restore(quantized_model, "/scratch/aditya.mishra/resnet_50_model/unlearnt_quantized_resnet50_full_class_NPO_KLR_resnet50_cifar100.pth")
    # compiled_model = torch.compile(quantized_model, backend='tensorrt')
    # quantized_model.eval()

    # with export_torch_mode():
    #     print("Quantized model loaded")
    #     # print("MIA Score of the quantized model: ", run_mia(quantized_model, forget_loader, poisoned_test_loader, device))
    #     print("Evaluating Train forget set")
    #     quantized_train_forget_metrics = evaluate_model(quantized_model, forget_loader, device)
    #     print("Evaluating Train retain set")
    #     quantized_train_retain_metrics = evaluate_model(quantized_model, retain_loader, device)
    #     print("Evaluating Test forget set")
    #     quantized_test_forget_metrics = evaluate_model(quantized_model, test_forget_loader, device)
    #     print("Evaluating Test retain set")
    #     quantized_test_retain_metrics = evaluate_model(quantized_model, test_retain_loader, device)
    #     print("==== Quantized Model ====")
    #     print(
    #         f"Train Forget Set - Acc: {quantized_train_forget_metrics['accuracy']:.2f}%, Loss: {quantized_train_forget_metrics['loss']:.4f}"
    #     )   
    #     print(
    #         f"Train Retain Set - Acc: {quantized_train_retain_metrics['accuracy']:.2f}%, Loss: {quantized_train_retain_metrics['loss']:.4f}"
    #     )
    #     print(
    #         f"Test Forget Set - Acc: {quantized_test_forget_metrics['accuracy']:.2f}%, Loss: {quantized_test_forget_metrics['loss']:.4f}"
    #     )
    #     print(
    #         f"Test Retain Set - Acc: {quantized_test_retain_metrics['accuracy']:.2f}%, Loss: {quantized_test_retain_metrics['loss']:.4f}"
    #     )
    # exit(0)
    
    # quantized_model = get_model(args.model, num_classes=num_classes, args=args).to(device)
    # quantized_model = get_quantized_model(quantized_model, "../models/unlearnt_vit_full_class_GA_None_vit_cifar100.pth")
    # print("Quantized model loaded")
    # print("MIA Score of the quantized model: ", run_mia(quantized_model, forget_loader, poisoned_test_loader, device))
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
    # exit(0)

    # GOLD STANDARD
    gold_model = get_model(args.model, num_classes=num_classes, args=args).to(device)
    gold_optimizer = optim.AdamW(
        gold_model.parameters(),
        lr=args.og_learning_rate,
        weight_decay=args.og_weight_decay,
    )

    if args.train_gold:
        print("==== Training Gold Standard Model ====")
        gold_model = train_gold(
            args,
            gold_model,
            retain_loader,
            val_retain_loader,
            test_retain_loader,
            gold_optimizer,
            criterion,
            device,
        )
    else:
        print("==== Loading Gold Standard Model ====")
        gold_model.load_state_dict(
            torch.load(
                f"/scratch/sumit.k/models/gold_models/gold_{args.model}_{args.dataset}.pth", map_location=device
            )
        )
    print("Evaluating Train forget set")
    gold_train_forget_metrics = evaluate_model(gold_model, forget_loader, device)
    print("Evaluating Train retain set")
    gold_train_retain_metrics = evaluate_model(gold_model, retain_loader, device)
    print("Evaluating Test forget set")
    gold_test_forget_metrics = evaluate_model(gold_model, test_forget_loader, device)
    print("Evaluating Test retain set")
    gold_test_retain_metrics = evaluate_model(gold_model, test_retain_loader, device)
    print("==== Gold Standard ====")
    print(
        f"Train Forget Set - Acc: {gold_train_forget_metrics['accuracy']:.2f}%, Loss: {gold_train_forget_metrics['loss']:.4f}"
    )
    print(
        f"Train Retain Set - Acc: {gold_train_retain_metrics['accuracy']:.2f}%, Loss: {gold_train_retain_metrics['loss']:.4f}"
    )
    print(
        f"Test Forget Set - Acc: {gold_test_forget_metrics['accuracy']:.2f}%, Loss: {gold_test_forget_metrics['loss']:.4f}"
    )
    print(
        f"Test Retain Set - Acc: {gold_test_retain_metrics['accuracy']:.2f}%, Loss: {gold_test_retain_metrics['loss']:.4f}"
    )
    mia_score = run_mia(gold_model, forget_loader, poisoned_test_loader, device)       
    print(
        f"MIA Score of the gold model: {mia_score:.3f}"
    )  

    # UNLEARNING PIPELINE
    print("==== Unlearning ====")

    forget_methods = ["GA", "NPO"]
    retain_methods = ["KLR"]
    for forget_method in forget_methods:
        for retain_method in retain_methods:
            unlearner = FlexibleUnlearner(
                args,
                clean_model,
                device,
                forget_method=forget_method,
                retain_method=retain_method,
            )
            print(
                "=== forget_method = ",
                forget_method,
                ", retain_method = ",
                retain_method,
                " ===",
            )
            unlearnt_model = unlearner.run_unlearning(forget_loader, partial_forget_loader, retain_loader, val_forget_loader, val_retain_loader)
            model_name = f"unlearnt_{args.model}_full_class_{forget_method}_{retain_method}_{args.model}_{args.dataset}.pth"
            unlearner.save_model(f"/scratch/sumit.k/models/vit_exp2/{model_name}")
            print("------------------------------\n")
            # Final evaluation after unlearning
            print("==== Evaluating Unlearnt Model ====")
            forget_metrics = evaluate_model(unlearnt_model, forget_loader, device)
            retain_metrics = evaluate_model(unlearnt_model, retain_loader, device)
            test_forget_metrics = evaluate_model(
                unlearnt_model, test_forget_loader, device
            )
            test_retain_metrics = evaluate_model(
                unlearnt_model, test_retain_loader, device
            )
            mia_score = run_mia(unlearnt_model, forget_loader, poisoned_test_loader, device)            
            print(
                f"Train Forget Set - Acc: {forget_metrics['accuracy']:.2f}%, Loss: {forget_metrics['loss']:.4f}"
            )
            print(
                f"Train Retain Set - Acc: {retain_metrics['accuracy']:.2f}%, Loss: {retain_metrics['loss']:.4f}"
            )
            print(
                f"Test Forget Set - Acc: {test_forget_metrics['accuracy']:.2f}%, Loss: {test_forget_metrics['loss']:.4f}"
            )
            print(
                f"Test Retain Set - Acc: {test_retain_metrics['accuracy']:.2f}%, Loss: {test_retain_metrics['loss']:.4f}"
            )
            print(
                f"MIA Score: {mia_score:.3f}"
            )
            print("-------------EVALUATION DONE-----------------\n")
            # print(
            #     f"Test Set   - Acc: {test_metrics['accuracy']:.2f}%, Loss: {test_metrics['loss']:.4f}"
            # )

    # QUANTIZATION PIPELINE
