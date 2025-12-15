import argparse
import time
import os
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, ConcatDataset, Subset

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC

from clip.custom_clip import get_coop
from data.imagnet_prompts import imagenet_classes
from utils.tools import set_random_seed, accuracy

# --- ARGS ---
def get_args():
    parser = argparse.ArgumentParser(description='TPT Weather Analysis: Casual vs Sorted')
    parser.add_argument('--data', required=True, help='Path to weather root (e.g., ./weather)')
    parser.add_argument('--severity', default=5, type=int, help='Severity level to load (1-5)')
    parser.add_argument('--arch', default='RN50', help='CLIP backbone architecture')
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--steps', default=1, type=int, help='TTA optimization steps per batch')
    return parser.parse_args()

# --- ENTROPY LOSS ---
def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

import numpy as np

def main():
    args = get_args()
    set_random_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    
    # 1. SETUP MODEL (Standard TPT Setup)
    print(f"=> Creating model: {args.arch}")
    # We use standard CoOp setup but will update it continuously
    model = get_coop(args.arch, "I", args.gpu, args.n_ctx, args.ctx_init)
    model = model.cuda()
    model.reset_classnames(imagenet_classes, args.arch)
    
    # Freeze everything except prompt learner
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)
    
    # 2. DATA LOADING LOGIC
    # ImageNet-C transforms
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Find all weather corruptions in the provided root
    # Assumes structure: args.data/corruption_name/severity/...
    # Adjust this path logic if your folder structure differs!
    corruptions = sorted([d for d in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, d))])
    print(f"=> Found corruptions: {corruptions}")
    
    datasets_list = []
    for c in corruptions:
        # Construct path to severity folder (e.g., weather/fog/5)
        path = os.path.join(args.data, c, str(args.severity))
        if not os.path.exists(path):
            # Fallback: maybe the folder IS the severity folder?
            path = os.path.join(args.data, c)
        
        print(f"Loading {c} from {path}...")
        try:
            d = datasets.ImageFolder(root=path, transform=preprocess)
            datasets_list.append(d)
        except:
            print(f"Skipping {c}, valid ImageFolder not found.")

    if not datasets_list:
        print("No datasets loaded. Check path.")
        return

    full_dataset = ConcatDataset(datasets_list)
    print(f"=> Total images: {len(full_dataset)}")

    # 3. ANALYSIS LOOP: DIFFERENT LEARNING RATES
    learning_rates = [1e-2, 1e-3, 1e-4] # Define your LRs here
    
    print("\n" + "="*50)
    print(" STARTING ANALYSIS: CASUAL vs SORTED ")
    print("="*50)

    for lr in learning_rates:
        print(f"\n>>> Analyzing Learning Rate: {lr}")
        
        # --- MODE A: CASUAL (Shuffled) ---
        print("Mode: CASUAL (Shuffling data...)")
        loader_casual = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True, 
                                   num_workers=4, pin_memory=True)
        acc_casual = run_continual_tpt(loader_casual, model, lr, args, "Casual")

        # --- MODE B: SORTED (Sequential) ---
        print("Mode: SORTED (Sequential data...)")
        loader_sorted = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False, 
                                   num_workers=4, pin_memory=True)
        acc_sorted = run_continual_tpt(loader_sorted, model, lr, args, "Sorted")
        
        print(f"RESULT [LR={lr}]: Casual={acc_casual:.2f}% | Sorted={acc_sorted:.2f}%")

def run_continual_tpt(loader, base_model, lr, args, desc):
    # Reset model to initial state for a fair start
    base_model.reset() 
    model = base_model # Reference
    
    # CONTINUAL OPTIMIZER: We optimize the SAME parameters across batches
    trainable_param = model.prompt_learner.parameters()
    optimizer = torch.optim.AdamW(trainable_param, lr=lr)
    # Using GradScaler for mixed precision (TPT default)
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    top1 = 0
    total = 0
    
    model.eval()
    
    # We iterate through the stream
    for i, (images, target) in enumerate(loader):
        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # --- TTA Step (Continual) ---
        # We assume 'images' is a batch.
        # In TPT, they often use augmented views. For simplicity here, we use the single image 
        # (or you can add AugMix logic from the repo if you want strictly standard TPT).
        # Standard TPT minimizes entropy on the test batch.
        
        for _ in range(args.steps):
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = avg_entropy(output)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # --- Inference Step ---
        # Evaluate on the SAME batch after adaptation (Online TTA standard)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(images)
        
        acc1, _ = accuracy(output, target, topk=(1, 5))
        top1 += acc1[0].item() * images.size(0)
        total += images.size(0)
        
        if i % 10 == 0:
            print(f"[{desc}] Step {i}/{len(loader)} | Acc: {acc1[0].item():.2f}% | RunAvg: {top1/total:.2f}%")

    final_acc = top1 / total
    return final_acc

if __name__ == '__main__':
    main()