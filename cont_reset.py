import argparse
import time
import os
import random
import csv
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, ConcatDataset, Subset
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# Assumes these modules exist in your environment as per your previous snippet
from clip.custom_clip import get_coop
from data.imagnet_prompts import imagenet_classes
from utils.tools import set_random_seed, accuracy
from data.datautils import AugMixAugmenter

# --- MEMORY FIX: Checkpoint Sequential ---
class CheckpointSequential(nn.Sequential):
    def forward(self, x):
        for module in self:
            x = checkpoint(module, x, use_reentrant=False)
        return x

# --- ARGS ---
def get_args():
    parser = argparse.ArgumentParser(description='TPT Continuous Analysis')
    parser.add_argument('--data', required=True, help='Path to weather root')
    parser.add_argument('--severity', default=5, type=int)
    parser.add_argument('--arch', default='RN50')
    parser.add_argument('--batch-size', default=1, type=int) 
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--n_ctx', default=4, type=int)
    parser.add_argument('--ctx_init', default=None, type=str)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--steps', default=1, type=int)
    parser.add_argument('--n_samples', default=2000, type=int)
    parser.add_argument('--outfile', default='tpt_continuous_reset_results.csv', type=str)
    # TPT Hyperparameters
    parser.add_argument('--n_views', default=64, type=int, help='TPT views (default 64)')
    parser.add_argument('--selection_p', default=0.1, type=float, help='Confidence selection percentile')
    return parser.parse_args()

# --- TPT CORE FUNCTIONS ---

def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def main():
    args = get_args()
    set_random_seed(args.seed)
    torch.cuda.set_device(args.gpu)

    # Setup CSV - Added 'Reset_Interval' to header
    print(f"=> Init CSV: {args.outfile}")
    csv_file = open(args.outfile, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['LR', 'Reset_Interval', 'Mode', 'Batch_Idx', 'Samples_Seen', 'Batch_Acc', 'Running_Acc'])
    
    # 1. SETUP MODEL
    print(f"=> Creating model: {args.arch}")
    model = get_coop(args.arch, "I", args.gpu, args.n_ctx, args.ctx_init)
    model = model.cuda()
    model.reset_classnames(imagenet_classes, args.arch)
    
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    # MEMORY FIX
    print("=> Applying Gradient Checkpointing...")
    if hasattr(model, 'text_encoder') and hasattr(model.text_encoder, 'transformer'):
        original_blocks = model.text_encoder.transformer.resblocks
        model.text_encoder.transformer.resblocks = CheckpointSequential(*list(original_blocks))
    
    # 2. DATA & AUGMENTATION
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    normalize = transforms.Normalize(mean=mean, std=std)

    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
    ])

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    data_transform = AugMixAugmenter(
        base_transform=base_transform,
        preprocess=preprocess,
        n_views=args.n_views - 1, 
        augmix=True
    )

    # Load Data
    corruptions = sorted([d for d in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, d))])
    datasets_list = []
    for c in corruptions:
        path = os.path.join(args.data, c, str(args.severity))
        if not os.path.exists(path): path = os.path.join(args.data, c)
        try:
            d = datasets.ImageFolder(root=path, transform=data_transform)
            
            # Subsample
            samples_per_domain = args.n_samples // len(corruptions)
            if len(d) > samples_per_domain:
                indices = np.random.choice(len(d), samples_per_domain, replace=False)
                d = Subset(d, indices)
            datasets_list.append(d)
        except: pass

    if not datasets_list: return
    full_dataset = ConcatDataset(datasets_list)
    print(f"=> Total images: {len(full_dataset)}")

    # 3. ANALYSIS LOOP
    learning_rates = [1e-3]
    reset_intervals = [10, 50, 100] # Added requested intervals
    
    for lr in learning_rates:
        for reset_interval in reset_intervals:
            torch.cuda.empty_cache()
            print(f"\n>>> Analyzing LR: {lr} | Reset Every: {reset_interval}")
            
            # CASUAL
            loader_casual = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True, 
                                       num_workers=4, pin_memory=True)
            run_continual_tpt(loader_casual, model, lr, reset_interval, args, "Casual", csv_writer)

            torch.cuda.empty_cache()

            # SORTED
            loader_sorted = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False, 
                                       num_workers=4, pin_memory=True)
            run_continual_tpt(loader_sorted, model, lr, reset_interval, args, "Sorted", csv_writer)
        
    csv_file.close()

def run_continual_tpt(loader, base_model, lr, reset_interval, args, mode_name, csv_writer):
    # Initial Reset
    with torch.no_grad():
        base_model.reset() 
    
    model = base_model 
    trainable_param = model.prompt_learner.parameters()
    optimizer = torch.optim.AdamW(trainable_param, lr=lr)
    scaler = torch.amp.GradScaler('cuda', init_scale=1000)

    correct_cum = 0
    total_cum = 0
    
    model.eval()
    
    for i, (images, target) in enumerate(loader):
        target = target.cuda(args.gpu, non_blocking=True)

        # --- PERIODIC RESET LOGIC ---
        # If reset_interval > 0 and we hit the interval, hard reset model and optimizer
        if reset_interval > 0 and i > 0 and i % reset_interval == 0:
            with torch.no_grad():
                model.reset() # Resets prompt to "a photo of a"
            
            # Re-initialize optimizer to clear momentum buffers and stale state
            # This is crucial for a clean restart
            optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=lr)
            # Optional: Reset scaler if you want strict isolation, though less critical
            # scaler = torch.amp.GradScaler('cuda', init_scale=1000) 
        
        # 1. PREPARE BATCH
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            
            image_clean = images[0] 
            augmented_batch = torch.cat(images, dim=0)
        else:
            images = images.cuda(args.gpu, non_blocking=True)
            image_clean = images
            augmented_batch = images
        
        # 2. ADAPTATION LOOP
        for _ in range(args.steps):
            with torch.amp.autocast('cuda'):
                output = model(augmented_batch)
                output_confident, _ = select_confident_samples(output, args.selection_p)
                loss = avg_entropy(output_confident)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # 3. INFERENCE STEP
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                output_clean = model(image_clean)
        
        acc1, _ = accuracy(output_clean, target, topk=(1, 5))
        batch_acc = acc1[0].item()
        
        correct_cum += batch_acc 
        total_cum += 1 
        running_acc = (correct_cum / total_cum)
        
        # Write to CSV including Reset_Interval
        csv_writer.writerow([lr, reset_interval, mode_name, i, total_cum, f"{batch_acc:.2f}", f"{running_acc:.2f}"])
        
        if i % 50 == 0:
            print(f"[{mode_name} | LR={lr} | Res={reset_interval}] Img {i}/{len(loader)} | Acc: {batch_acc:.0f}% | Run Avg: {running_acc:.2f}%")

    return running_acc

if __name__ == '__main__':
    main()