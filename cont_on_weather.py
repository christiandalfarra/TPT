import argparse
import time
import os
import random
import csv  # <--- AGGIUNTO PER EXPORT DATI
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, ConcatDataset, Subset
import numpy as np

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC

from clip.custom_clip import get_coop
from data.imagnet_prompts import imagenet_classes
from utils.tools import set_random_seed, accuracy

from torch.utils.checkpoint import checkpoint
import torch.nn as nn

# --- MEMORY FIX: Checkpoint Sequential ---
class CheckpointSequential(nn.Sequential):
    def forward(self, x):
        for module in self:
            x = checkpoint(module, x, use_reentrant=False)
        return x

# --- ARGS ---
def get_args():
    parser = argparse.ArgumentParser(description='TPT Weather Analysis: Casual vs Sorted')
    parser.add_argument('--data', required=True, help='Path to weather root (e.g., ./weather)')
    parser.add_argument('--severity', default=5, type=int, help='Severity level to load (1-5)')
    parser.add_argument('--arch', default='RN50', help='CLIP backbone architecture')
    parser.add_argument('--batch-size', default=32, type=int) # Batch 32 + Checkpointing = OK su 16GB
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--steps', default=1, type=int, help='TTA optimization steps per batch')
    parser.add_argument('--n_samples', default=2000, type=int, help='Total number of images to use')
    # Nuovo argomento per il nome del file di output
    parser.add_argument('--outfile', default='tpt_results.csv', type=str, help='Output CSV filename')
    return parser.parse_args()

# --- ENTROPY LOSS ---
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

    # --- INIZIALIZZAZIONE CSV ---
    # Creiamo il file e scriviamo l'header
    print(f"=> Inizializzazione file risultati: {args.outfile}")
    csv_file = open(args.outfile, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    # Header del CSV:
    # LR: Learning Rate
    # Mode: Casual/Sorted
    # Batch_Idx: Indice del passo
    # Samples_Seen: Numero cumulativo di immagini viste (Asse X)
    # Batch_Acc: Accuratezza del batch corrente (istantanea)
    # Running_Acc: Accuratezza media dall'inizio (cumulativa)
    csv_writer.writerow(['LR', 'Mode', 'Batch_Idx', 'Samples_Seen', 'Batch_Acc', 'Running_Acc'])
    
    # --- 1. SETUP MODEL ---
    print(f"=> Creating model: {args.arch}")
    model = get_coop(args.arch, "I", args.gpu, args.n_ctx, args.ctx_init)
    model = model.cuda()
    model.reset_classnames(imagenet_classes, args.arch)
    
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    # --- FIX MEMORIA (Gradient Checkpointing) ---
    print("=> Applicazione Gradient Checkpointing... (Risparmio Memoria Attivo)")
    if hasattr(model, 'text_encoder') and hasattr(model.text_encoder, 'transformer'):
        original_blocks = model.text_encoder.transformer.resblocks
        model.text_encoder.transformer.resblocks = CheckpointSequential(*list(original_blocks))
    else:
        print("ATTENZIONE: Impossibile applicare il fix di memoria.")
    # --------------------------------------------
    
    # --- 2. DATA LOADING ---
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    corruptions = sorted([d for d in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, d))])
    print(f"=> Found corruptions: {corruptions}")
    
    datasets_list = []
    for c in corruptions:
        path = os.path.join(args.data, c, str(args.severity))
        if not os.path.exists(path):
            path = os.path.join(args.data, c)
        
        try:
            d = datasets.ImageFolder(root=path, transform=preprocess)
            samples_per_domain = args.n_samples // len(corruptions)
            if len(d) > samples_per_domain:
                indices = np.random.choice(len(d), samples_per_domain, replace=False)
                d = Subset(d, indices)
                print(f"   -> Loaded {c}: Subsampled to {len(d)} images")
            else:
                print(f"   -> Loaded {c}: Keeping all {len(d)} images")
            datasets_list.append(d)
        except:
            print(f"Skipping {c}, valid ImageFolder not found.")

    if not datasets_list:
        print("No datasets loaded. Check path.")
        csv_file.close()
        return

    full_dataset = ConcatDataset(datasets_list)
    print(f"=> Total images for analysis: {len(full_dataset)}")

    # --- 3. ANALYSIS LOOP ---
    learning_rates = [1e-2, 1e-3, 1e-4]
    
    print("\n" + "="*50)
    print(" STARTING ANALYSIS: CASUAL vs SORTED ")
    print("="*50)

    for lr in learning_rates:
        torch.cuda.empty_cache()
        print(f"\n>>> Analyzing Learning Rate: {lr}")
        
        # --- MODE A: CASUAL ---
        print("Mode: CASUAL (Shuffling data...)")
        loader_casual = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True, 
                                   num_workers=4, pin_memory=True)
        # Passiamo csv_writer alla funzione
        run_continual_tpt(loader_casual, model, lr, args, "Casual", csv_writer)

        torch.cuda.empty_cache()

        # --- MODE B: SORTED ---
        print("Mode: SORTED (Sequential data...)")
        loader_sorted = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False, 
                                   num_workers=4, pin_memory=True)
        run_continual_tpt(loader_sorted, model, lr, args, "Sorted", csv_writer)
        
    csv_file.close()
    print(f"\nAnalysis Complete. Results saved to {args.outfile}")

def run_continual_tpt(loader, base_model, lr, args, mode_name, csv_writer):
    # Reset del modello
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
        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        batch_size_curr = images.size(0)

        # --- TTA Step ---
        for _ in range(args.steps):
            with torch.amp.autocast('cuda'):
                output = model(images)
                loss = avg_entropy(output)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # --- Inference Step ---
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                output = model(images)
        
        # Calcolo statistiche
        acc1, _ = accuracy(output, target, topk=(1, 5))
        batch_acc = acc1[0].item() # Accuratezza % di questo batch
        
        correct_cum += batch_acc * batch_size_curr / 100.0
        total_cum += batch_size_curr
        running_acc = (correct_cum / total_cum) * 100.0 # Accuratezza media totale fino ad ora
        
        # --- SCRITTURA CSV ---
        # Scriviamo una riga per ogni batch processato
        # Columns: [LR, Mode, Batch_Idx, Samples_Seen, Batch_Acc, Running_Acc]
        csv_writer.writerow([
            lr, 
            mode_name, 
            i, 
            total_cum, 
            f"{batch_acc:.2f}", 
            f"{running_acc:.2f}"
        ])
        
        if i % 10 == 0:
            print(f"[{mode_name} | LR={lr}] Step {i}/{len(loader)} | Batch Acc: {batch_acc:.2f}% | Run Avg: {running_acc:.2f}%")

    return running_acc

if __name__ == '__main__':
    main()