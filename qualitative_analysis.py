import argparse
import time
import os
from copy import deepcopy
from PIL import Image, ImageDraw, ImageFont # Aggiunto per disegnare sulle immagini
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.custom_clip import get_coop
from clip.cocoop import get_cocoop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset, build_subdataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask

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

def test_time_tuning(model, inputs, optimizer, scaler, args):
    if args.cocoop:
        image_feature, pgen_ctx = inputs
        pgen_ctx.requires_grad = True
        optimizer = torch.optim.AdamW([pgen_ctx], args.lr)
    
    selected_idx = None
    for j in range(args.tta_steps):
        with torch.cuda.amp.autocast():
            if args.cocoop:
                output = model((image_feature, pgen_ctx))
            else:
                output = model(inputs) 

            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples(output, args.selection_p)

            loss = avg_entropy(output)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    if args.cocoop:
        return pgen_ctx
    return

# --- NUOVE FUNZIONI DI SUPPORTO PER L'ANALISI QUALITATIVA ---
def unnormalize_and_save(tensor, filename, text_info):
    """Salva l'immagine con le etichette scritte sopra per il report."""
    # De-normalizzazione ImageNet
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(tensor.device)
    img = tensor * std + mean
    img = torch.clamp(img, 0, 1)
    
    # Conversione a PIL
    img_pil = transforms.ToPILImage()(img.cpu())
    
    # Aggiungi spazio bianco per il testo
    from PIL import ImageOps
    # Espandi bordo superiore di 70 pixel
    img_with_border = ImageOps.expand(img_pil, border=(0, 70, 0, 0), fill='white')
    draw = ImageDraw.Draw(img_with_border)
    
    # Scrivi il testo (GT, CLIP, TPT)
    # Usa colori diversi: Rosso per errore, Verde per corretto
    gt_text, clip_text, tpt_text = text_info
    
    draw.text((10, 5),  gt_text, fill="black")
    draw.text((10, 25), clip_text, fill="green" if "CORRECT" in clip_text else "red")
    draw.text((10, 45), tpt_text, fill="green" if "CORRECT" in tpt_text else "red")
    
    img_with_border.save(filename)
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning Analysis')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A', help='test dataset (A/R/V/K/I)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='Batch size MUST be 1 for analysis')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', default=True, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_dir', default='./qualitative_report', type=str, help='Output folder for images')

    args = parser.parse_args()
    set_random_seed(args.seed)
    
    # Forza batch size a 1 per analisi singola
    args.batch_size = 1
    
    main_worker(args.gpu, args)

def main_worker(gpu, args):
    args.gpu = gpu
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    # Load Classes
    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    else:
        # Gestione specifica per ImageNet-A/R etc
        classnames_all = imagenet_classes
        if args.test_sets == 'A':
            classnames = [classnames_all[i] for i in imagenet_a_mask]
        elif args.test_sets == 'R':
            classnames = []
            for i, m in enumerate(imagenet_r_mask):
                if m: classnames.append(classnames_all[i])
        else:
            classnames = classnames_all

    # Create Model
    if args.cocoop:
        model = get_cocoop(args.arch, args.test_sets, 'cpu', args.n_ctx)
        assert args.load is not None
        load_model_weight(args.load, model, 'cpu', args)
        model_state = deepcopy(model.state_dict())
    else:
        model = get_coop(args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init)
        if args.load is not None:
            pretrained_ctx = torch.load(args.load)['state_dict']['ctx']
            with torch.no_grad():
                model.prompt_learner[0].ctx.copy_(pretrained_ctx)
                model.prompt_learner[0].ctx_init_state = pretrained_ctx
        model_state = None

    for name, param in model.named_parameters():
        if not args.cocoop:
            if "prompt_learner" not in name: param.requires_grad_(False)
        else:
            if "text_encoder" not in name: param.requires_grad_(False)
    
    model = model.cuda(args.gpu)

    # Optimizer
    if args.cocoop:
        optimizer = None
        optim_state = None
    else:
        trainable_param = model.prompt_learner.parameters()
        optimizer = torch.optim.AdamW(trainable_param, args.lr)
        optim_state = deepcopy(optimizer.state_dict())

    scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    # Dataset Setup
    # TPT richiede Augmentations (AugMixAugmenter)
    base_transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=BICUBIC),
        transforms.CenterCrop(args.resolution)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    
    # N_views=63 per avere abbastanza variazioni per TPT
    data_transform = AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=True)

    print(f"evaluating: {args.test_sets}")
    
    # Reset classnames in model
    if args.cocoop:
        model.prompt_generator.reset_classnames(classnames, args.arch)
    else:
        model.reset_classnames(classnames, args.arch)

    # Dataset build
    val_dataset = build_dataset(args.test_sets, data_transform, args.data, mode=args.dataset_mode)
    print("number of test samples: {}".format(len(val_dataset)))
    
    val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=1, shuffle=True, # Shuffle per trovare casi diversi
                num_workers=args.workers, pin_memory=True)
            
    # Avvia Analisi
    test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, classnames)


def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, classnames):
    # Setup salvataggio
    os.makedirs(args.out_dir, exist_ok=True)
    found_success = 0
    found_failure = 0
    target_count = 5
    print(f"--- INIZIO ANALISI QUALITATIVA ---\nCerco {target_count} Successi e {target_count} Fallimenti...")

    model.eval()
    
    for i, (images, target) in enumerate(val_loader):
        # Stop se abbiamo finito
        if found_success >= target_count and found_failure >= target_count:
            print(f"Analisi Completata! Controlla la cartella {args.out_dir}")
            return

        target = target.cuda(args.gpu, non_blocking=True)
        
        # Gestione immagini (AugMixAugmenter ritorna una lista)
        # images[0] è l'immagine pulita (center crop)
        # images[1:] sono le versioni augmentate
        if isinstance(images, list):
            clean_image = images[0].cuda(args.gpu, non_blocking=True)
            # TPT inputs (tutte le viste concatenate)
            tpt_inputs = [img.cuda(args.gpu, non_blocking=True) for img in images]
            tpt_inputs = torch.cat(tpt_inputs, dim=0) # [64, C, H, W]
        else:
            # Fallback se non c'è augmix
            clean_image = images.cuda(args.gpu, non_blocking=True)
            tpt_inputs = clean_image

        # -----------------------------------------------------
        # 1. BASELINE (CLIP Zero-shot) - Prima del Tuning
        # -----------------------------------------------------
        # Reset model to default prompt
        if not args.cocoop:
            with torch.no_grad(): model.reset()
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output_base = model(clean_image) # Inferenza su immagine pulita
            probs_base = torch.softmax(output_base, dim=1)
            conf_base, pred_base = probs_base.max(1)
            
        is_base_correct = (pred_base == target).item()

        # -----------------------------------------------------
        # 2. TPT (Test-Time Prompt Tuning)
        # -----------------------------------------------------
        # Reset again before tuning
        if not args.cocoop:
            with torch.no_grad(): model.reset()
            optimizer.load_state_dict(optim_state) # Reset optimizer
            
            # Tuning usando le viste augmentate (tpt_inputs)
            test_time_tuning(model, tpt_inputs, optimizer, scaler, args)
        
        # Inferenza finale post-tuning su immagine pulita
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output_tpt = model(clean_image)
            probs_tpt = torch.softmax(output_tpt, dim=1)
            conf_tpt, pred_tpt = probs_tpt.max(1)

        is_tpt_correct = (pred_tpt == target).item()

        # -----------------------------------------------------
        # 3. CONFRONTO E SALVATAGGIO
        # -----------------------------------------------------
        gt_name = classnames[target.item()]
        base_name = classnames[pred_base.item()]
        tpt_name = classnames[pred_tpt.item()]
        
        # Stringhe info
        info_gt = f"GT: {gt_name}"
        info_clip = f"CLIP: {base_name} ({conf_base.item():.1%}) [{'CORRECT' if is_base_correct else 'WRONG'}]"
        info_tpt = f"TPT: {tpt_name} ({conf_tpt.item():.1%}) [{'CORRECT' if is_tpt_correct else 'WRONG'}]"

        # Caso A: Successo (CLIP Sbaglia -> TPT Corregge)
        if not is_base_correct and is_tpt_correct and found_success < target_count:
            print(f"[SUCCESS] {gt_name}: CLIP({base_name}) -> TPT({tpt_name})")
            fname = os.path.join(args.out_dir, f"success_{found_success}_{gt_name}.png")
            unnormalize_and_save(clean_image[0], fname, [info_gt, info_clip, info_tpt])
            found_success += 1

        # Caso B: Fallimento / Over-correction (CLIP Giusto -> TPT Sbaglia)
        elif is_base_correct and not is_tpt_correct and found_failure < target_count:
            print(f"[FAILURE] {gt_name}: CLIP({base_name}) -> TPT({tpt_name})")
            fname = os.path.join(args.out_dir, f"fail_{found_failure}_{gt_name}.png")
            unnormalize_and_save(clean_image[0], fname, [info_gt, info_clip, info_tpt])
            found_failure += 1

        # Log progress ogni tanto
        if (i+1) % 50 == 0:
            print(f"Processed {i+1} images... Found Success: {found_success}/3, Fail: {found_failure}/3")

if __name__ == '__main__':
    main()