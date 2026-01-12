import os
import subprocess

# CONFIGURAZIONE
DATA_PATH = "../datasets/imagenet-adversarial" 
DATASET = "A" 
ARCH = "RN50"

# Definisci gli esperimenti
# CORREZIONI APPLICATE:
# --n_views   -> -b (Batch size = numero di viste in TPT)
# --steps     -> --tta_steps
# --output    -> Rimossa (gestita tramite reindirizzamento file)
experiments = {
    # 1. Variare le Viste (Augmentations) -> Parametro -b
    "views_8": {"-b": "8", "--tta_steps": "1", "--selection_p": "0.1", "--lr": "0.005"},
    "views_16": {"-b": "16", "--tta_steps": "1", "--selection_p": "0.1", "--lr": "0.005"},
    "views_32": {"-b": "32", "--tta_steps": "1", "--selection_p": "0.1", "--lr": "0.005"},
    "views_64": {"-b": "64", "--tta_steps": "1", "--selection_p": "0.1", "--lr": "0.005"},
    
    # 2. Variare la Selection Confidence
}

""" "conf_0.05": {"-b": "64", "--tta_steps": "1", "--selection_p": "0.05", "--lr": "0.005"},
    "conf_0.1": {"-b": "64", "--tta_steps": "1", "--selection_p": "0.1", "--lr": "0.005"},
    "conf_0.2": {"-b": "64", "--tta_steps": "1", "--selection_p": "0.2", "--lr": "0.005"},
    "conf_0.5":  {"-b": "64", "--tta_steps": "1", "--selection_p": "0.5",  "--lr": "0.005"},
    
    # 3. Variare Steps di ottimizzazione -> Parametro --tta_steps
    "steps_2":   {"-b": "64", "--tta_steps": "2", "--selection_p": "0.1", "--lr": "0.005"},
    
    # 4. Variare Learning Rate
    "lr_1e-2":   {"-b": "64", "--tta_steps": "1", "--selection_p": "0.1", "--lr": "1e-2"}, """

def run_cmd(name, args):
    print(f"\n=== Running Experiment: {name} ===")
    
    # Costruzione comando base
    cmd = [
        "python3", "tpt_classification.py",
        "--test_sets", DATASET,
        "-a", ARCH,
        "--gpu", "0",
        "--ctx_init", "a_photo_of_a",
        "--tpt"
    ]
    
    # Aggiungi gli argomenti specifici dell'esperimento
    for k, v in args.items():
        cmd.append(k)
        cmd.append(v)
    
    # Aggiungi il DATA_PATH come argomento POSIZIONALE (alla fine, senza flag --data)
    cmd.append(DATA_PATH)
    
    # Gestione Output File
    output_dir = f"results/analysis_logs"
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, f"{name}.txt")
    
    print("Command:", " ".join(cmd))
    print(f"Logging to: {log_file_path}")
    
    # Esegui e reindirizza stdout e stderr sul file di log
    with open(log_file_path, "w") as log_file:
        subprocess.run(cmd, stdout=log_file, stderr=log_file)

if __name__ == "__main__":
    for exp_name, exp_args in experiments.items():
        run_cmd(exp_name, exp_args)
