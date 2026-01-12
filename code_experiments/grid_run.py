import os
import subprocess
import time
import re
import csv

# --- CONFIGURAZIONE ---
DATA_PATH = "../datasets/imagenet-adversarial"
DATASET = "A"        
ARCH = "RN50"        
OUTPUT_DIR = "mega_run_logs"
SUMMARY_FILE = "final_summary.csv"

# --- IPERPARAMETRI (GRID SEARCH) ---
learning_rates = ["0.0025","0.005","0.01"] 
confidence_values = ["0.05","0.1","0.5"]
tta_steps_values = ["1","2"]
n_views_values = ["8","16","32","64"]

def parse_accuracy_from_log(log_path):
    """
    Legge il file di log e cerca l'accuratezza finale Top-1.
    Logica: Cerca la stringa esatta 'Acc. on testset [X]: @1' e cattura tutto fino allo slash '/'.
    Esempio target: "=> Acc. on testset [A]: @1 25.75/ @5 55.625"
    """
    try:
        if not os.path.exists(log_path):
            return 0.0
            
        with open(log_path, "r") as f:
            content = f.read()
            
        # REGEX SPIEGAZIONE:
        # Acc\. on testset   -> Cerca letteralmente "Acc. on testset"
        # \[.*?\]            -> Cerca le parentesi quadre con qualsiasi lettera dentro (es. [A] o [R])
        # : @1\s* -> Cerca ": @1" seguito da spazi opzionali
        # (.*?)              -> GRUPPO DI CATTURA: Prendi qualsiasi carattere (il numero)...
        # /                  -> ...fino a quando non trovi lo slash "/"
        match = re.findall(r"Acc\. on testset \[.*?\]: @1\s*(.*?)/", content)
        
        if match:
            # Prende l'ultimo risultato e rimuove eventuali spazi bianchi extra
            return float(match[-1].strip())
        else:
            return 0.0
    except Exception as e:
        # Se il log è corrotto o incompleto
        # print(f"Errore parsing log {log_path}: {e}") # Decommenta per debug
        return 0.0

def run_experiment(lr, conf, step, views, exp_index, total_exps):
    exp_name = f"{DATASET}_{ARCH}_b{views}_step{step}_conf{conf}_lr{lr}"
    log_file_path = os.path.join(OUTPUT_DIR, f"{exp_name}.txt")
    
    print(f"\n[{exp_index}/{total_exps}] Running: {exp_name}")

    cmd = [
        "python3", "tpt_classification.py",
        "--test_sets", DATASET,
        "-a", ARCH,
        "--gpu", "0",
        "--ctx_init", "a_photo_of_a",
        "--tpt",
        "-b", views,
        "--tta_steps", step,
        "--selection_p", conf,
        "--lr", lr,
        DATA_PATH
    ]
    
    # Esecuzione
    start_time = time.time()
    with open(log_file_path, "w") as log_file:
        subprocess.run(cmd, stdout=log_file, stderr=log_file)
    elapsed = time.time() - start_time
    
    # Parsing del risultato
    accuracy = parse_accuracy_from_log(log_file_path)
    print(f"   -> Done in {elapsed:.1f}s | Accuracy trovata: {accuracy}%")
    
    # Ritorna un dizionario con i dati di questa run
    return {
        "Exp_Name": exp_name,
        "Views": int(views),
        "Steps": int(step),
        "Confidence": float(conf),
        "LR": float(lr),
        "Accuracy": accuracy,
        "Time_Sec": round(elapsed, 1)
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_data = [] # Qui accumuliamo i risultati
    
    total_experiments = len(learning_rates) * len(confidence_values) * len(tta_steps_values) * len(n_views_values)
    current_count = 1
    
    print(f"STARTING GRID SEARCH: {total_experiments} esperimenti totali.")
    
    try:
        for lr in learning_rates:
            for conf in confidence_values:
                for step in tta_steps_values:
                    for views in n_views_values:
                        
                        # Esegui e ottieni i dati
                        result = run_experiment(lr, conf, step, views, current_count, total_experiments)
                        results_data.append(result)
                        current_count += 1
                        
    except KeyboardInterrupt:
        print("\nInterrotto dall'utente! Generazione tabella parziale...")

    # --- GENERAZIONE REPORT FINALE ---
    
    # 2. Salva su CSV
    keys = ["Exp_Name", "Accuracy", "Views", "Steps", "Confidence", "LR", "Time_Sec"]
    with open(SUMMARY_FILE, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results_data)
        
    print("\n" + "="*80)
    print(f"RISULTATI FINALI (Top 10) - Salvati in {SUMMARY_FILE}")
    print("="*80)
    
    # 3. Stampa tabella carina a terminale (Header)
    header = f"{'Accuracy':<10} | {'Views':<5} | {'Steps':<5} | {'Conf':<5} | {'LR':<8} | {'Time':<6}"
    print(header)
    print("-" * len(header))
    
    # Stampa righe
    for res in results_data[:15]: # Mostra solo i primi 15 a schermo
        print(f"{res['Accuracy']:<10} | {res['Views']:<5} | {res['Steps']:<5} | {res['Confidence']:<5} | {res['LR']:<8} | {res['Time_Sec']:<6}")
        
    print("="*80)

if __name__ == "__main__":
    main()