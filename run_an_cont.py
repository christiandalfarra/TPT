import subprocess
import sys
import os
import time

# --- CONFIGURAZIONE ---

# Percorso al dataset ImageNet-A (MODIFICA QUESTO)
DATA_PATH = "../datasets/imagenet-adversarial" 

# Lista degli script da testare
scripts = [
    {
        "name": "Casual Order",
        "file": "tpt_continuous_casual_order.py",
        "color": "blue" # Per eventuali plot futuri
    },
    {
        "name": "Ordinate Order",
        "file": "tpt_continuous_ordinate.py",
        "color": "orange"
    }
]

# Iperparametri da testare
learning_rates = [0.00001, 0.0001, 0.005]  # 1e-5, 5e-3
reset_intervals = [0, 10]         # 0 = mai (o continuo puro), 10 = ogni 10 img

# Parametri fissi comuni
ARCH = "RN50"
GPU_ID = "0"

# Nome del flag nel tuo codice per il reset (VERIFICA QUESTO NOME NEI TUOI SCRIPT)
RESET_FLAG = "--reset_interval" 

# --- ESECUZIONE ---

def run_experiment():
    total_experiments = len(scripts) * len(learning_rates) * len(reset_intervals)
    current_count = 0

    print(f"=== INIZIO ANALISI COMPLETA: {total_experiments} Esperimenti totali ===\n")

    for script_info in scripts:
        script_file = script_info["file"]
        mode_name = script_info["name"]

        for lr in learning_rates:
            for reset_val in reset_intervals:
                current_count += 1
                
                print(f"[{current_count}/{total_experiments}] Running: {mode_name}")
                print(f"    -> LR: {lr}")
                print(f"    -> Reset: {reset_val}")
                print(f"    -> Script: {script_file}")

                # Costruzione del comando
                # Esempio: python tpt_continuous_casual_order.py /data/path --arch ViT-B/16 --tpt --lr 0.005 --reset_interval 10
                cmd = [
                    sys.executable,   # Percorso all'interprete python corrente
                    script_file,      # Il file .py
                    DATA_PATH,        # Argomento posizionale 'data'
                    "--test_sets", "A",  # Dataset ImageNet-A
                    "--ctx_init", "a_photo_of_a", # Contestualizzazione
                    "-b", "64",        # Batch size (numero di viste)
                    "--selection_p", "0.1", # Probabilità di selezione
                    "--continuous",    # Abilita analisi continua
                    "--arch", ARCH,
                    "--tpt",          # Abilita TPT
                    "--gpu", GPU_ID,
                    "--lr", str(lr),
                    RESET_FLAG, str(reset_val)
                ]

                # Esecuzione
                try:
                    start_time = time.time()
                    
                    # subprocess.run attende che lo script finisca prima di passare al prossimo
                    result = subprocess.run(cmd, check=True, text=True)
                    
                    elapsed = time.time() - start_time
                    print(f"    [COMPLETATO] Tempo impiegato: {elapsed:.2f}s\n")
                    
                except subprocess.CalledProcessError as e:
                    print(f"    [ERRORE] Lo script {script_file} è fallito con LR={lr} e Reset={reset_val}.")
                    print(f"    Dettaglio errore: {e}\n")
                    # Opzionale: break o continue a seconda se vuoi fermare tutto o provare il prossimo

    print("=== ANALISI TERMINATA ===")
    print("Controlla i file CSV generati nella cartella corrente o nella cartella di output specificata dai tuoi script.")

if __name__ == "__main__":
    # Verifica che i file esistano prima di partire
    if not os.path.exists(scripts[0]["file"]) or not os.path.exists(scripts[1]["file"]):
        print("ERRORE: Non trovo i file degli script (.py) nella cartella corrente.")
    else:
        run_experiment()