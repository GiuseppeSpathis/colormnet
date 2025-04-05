import os
import shutil
import subprocess
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import time # Per dare tempo al filesystem, se necessario

# --- Configurazione dei Path ---
# Modifica questi path secondo la tua struttura esatta, se necessario
# MA input/result sono fissi su 'blackswan' come richiesto
SCRATCH = "/scratch.hpc/giuseppe.spathis/"
BASE_DIR = "/scratch.hpc/giuseppe.spathis/colormnet"
TEST_SET_DIR = os.path.join(BASE_DIR, "test_set")
INPUT_DIR_BASE = os.path.join(BASE_DIR, "input")
RESULT_DIR_BASE = os.path.join(BASE_DIR, "result")

# --- USARE ESATTAMENTE 'blackswan' COME RICHIESTO ---
INPUT_SUBDIR_NAME = "blackswan"
RESULT_SUBDIR_NAME = "blackswan"
# ------------------------------------------------------

INPUT_DIR = os.path.join(INPUT_DIR_BASE, INPUT_SUBDIR_NAME)
RESULT_DIR = os.path.join(RESULT_DIR_BASE, RESULT_SUBDIR_NAME)
INFERENCE_SCRIPT = os.path.join(BASE_DIR, "test.py")
OUTPUT_CSV = os.path.join(SCRATCH, "ssim_results.csv")
CUDA_DEVICE = "0" # Modifica se necessario

# --- Funzioni Helper (rimangono invariate rispetto a prima) ---

def clean_directory(dir_path):
    """Rimuove tutti i file e le sottocartelle in una directory."""
    print(f"Pulizia directory: {dir_path}")
    if not os.path.exists(dir_path):
        print("Directory non esistente, la creo.")
        os.makedirs(dir_path)
        return
    try:
        # Rimuove file e sottodirectory
        for item_name in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item_name)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        print("Pulizia completata.")
    except Exception as e:
        print(f"Errore durante la pulizia di {dir_path}: {e}")

def prepare_input_images(src_folder, dest_folder):
    """Copia le immagini da src a dest convertendole in scala di grigi."""
    print(f"Preparazione input da {src_folder} a {dest_folder} (Grayscale)")
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    image_files = sorted([f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    if not image_files:
        print(f"Attenzione: Nessuna immagine trovata in {src_folder}")
        return False

    for filename in image_files:
        src_path = os.path.join(src_folder, filename)
        dest_path = os.path.join(dest_folder, filename)
        try:
            with Image.open(src_path) as img:
                # Verifica se l'immagine è già in scala di grigi ('L') o con canale alpha ('LA')
                if img.mode == 'L' or img.mode == 'LA':
                     grayscale_img = img.convert('L') # Assicura che sia solo 'L'
                     print(f"Info: Immagine {filename} già grayscale, salvataggio in {dest_path}")
                     grayscale_img.save(dest_path)
                else:
                     # Converti a colori prima (se non lo è già) per gestire palette, poi a 'L'
                     # Questo evita potenziali problemi con palette indexate
                     rgb_img = img.convert('RGB')
                     grayscale_img = rgb_img.convert('L')
                     grayscale_img.save(dest_path)

        except Exception as e:
            print(f"Errore durante la preparazione dell'immagine {filename}: {e}")
            return False # Interrompi se c'è un errore
    print(f"Preparazione di {len(image_files)} immagini completata.")
    return True

def run_inference(script_path, cuda_device):
    """Esegue lo script di inferenza."""
    print(f"Avvio inferenza: {script_path} su CUDA:{cuda_device}")
    print(f"Input directory attesa da test.py: {INPUT_DIR}")
    print(f"Output directory attesa da test.py: {RESULT_DIR}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_device
    # Assumiamo che lo script test.py sia eseguito dalla sua directory base
    # o che i path interni allo script siano relativi a BASE_DIR.
    # Se test.py deve essere eseguito da una directory specifica,
    # potresti dover aggiungere il parametro `cwd=BASE_DIR` a subprocess.run
    command = ["python", script_path]
    try:
        # Usiamo subprocess.run per attendere il completamento
        result = subprocess.run(command, env=env, check=True, capture_output=True, text=True, cwd=BASE_DIR) # Aggiunto CWD se necessario
        print("Inferenza completata con successo.")
        # print("Output script:\n", result.stdout) # Decommenta per debug
        if result.stderr: # Stampa eventuali errori standard anche se il codice è 0
             print("Stderr script:\n", result.stderr)
        # Attendi un istante per assicurarti che i file siano scritti sul filesystem
        time.sleep(2)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Errore durante l'esecuzione dell'inferenza:")
        print(f"Comando: {' '.join(e.cmd)}")
        print(f"Working Directory: {BASE_DIR}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"Errore: Script di inferenza non trovato: {script_path}")
        return False

def calculate_ssim_for_folder(original_color_folder, result_color_folder):
    """Calcola SSIM tra immagini originali a colori e risultati colorizzati."""
    print(f"Calcolo SSIM tra {original_color_folder} e {result_color_folder}")
    ssim_values = []

    original_files = sorted([f for f in os.listdir(original_color_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    # Assicurati che la cartella risultato esista prima di leggerla
    if not os.path.exists(result_color_folder):
        print(f"Errore: La cartella dei risultati {result_color_folder} non esiste dopo l'inferenza.")
        return None # Non possiamo calcolare SSIM
    result_files = sorted([f for f in os.listdir(result_color_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

    # Controllo corrispondenza file (basato sui nomi ordinati)
    if len(original_files) != len(result_files):
        print(f"Attenzione: Numero di file non corrispondente tra {original_color_folder} ({len(original_files)}) e {result_color_folder} ({len(result_files)})")
        # Potresti voler gestire questo caso in modo più robusto, qui procediamo con i file che matchano per nome

    print(f"Trovati {len(original_files)} file originali e {len(result_files)} file risultato.")

    matched_files = 0
    for filename in original_files:
        original_path = os.path.join(original_color_folder, filename)
        result_path = os.path.join(result_color_folder, filename) # Assumiamo che il nome file sia lo stesso

        if not os.path.exists(result_path):
            print(f"Attenzione: File risultato corrispondente non trovato per {filename} in {result_color_folder}. Salto.")
            continue

        try:
            # Apre l'originale e converte in RGB per sicurezza (gestisce palette etc.)
            img_orig_pil = Image.open(original_path).convert('RGB')
            img_orig = np.array(img_orig_pil)

            # Apre il risultato e converte in RGB per sicurezza
            img_result_pil = Image.open(result_path).convert('RGB')
            img_result = np.array(img_result_pil)


            # Assicurati che le immagini abbiano le stesse dimensioni
            if img_orig.shape != img_result.shape:
                 print(f"Attenzione: Dimensioni non corrispondenti per {filename} (Originale: {img_orig.shape}, Risultato: {img_result.shape}). Salto SSIM.")
                 continue

            # Calcola SSIM. data_range è importante. channel_axis=-1 per immagini a colori RGB.
            # Se una delle due immagini fosse grayscale per errore, ssim darebbe errore o risultati strani.
            try:
                # data_range=255 assume che i valori dei pixel siano in [0, 255]
                s = ssim(img_orig, img_result, data_range=255, channel_axis=-1)
                ssim_values.append(s)
                matched_files += 1
            except ValueError as ve:
                 # Potrebbe accadere se win_size è più grande dell'immagine, o altri problemi
                 # es. se un'immagine fosse letta come grayscale nonostante la conversione
                 print(f"Errore nel calcolo SSIM per {filename}: {ve}. Salto.")
                 print(f"  Shape originale: {img_orig.shape}, dtype: {img_orig.dtype}")
                 print(f"  Shape risultato: {img_result.shape}, dtype: {img_result.dtype}")
                 continue

        except Exception as e:
            print(f"Errore nell'apertura o processamento dell'immagine {filename}: {e}")

    if not ssim_values:
        print("Attenzione: Nessun valore SSIM calcolato per questa cartella.")
        return None

    average_ssim = np.mean(ssim_values)
    print(f"Calcolata SSIM su {matched_files} coppie di file. Media SSIM: {average_ssim:.4f}")
    return average_ssim

# --- Main Execution Logic (invariata) ---

def main(is_pre_finetuning):
    print("Avvio pipeline di calcolo SSIM...")

    subfolders = sorted([f for f in os.listdir(TEST_SET_DIR) if os.path.isdir(os.path.join(TEST_SET_DIR, f))])
    if not subfolders:
        print(f"Errore: Nessuna sottocartella trovata in {TEST_SET_DIR}")
        return

    print(f"Trovate le seguenti sottocartelle nel test set: {', '.join(subfolders)}")

    results = []
    all_avg_ssims = []

    for folder_name in subfolders:
        print(f"\n--- Elaborazione cartella: {folder_name} ---")
        original_frames_dir = os.path.join(TEST_SET_DIR, folder_name)

        # 1. Pulisci le directory di input e output ('blackswan') precedenti
        clean_directory(INPUT_DIR)
        clean_directory(RESULT_DIR) # Pulisce anche i risultati precedenti

        # 2. Prepara le immagini di input (grayscale) in input/blackswan
        if not prepare_input_images(original_frames_dir, INPUT_DIR):
            print(f"Errore nella preparazione input per {folder_name}. Salto questa cartella.")
            results.append({'Folder': folder_name, 'Average_SSIM': 'Error_Input', 'Phase': 'Pre-Finetuning' if is_pre_finetuning else 'Post-Finetuning'})
            continue # Vai alla prossima cartella

        # 3. Esegui l'inferenza (che leggerà da input/blackswan e scriverà in result/blackswan)
        if not run_inference(INFERENCE_SCRIPT, CUDA_DEVICE):
            print(f"Errore durante l'inferenza per {folder_name}. Salto questa cartella.")
            results.append({'Folder': folder_name, 'Average_SSIM': 'Error_Inference', 'Phase': 'Pre-Finetuning' if is_pre_finetuning else 'Post-Finetuning'})
            continue # Vai alla prossima cartella

        # 4. Calcola SSIM tra originali (a colori da test_set/folder_name) e risultati (colorizzati da result/blackswan)
        avg_ssim = calculate_ssim_for_folder(original_frames_dir, RESULT_DIR)

        phase_label = 'Pre-Finetuning' if is_pre_finetuning else 'Post-Finetuning'

        if avg_ssim is not None:
            results.append({'Folder': folder_name, 'Average_SSIM': f"{avg_ssim:.4f}", 'Phase': phase_label})
            all_avg_ssims.append(avg_ssim)
        else:
            print(f"SSIM non calcolabile per {folder_name}.")
            # Distinguiamo l'errore se possibile (es. mancata creazione result dir)
            if not os.path.exists(RESULT_DIR):
                 ssim_val = 'Error_ResultDir_Missing'
            else:
                 ssim_val = 'N/A_Calculation'
            results.append({'Folder': folder_name, 'Average_SSIM': ssim_val, 'Phase': phase_label})

    # 5. Calcola SSIM totale (media delle medie) e salva in CSV
    if all_avg_ssims:
        total_average_ssim = np.mean(all_avg_ssims)
        print(f"\n--- Risultato Complessivo ---")
        print(f"SSIM Media Totale: {total_average_ssim:.4f}")
        # Aggiungi riga per la media totale
        results.append({'Folder': 'Overall_Average', 'Average_SSIM': f"{total_average_ssim:.4f}", 'Phase': phase_label})
    else:
        print("\nNessun valore SSIM medio calcolato, impossibile calcolare la media totale.")
        # Controlla se ci sono stati errori precedenti
        if any('Error' in str(r['Average_SSIM']) for r in results):
             avg_label = 'Error_In_Folders'
        elif any('N/A' in str(r['Average_SSIM']) for r in results):
             avg_label = 'N/A_Calculation'
        else:
             avg_label = 'No_Valid_Data'

        results.append({'Folder': 'Overall_Average', 'Average_SSIM': avg_label, 'Phase': phase_label})

    # Scrittura CSV
    try:
        df = pd.DataFrame(results)
        # Definisci l'ordine delle colonne
        if is_pre_finetuning:
            column_order = ['Folder', 'Average_SSIM', 'Phase']
        else:
            # Se non è pre-finetuning, potresti voler omettere la colonna Phase
            # o lasciarla come 'Post-Finetuning'. La lascio per coerenza.
            column_order = ['Folder', 'Average_SSIM', 'Phase']
            # Se vuoi ometterla in questo caso:
            # df = df[['Folder', 'Average_SSIM']]
            # column_order = ['Folder', 'Average_SSIM']


        df = df[column_order] # Riordina colonne
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nRisultati salvati in: {OUTPUT_CSV}")
    except Exception as e:
        print(f"Errore durante il salvataggio del file CSV: {e}")

    print("\nPipeline completata.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline per calcolare SSIM per ColorMNet usando input/result/blackswan.")
    parser.add_argument(
        '--pre_finetuning',
        action='store_true', # Se presente, imposta a True
        help="Indica se l'esecuzione è pre-finetuning (aggiunge colonna 'Phase' al CSV)"
    )
    args = parser.parse_args()

    # Controlla se le directory base esistono, altrimenti esci
    if not os.path.isdir(BASE_DIR):
         print(f"Errore: Directory base non trovata: {BASE_DIR}")
         exit(1)
    if not os.path.isdir(TEST_SET_DIR):
         print(f"Errore: Directory test set non trovata: {TEST_SET_DIR}")
         exit(1)
    if not os.path.isfile(INFERENCE_SCRIPT):
         print(f"Errore: Script di inferenza non trovato: {INFERENCE_SCRIPT}")
         exit(1)

    # Crea le directory input/blackswan e result/blackswan se non esistono
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)


    main(args.pre_finetuning)
