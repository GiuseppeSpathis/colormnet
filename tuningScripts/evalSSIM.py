import os
import shutil
import subprocess
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import time  # Per dare tempo al filesystem, se necessario
import glob # Per trovare i file da spostare

# Importa metriche già presenti
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Import opzionali per LPIPS e FID (gestiti con try-except)
try:
    import torch
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Attenzione: Librerie 'torch' o 'lpips' non trovate. Calcolo LPIPS non disponibile.")

try:
    from pytorch_fid.fid_score import calculate_fid_given_paths
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("Attenzione: Libreria 'pytorch-fid' non trovata. Calcolo FID non disponibile.")


# --- Configurazione dei Path ---
SCRATCH = "/scratch.hpc/giuseppe.spathis/"
BASE_DIR = "/scratch.hpc/giuseppe.spathis/colormnet"
TEST_SET_DIR = os.path.join(BASE_DIR, "test_set") # Cartella con sottocartelle (001, 002...) originali a colori

# Directory usate NATIVAMENTE da test.py (come nello script originale)
INPUT_DIR_BASE = os.path.join(BASE_DIR, "input")
RESULT_DIR_BASE = os.path.join(BASE_DIR, "result")
INPUT_SUBDIR_NAME = "blackswan"  # Ripristinato nome originale
RESULT_SUBDIR_NAME = "blackswan" # Ripristinato nome originale
INPUT_DIR = os.path.join(INPUT_DIR_BASE, INPUT_SUBDIR_NAME) # Path input grayscale usato da test.py
RESULT_DIR = os.path.join(RESULT_DIR_BASE, RESULT_SUBDIR_NAME) # Path output colorizzato scritto da test.py

# NUOVE DIRECTORY DI OUTPUT FINALE (dove spostare i risultati DOPO l'inferenza)
RESULTS_FINAL_BASE_DIR = os.path.join(SCRATCH, "results") # Base per i risultati finali
RESULTS_PRE_TUNING_BASE = os.path.join(RESULTS_FINAL_BASE_DIR, "preTuning")
RESULTS_POST_TUNING_BASE = os.path.join(RESULTS_FINAL_BASE_DIR, "postTuning")

INFERENCE_SCRIPT = os.path.join(BASE_DIR, "test.py")
OUTPUT_CSV = os.path.join(SCRATCH, "ssim_psnr_fid_lpips_results.csv") # Nome file CSV
CUDA_DEVICE = "0"

# --- Path del checkpoint per post-finetuning ---
POST_FINETUNING_CHECKPOINT = "/scratch.hpc/giuseppe.spathis/colormnet/savingTuning/saves/tuning_s2/tuning_s2_4000.pth"

# --- Funzioni Helper ---

# Funzione clean_directory come nell'originale
def clean_directory(dir_path):
    """Rimuove tutti i file e le sottocartelle in una directory, creandola se non esiste."""
    print(f"Pulizia directory: {dir_path}")
    if not os.path.exists(dir_path):
        print("Directory non esistente, la creo.")
        os.makedirs(dir_path)
        return
    try:
        for item_name in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item_name)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        print("Pulizia completata.")
    except Exception as e:
        print(f"Errore durante la pulizia di {dir_path}: {e}")
        # Considera se rilanciare l'errore qui se la pulizia è critica
        # raise

# Funzione prepare_input_images come nell'originale
def prepare_input_images(src_folder, dest_folder):
    """Copia le immagini da src a dest convertendole in scala di grigi."""
    print(f"Preparazione input grayscale da {src_folder} a {dest_folder}")
    # Assicurati che dest_folder esista (clean_directory dovrebbe averla gestita)
    os.makedirs(dest_folder, exist_ok=True)

    image_files = sorted([f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    if not image_files:
        print(f"Attenzione: Nessuna immagine trovata in {src_folder}")
        return False # Indica che non c'erano immagini da preparare

    prepared_count = 0
    for filename in image_files:
        src_path = os.path.join(src_folder, filename)
        dest_path = os.path.join(dest_folder, filename)
        try:
            with Image.open(src_path) as img:
                # Converti sempre a RGB prima, poi a L
                rgb_img = img.convert('RGB')
                grayscale_img = rgb_img.convert('L')
                grayscale_img.save(dest_path)
                prepared_count += 1
        except Exception as e:
            print(f"Errore durante la preparazione dell'immagine {filename}: {e}")
            continue # Continua con le prossime immagini

    if prepared_count == 0 and len(image_files) > 0:
         print(f"Errore: Nessuna immagine è stata preparata con successo da {src_folder}")
         return False
    elif prepared_count < len(image_files):
         print(f"Attenzione: Preparazione completata per {prepared_count}/{len(image_files)} immagini.")
    else:
         print(f"Preparazione di {prepared_count} immagini completata con successo.")

    return prepared_count > 0

# Funzione run_inference ESATTAMENTE come nell'originale (senza input/output args)
def run_inference(script_path, cuda_device, is_pre_finetuning):
    """Esegue lo script di inferenza (che usa i suoi path interni),
       aggiungendo il checkpoint se non è pre-finetuning."""
    phase_info = "(Pre-Finetuning)" if is_pre_finetuning else "(Post-Finetuning)"
    print(f"Avvio inferenza {phase_info}: {script_path} su CUDA:{cuda_device}")
    # Stampa le directory che test.py DOVREBBE usare internamente
    print(f"Input directory attesa INTERNAMENTE da test.py: {INPUT_DIR}")
    print(f"Output directory attesa INTERNAMENTE da test.py: {RESULT_DIR}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_device
    command = ["python", script_path] # Comando base

    # Aggiungi checkpoint solo se post-finetuning
    if not is_pre_finetuning:
        if not os.path.exists(POST_FINETUNING_CHECKPOINT):
            print(f"ERRORE CRITICO: Checkpoint post-finetuning non trovato: {POST_FINETUNING_CHECKPOINT}")
            print("L'inferenza (post-finetuning) non può procedere senza il checkpoint specificato.")
            return False
        checkpoint_arg_name = "--model"  # Come nell'originale
        command.extend([checkpoint_arg_name, POST_FINETUNING_CHECKPOINT])
        print(f"Utilizzo checkpoint specificato per post-finetuning ({checkpoint_arg_name}): {POST_FINETUNING_CHECKPOINT}")

    # Esecuzione come nell'originale
    try:
        print(f"Esecuzione comando: {' '.join(command)}")
        # Esegui lo script dalla sua directory base
        result = subprocess.run(command, env=env, check=True, capture_output=True, text=True, cwd=BASE_DIR)
        print("Inferenza completata con successo.")
        stderr_output = result.stderr.strip()
        if stderr_output:
             # Potresti voler filtrare warning comuni qui se necessario
            print("Stderr dallo script di inferenza:\n", stderr_output)
        time.sleep(2) # Dai tempo al filesystem
        # Verifica se l'output directory attesa (RESULT_DIR) contiene file
        if not os.path.isdir(RESULT_DIR) or not os.listdir(RESULT_DIR):
             print(f"Attenzione: L'inferenza è terminata senza errori, ma la cartella di output attesa {RESULT_DIR} è vuota o non esiste.")
             # Potrebbe essere un problema con test.py, ma l'esecuzione è ok
        return True
    except subprocess.CalledProcessError as e:
        print(f"Errore durante l'esecuzione dell'inferenza:")
        print(f"Comando: {' '.join(e.cmd)}")
        print(f"Working Directory: {BASE_DIR}")
        print(f"Return code: {e.returncode}")
        print(f"Output (stdout): {e.stdout}")
        print(f"Error (stderr): {e.stderr}") # Stampa l'errore stderr che hai visto prima
        return False
    except FileNotFoundError:
        print(f"Errore: Script di inferenza non trovato: {script_path}")
        return False
    except Exception as e:
        print(f"Errore imprevisto durante l'esecuzione di run_inference: {e}")
        return False


# Funzione per spostare i risultati (questa rimane necessaria)
def move_results(source_dir, target_dir):
    """Sposta i file immagine da source_dir a target_dir."""
    print(f"Spostamento risultati da {source_dir} a {target_dir}")
    moved_count = 0
    if not os.path.isdir(source_dir):
        print(f"Errore: La cartella sorgente {source_dir} non esiste. Nessun file da spostare.")
        return False
    if not os.listdir(source_dir):
        print(f"Attenzione: La cartella sorgente {source_dir} è vuota. Nessun file da spostare.")
        return True # Non è un errore

    # target_dir dovrebbe essere stata creata/pulita
    os.makedirs(target_dir, exist_ok=True) # Assicurati esista

    files_to_move = glob.glob(os.path.join(source_dir, '*.png')) + \
                    glob.glob(os.path.join(source_dir, '*.jpg')) + \
                    glob.glob(os.path.join(source_dir, '*.jpeg')) + \
                    glob.glob(os.path.join(source_dir, '*.bmp')) + \
                    glob.glob(os.path.join(source_dir, '*.tiff'))

    if not files_to_move:
        print(f"Attenzione: Nessun file immagine trovato in {source_dir} da spostare.")
        return True # Non è un errore

    for src_path in files_to_move:
        filename = os.path.basename(src_path)
        dest_path = os.path.join(target_dir, filename)
        try:
            shutil.move(src_path, dest_path)
            moved_count += 1
        except Exception as e:
            print(f"Errore durante lo spostamento di {filename} da {source_dir} a {target_dir}: {e}")
            # Potresti voler accumulare errori invece di ritornare False subito
            continue

    if moved_count == 0 and len(files_to_move) > 0:
         print(f"Errore: Nessun file è stato spostato con successo da {source_dir}")
         return False
    elif moved_count < len(files_to_move):
         print(f"Attenzione: Spostati solo {moved_count}/{len(files_to_move)} file da {source_dir}.")
         return False # Consideralo fallimento parziale
    else:
        print(f"Spostati {moved_count} file con successo in {target_dir}.")
        return True


# --- Funzioni per le Metriche (FID, SSIM, PSNR, LPIPS) ---
#    (Queste funzioni rimangono sostanzialmente invariate rispetto all'ultima versione,
#     leggono dalle cartelle che gli vengono passate)

def calculate_fid(original_color_folder, result_color_folder, batch_size=50, dims=2048):
    """Calcola FID tra due cartelle di immagini."""
    if not FID_AVAILABLE:
        print("Calcolo FID saltato (libreria pytorch-fid non trovata).")
        return None
    # ... (resto della funzione calculate_fid invariato) ...
    print(f"Calcolo FID tra: {original_color_folder} e {result_color_folder}")
    if not os.path.isdir(original_color_folder) or not os.listdir(original_color_folder):
        print(f"Errore FID: La cartella originale '{original_color_folder}' non esiste o è vuota.")
        return None
    if not os.path.isdir(result_color_folder) or not os.listdir(result_color_folder):
        print(f"Errore FID: La cartella risultati '{result_color_folder}' non esiste o è vuota.")
        return None
    try:
        paths = [original_color_folder, result_color_folder]
        fid_device = "cpu"
        if LPIPS_AVAILABLE and torch.cuda.is_available():
             try:
                  cuda_device_idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(',')[0]
                  fid_device = f"cuda:{cuda_device_idx}"
             except Exception as e:
                  print(f"Attenzione: impossibile determinare CUDA device specifico ({e}), usando cuda:0 o cpu.")
                  fid_device = "cuda:0"
                  if not torch.cuda.is_available(): fid_device="cpu"
        print(f"Utilizzo device '{fid_device}' per FID.")
        fid_value = calculate_fid_given_paths(paths, batch_size, fid_device, dims)
        if np.isnan(fid_value): # Aggiunto controllo NaN
             print(f"Attenzione: FID calcolato è NaN. Possibile problema con le immagini o attivazioni.")
             return None
        print(f"Calcolato FID: {fid_value:.4f}")
        return fid_value
    except Exception as e:
        print(f"Errore durante il calcolo FID: {e}")
        return None


def calculate_metrics_for_folder(original_color_folder, result_color_folder):
    """Calcola SSIM, PSNR, LPIPS per coppie di immagini e FID per l'intera cartella."""
    print(f"Calcolo metriche tra {original_color_folder} e {result_color_folder}")
    ssim_values = []
    psnr_values = []
    lpips_values = []
    lpips_model = None
    lpips_device = "cpu"
    # ... (resto della funzione calculate_metrics_for_folder invariato,
    #      inclusa inizializzazione LPIPS, verifica cartelle, loop sui file comuni,
    #      calcolo SSIM, PSNR, LPIPS per coppia) ...

    # --- Inizializza LPIPS (solo se disponibile) ---
    if LPIPS_AVAILABLE:
        try:
            if torch.cuda.is_available():
                 try:
                     cuda_device_idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(',')[0]
                     lpips_device = torch.device(f"cuda:{cuda_device_idx}")
                 except Exception as e:
                     print(f"Attenzione: impossibile determinare CUDA device specifico ({e}), usando cuda:0 o cpu.")
                     lpips_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            else:
                 lpips_device = torch.device("cpu")
            print(f"Utilizzo device '{lpips_device}' per LPIPS.")
            lpips_model = lpips.LPIPS(net='alex').to(lpips_device)
            lpips_model.eval()
        except Exception as e:
            print(f"Errore nell'inizializzazione di LPIPS: {e}. Calcolo LPIPS saltato.")
            lpips_model = None
    else:
        print("Calcolo LPIPS saltato (librerie non disponibili).")

    # --- Verifica Esistenza Cartelle ---
    if not os.path.isdir(original_color_folder):
         print(f"Errore: La cartella originale {original_color_folder} non esiste.")
         return None, None, None, None
    if not os.path.isdir(result_color_folder):
         print(f"Errore: La cartella dei risultati {result_color_folder} non esiste.")
         return None, None, None, None

    # --- Trova e Abbina File ---
    original_files = sorted([f for f in os.listdir(original_color_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    result_files = sorted([f for f in os.listdir(result_color_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

    print(f"Trovati {len(original_files)} file originali e {len(result_files)} file risultato.")

    if not original_files:
        print(f"Errore: Nessun file immagine trovato in {original_color_folder}.")
        return None, None, None, None
    if not result_files:
        print(f"Errore: Nessun file immagine trovato in {result_color_folder}.")
        fid_value = calculate_fid(original_color_folder, result_color_folder)
        return None, None, None, fid_value

    original_set = set(original_files)
    result_set = set(result_files)
    common_files = sorted(list(original_set.intersection(result_set)))
    missing_in_result = sorted(list(original_set - result_set))
    extra_in_result = sorted(list(result_set - original_set))

    if not common_files:
        print(f"Errore: Nessun file con nome corrispondente trovato tra le cartelle.")
        fid_value = calculate_fid(original_color_folder, result_color_folder)
        return None, None, None, fid_value

    if missing_in_result or extra_in_result:
         print(f"Attenzione: Discrepanza nei file tra le cartelle:")
         if missing_in_result: print(f"  - File mancanti in '{os.path.basename(result_color_folder)}': {len(missing_in_result)}")
         if extra_in_result: print(f"  - File extra in '{os.path.basename(result_color_folder)}': {len(extra_in_result)}")
         print(f"  -> Calcolo metriche SSIM/PSNR/LPIPS solo per i {len(common_files)} file comuni.")

    # --- Calcola Metriche per Coppia ---
    matched_files_count = 0
    for filename in common_files:
        original_path = os.path.join(original_color_folder, filename)
        result_path = os.path.join(result_color_folder, filename)
        try:
            img_orig_pil = Image.open(original_path).convert('RGB')
            img_orig = np.array(img_orig_pil)
            img_result_pil = Image.open(result_path).convert('RGB')
            img_result = np.array(img_result_pil)

            if img_orig.shape != img_result.shape:
                print(f"Attenzione: Dimensioni non corrispondenti per {filename}. Salto.")
                continue

            # SSIM e PSNR
            try:
                current_ssim = ssim(img_orig, img_result, data_range=255, channel_axis=-1)
                current_psnr = psnr(img_orig, img_result, data_range=255)
                if np.isinf(current_psnr): current_psnr = 100.0
                ssim_values.append(current_ssim)
                psnr_values.append(current_psnr)
            except Exception as e:
                print(f"Errore SSIM/PSNR per {filename}: {e}. Salto.")
                continue

            # LPIPS
            if lpips_model is not None:
                try:
                    img_orig_tensor = lpips.im2tensor(img_orig).to(lpips_device)
                    img_result_tensor = lpips.im2tensor(img_result).to(lpips_device)
                    with torch.no_grad():
                        lp = lpips_model(img_orig_tensor, img_result_tensor)
                    lpips_values.append(lp.item())
                except Exception as e:
                    print(f"Errore LPIPS per {filename}: {e}. Salto LPIPS per questa.")

            matched_files_count += 1
        except Exception as e:
            print(f"Errore processamento immagine {filename}: {e}")

    # --- Calcola Medie e FID ---
    average_ssim = np.mean(ssim_values) if ssim_values else None
    average_psnr = np.mean(psnr_values) if psnr_values else None
    average_lpips = np.mean(lpips_values) if lpips_values else None

    print("-" * 20)
    if average_ssim is not None: print(f"Media SSIM ({matched_files_count} immagini): {average_ssim:.4f}")
    else: print("Media SSIM: N/A")
    if average_psnr is not None: print(f"Media PSNR ({matched_files_count} immagini): {average_psnr:.2f}")
    else: print("Media PSNR: N/A")
    if average_lpips is not None: print(f"Media LPIPS ({len(lpips_values)} immagini): {average_lpips:.4f}")
    elif lpips_model is not None: print("Media LPIPS: N/A")
    else: print("Media LPIPS: N/A (Modello non caricato)")
    print("-" * 20)

    fid_value = calculate_fid(original_color_folder, result_color_folder)

    return average_ssim, average_psnr, average_lpips, fid_value


# --- Main Execution Logic (Aggiornato per usare la sequenza corretta) ---
def main(is_pre_finetuning):
    phase_log = "Pre-Finetuning" if is_pre_finetuning else "Post-Finetuning"
    print(f"\n{'='*60}")
    print(f"AVVIO PIPELINE ({phase_log})")
    print(f"{'='*60}\n")

    # Determina la directory base dei risultati FINALI per questa fase
    phase_result_base_dir = RESULTS_PRE_TUNING_BASE if is_pre_finetuning else RESULTS_POST_TUNING_BASE
    print(f"Directory base per i risultati finali di questa fase: {phase_result_base_dir}")
    os.makedirs(phase_result_base_dir, exist_ok=True)

    try:
        subfolders = sorted([f for f in os.listdir(TEST_SET_DIR) if os.path.isdir(os.path.join(TEST_SET_DIR, f))])
    except Exception as e:
         print(f"ERRORE CRITICO: Impossibile leggere le sottocartelle in {TEST_SET_DIR}: {e}")
         exit(1)

    if not subfolders:
        print(f"Errore: Nessuna sottocartella trovata in {TEST_SET_DIR}")
        return

    print(f"Trovate {len(subfolders)} sottocartelle nel test set: {', '.join(subfolders)}\n")

    results_list = []
    all_avg_ssims, all_avg_psnrs, all_avg_lpips, all_avg_fids = [], [], [], []

    # --- Loop sulle Sottocartelle ---
    for folder_name in subfolders:
        print(f"\n--- Elaborazione cartella: {folder_name} ---\n")
        original_frames_dir = os.path.join(TEST_SET_DIR, folder_name) # Sorgente originale colori
        # Destinazione FINALE per i risultati di questa cartella/fase
        target_result_dir = os.path.join(phase_result_base_dir, folder_name)

        folder_ssim, folder_psnr, folder_lpips, folder_fid = None, None, None, None
        error_stage = None

        try:
            # Fase 0: Pulizia directory
            # Pulisci le directory usate da test.py (INPUT_DIR, RESULT_DIR)
            # Pulisci la directory di destinazione finale (target_result_dir)
            print("Fase 0: Pulizia Directory...")
            clean_directory(INPUT_DIR)
            clean_directory(RESULT_DIR)
            clean_directory(target_result_dir) # Pulisce/Crea la destinazione finale
            print("Fase 0: Completata.")

            # Fase 1: Preparazione Input Grayscale (in INPUT_DIR)
            print("\nFase 1: Preparazione Input Grayscale...")
            if not prepare_input_images(original_frames_dir, INPUT_DIR):
                print(f"Errore/Nessuna immagine in input per {folder_name}. Salto.")
                error_stage = 'Input_Prep'
                raise StopIteration
            print("Fase 1: Completata.")

            # Fase 2: Esecuzione Inferenza (output va in RESULT_DIR)
            print("\nFase 2: Esecuzione Inferenza...")
            # Usa la funzione run_inference originale!
            if not run_inference(INFERENCE_SCRIPT, CUDA_DEVICE, is_pre_finetuning):
                print(f"Errore durante l'inferenza per {folder_name}. Salto spostamento e metriche.")
                error_stage = 'Inference'
                raise StopIteration
            print("Fase 2: Completata.")

            # Fase 3: Spostamento Risultati (da RESULT_DIR a target_result_dir)
            print("\nFase 3: Spostamento Risultati...")
            if not move_results(RESULT_DIR, target_result_dir):
                 print(f"Errore durante lo spostamento dei risultati per {folder_name}. Salto metriche.")
                 error_stage = 'Move_Results'
                 raise StopIteration
            print("Fase 3: Completata.")

            # Fase 4: Calcolo Metriche (su target_result_dir)
            print("\nFase 4: Calcolo Metriche...")
            folder_ssim, folder_psnr, folder_lpips, folder_fid = calculate_metrics_for_folder(
                original_frames_dir, target_result_dir # Leggi da originale e target finale!
            )
            print("Fase 4: Completata.")

        except StopIteration:
             print(f"Elaborazione interrotta per {folder_name} nella fase: {error_stage}")
        except Exception as e:
             print(f"Errore imprevisto durante l'elaborazione di {folder_name}: {e}")
             error_stage = error_stage or 'Unknown'
        finally:
            # Registra risultati/errori
            # ... (logica di formattazione stringhe e append a results_list come prima) ...
            ssim_val_str = f"{folder_ssim:.4f}" if folder_ssim is not None else ('N/A' if error_stage is None else f"Error_{error_stage}")
            psnr_val_str = f"{folder_psnr:.2f}" if folder_psnr is not None else ('N/A' if error_stage is None else f"Error_{error_stage}")
            lpips_val_str = f"{folder_lpips:.4f}" if folder_lpips is not None else ('N/A' if error_stage is None or error_stage in ['Move_Results', 'Inference', 'Input_Prep'] else f"Error_{error_stage}")
            if folder_fid is not None:
                 fid_val_str = f"{folder_fid:.4f}"
            elif error_stage is None and not FID_AVAILABLE:
                 fid_val_str = "N/A_FID_Unavailable"
            elif error_stage is None and FID_AVAILABLE:
                 fid_val_str = "N/A_FID_Calculation"
            else:
                 fid_val_str = f"Error_{error_stage}"

            results_list.append({
                'Folder': folder_name, 'Average_SSIM': ssim_val_str, 'Average_PSNR': psnr_val_str,
                'Average_LPIPS': lpips_val_str, 'Average_FID': fid_val_str, 'Phase': phase_log
            })

            # Aggiungi alle medie totali solo se non ci sono stati errori
            if error_stage is None:
                if folder_ssim is not None: all_avg_ssims.append(folder_ssim)
                if folder_psnr is not None: all_avg_psnrs.append(folder_psnr)
                if folder_lpips is not None: all_avg_lpips.append(folder_lpips)
                if folder_fid is not None and np.isfinite(folder_fid): all_avg_fids.append(folder_fid)

            print(f"\n--- Fine Elaborazione cartella: {folder_name} ---")


    # --- Calcola e Stampa Medie Totali ---
    # ... (logica per calcolare medie e formattare stringhe come prima) ...
    print("\n" + "="*60)
    print(f"CALCOLO MEDIE COMPLESSIVE ({phase_log})")
    print("="*60)
    total_average_ssim = np.mean(all_avg_ssims) if all_avg_ssims else None
    total_average_psnr = np.mean(all_avg_psnrs) if all_avg_psnrs else None
    total_average_lpips = np.mean(all_avg_lpips) if all_avg_lpips else None
    total_average_fid = np.mean(all_avg_fids) if all_avg_fids else None
    ssim_avg_str = f"{total_average_ssim:.4f}" if total_average_ssim is not None else 'N/A'
    psnr_avg_str = f"{total_average_psnr:.2f}" if total_average_psnr is not None else 'N/A'
    lpips_avg_str = f"{total_average_lpips:.4f}" if total_average_lpips is not None else ('N/A_Unavailable' if not LPIPS_AVAILABLE else 'N/A')
    fid_avg_str = f"{total_average_fid:.4f}" if total_average_fid is not None else ('N/A_Unavailable' if not FID_AVAILABLE else 'N/A')
    print(f"SSIM Media Totale ({len(all_avg_ssims)} cartelle): {ssim_avg_str}")
    print(f"PSNR Media Totale ({len(all_avg_psnrs)} cartelle): {psnr_avg_str}")
    print(f"LPIPS Media Totale ({len(all_avg_lpips)} cartelle): {lpips_avg_str}")
    print(f"FID Media Totale ({len(all_avg_fids)} cartelle): {fid_avg_str}")
    print("-" * 60)


    # --- Salva Risultati su CSV ---
    # ... (logica per salvare il DataFrame come prima) ...
    if not results_list:
        print("\nNessun risultato da salvare nel CSV.")
    else:
        results_list.append({
            'Folder': 'Overall_Average', 'Average_SSIM': ssim_avg_str, 'Average_PSNR': psnr_avg_str,
            'Average_LPIPS': lpips_avg_str, 'Average_FID': fid_avg_str, 'Phase': phase_log
        })
        try:
            df = pd.DataFrame(results_list)
            column_order = ['Folder', 'Average_SSIM', 'Average_PSNR', 'Average_LPIPS', 'Average_FID', 'Phase']
            df = df.reindex(columns=column_order)
            output_file = OUTPUT_CSV
            file_exists = os.path.exists(output_file)
            write_header = not file_exists if not is_pre_finetuning else True
            write_mode = 'w' if is_pre_finetuning else 'a'
            print(f"\nSalvataggio risultati in: {output_file} (Modalità: {write_mode.upper()}, Header: {write_header})")
            df.to_csv(output_file, index=False, mode=write_mode, header=write_header)
            print("Salvataggio CSV completato.")
        except Exception as e:
            print(f"ERRORE durante il salvataggio del file CSV ({output_file}): {e}")

    print("\n" + "="*60)
    print(f"PIPELINE ({phase_log}) COMPLETATA")
    print("="*60 + "\n")


# --- Blocco di Esecuzione Principale ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline per colorizzazione, spostamento risultati e calcolo metriche (SSIM, PSNR, FID, LPIPS) per ColorMNet.")
    # ... (argparse come prima) ...
    parser.add_argument(
        '--pre_finetuning',
        action='store_true',
        help="Esegui in modalità PRE-finetuning (usa modello base, salva risultati in 'preTuning', sovrascrive CSV)."
    )
    args = parser.parse_args()

    # ... (stampa modalità operativa e controlli preliminari come prima) ...
    if args.pre_finetuning:
        print("Modalità operativa: PRE-FINETUNING")
        print(f"I risultati verranno salvati in: {RESULTS_PRE_TUNING_BASE}")
        print(f"Il file CSV ({OUTPUT_CSV}) verrà sovrascritto.")
    else:
        print("Modalità operativa: POST-FINETUNING")
        print(f"I risultati verranno salvati in: {RESULTS_POST_TUNING_BASE}")
        print(f"I risultati verranno aggiunti al file CSV ({OUTPUT_CSV}).")
        if not os.path.exists(POST_FINETUNING_CHECKPOINT):
             print(f"\n!!! ATTENZIONE CRITICA !!! Checkpoint post-finetuning NON TROVATO:")
             print(f"  {POST_FINETUNING_CHECKPOINT}")
        else:
             print(f"Utilizzerà il checkpoint post-finetuning: {POST_FINETUNING_CHECKPOINT}")
    print("-" * 30)

    essential_paths = {
        "Directory Base ColormNet": BASE_DIR, "Directory Test Set": TEST_SET_DIR,
        "Script Inferenza": INFERENCE_SCRIPT
    }
    abort = False
    for name, path in essential_paths.items():
        check = os.path.isdir if "Directory" in name else os.path.isfile
        if not check(path):
            print(f"ERRORE CRITICO: {name} non trovato/a in '{path}'")
            abort = True
    if abort: exit(1)

    # Crea directory base INPUT/RESULT usate da test.py se non esistono
    # clean_directory nel loop le svuoterà comunque
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Avvia la pipeline principale
    main(args.pre_finetuning)
