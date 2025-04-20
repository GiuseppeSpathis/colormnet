import os
import shutil
import subprocess
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import time  # Per dare tempo al filesystem, se necessario

# Importa metriche già presenti
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr  # <-- NUOVO IMPORT

# --- Configurazione dei Path ---
SCRATCH = "/scratch.hpc/giuseppe.spathis/"
BASE_DIR = "/scratch.hpc/giuseppe.spathis/colormnet"
TEST_SET_DIR = os.path.join(BASE_DIR, "test_set")
INPUT_DIR_BASE = os.path.join(BASE_DIR, "input")
RESULT_DIR_BASE = os.path.join(BASE_DIR, "result")
INPUT_SUBDIR_NAME = "blackswan"
RESULT_SUBDIR_NAME = "blackswan"
INPUT_DIR = os.path.join(INPUT_DIR_BASE, INPUT_SUBDIR_NAME)
RESULT_DIR = os.path.join(RESULT_DIR_BASE, RESULT_SUBDIR_NAME)
INFERENCE_SCRIPT = os.path.join(BASE_DIR, "test.py")
OUTPUT_CSV = os.path.join(SCRATCH, "ssim_psnr_fid_lpips_resultsAlluminium.csv")  # <-- Aggiornato nome file CSV
CUDA_DEVICE = "0"

# --- Path del checkpoint per post-finetuning ---
POST_FINETUNING_CHECKPOINT = "/scratch.hpc/giuseppe.spathis/colormnet/savingTuning/saves/tuning_s2/tuning_s2_4000.pth"

# --- Definizione delle regioni in cui è presente l'alluminio per le cartelle "001" e "002" ---
region_mapping = {
    "001": {
        "center_x": 500,
        "center_y": 200,
        "rect_width": 500,
        "rect_height": 100
    },
    "002": {
        "center_x": 350,
        "center_y": 250,
        "rect_width": 300,
        "rect_height": 400
    }
}

# --- Funzioni Helper ---

def clean_directory(dir_path):
    """Rimuove tutti i file e le sottocartelle in una directory."""
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
                if img.mode == 'L' or img.mode == 'LA':
                    grayscale_img = img.convert('L')
                    grayscale_img.save(dest_path)
                else:
                    rgb_img = img.convert('RGB')
                    grayscale_img = rgb_img.convert('L')
                    grayscale_img.save(dest_path)
        except Exception as e:
            print(f"Errore durante la preparazione dell'immagine {filename}: {e}")
            return False
    print(f"Preparazione di {len(image_files)} immagini completata.")
    return True

def run_inference(script_path, cuda_device, is_pre_finetuning):
    """Esegue lo script di inferenza, aggiungendo il checkpoint se non è pre-finetuning."""
    phase_info = "(Pre-Finetuning)" if is_pre_finetuning else "(Post-Finetuning)"
    print(f"Avvio inferenza {phase_info}: {script_path} su CUDA:{cuda_device}")
    print(f"Input directory attesa da test.py: {INPUT_DIR}")
    print(f"Output directory attesa da test.py: {RESULT_DIR}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_device
    command = ["python", script_path]

    if not is_pre_finetuning:
        if not os.path.exists(POST_FINETUNING_CHECKPOINT):
                print(f"ERRORE CRITICO: Checkpoint post-finetuning non trovato: {POST_FINETUNING_CHECKPOINT}")
                print("L'inferenza non può procedere senza il checkpoint specificato.")
                return False
        checkpoint_arg_name = "--model"  # Corretto per test.py
        command.extend([checkpoint_arg_name, POST_FINETUNING_CHECKPOINT])
        print(f"Utilizzo checkpoint specificato per post-finetuning (--model): {POST_FINETUNING_CHECKPOINT}")

    try:
        print(f"Esecuzione comando: {' '.join(command)}")
        result = subprocess.run(command, env=env, check=True, capture_output=True, text=True, cwd=BASE_DIR)
        print("Inferenza completata con successo.")
        if result.stderr:
            print("Stderr script:\n", result.stderr)
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

def crop_image(np_img, region):
    """
    Ritaglia l'immagine (numpy array) secondo la regione data.
    La regione è un dizionario con "center_x", "center_y", "rect_width", "rect_height".
    """
    center_x = region["center_x"]
    center_y = region["center_y"]
    rect_width = region["rect_width"]
    rect_height = region["rect_height"]
    # Calcola le coordinate del rettangolo
    left = int(center_x - rect_width / 2)
    top = int(center_y - rect_height / 2)
    right = left + rect_width
    bottom = top + rect_height
    # Le immagini numpy hanno shape (H, W, C): [y, x, c]
    return np_img[top:bottom, left:right]

# --- Funzione per il calcolo di FID ---
def calculate_fid(original_color_folder, result_color_folder, batch_size=50, dims=2048, device='cuda'):
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
        paths = [original_color_folder, result_color_folder]
        # Forziamo l'uso della GPU specificata tramite CUDA_VISIBLE_DEVICES se disponibile
        fid_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        fid_value = calculate_fid_given_paths(paths, batch_size, fid_device, dims)
        print(f"Calcolato FID per la cartella (su {fid_device}): {fid_value:.4f}")
        return fid_value
    except ImportError:
        print("Errore: La libreria pytorch-fid non è installata. Salto calcolo FID.")
        print("Puoi installarla con: pip install pytorch-fid")
        return None
    except Exception as e:
        print(f"Errore nel calcolo FID per la cartella: {e}")
        return None

# --- Calcolo di SSIM, PSNR, LPIPS ---
def calculate_metrics_for_folder(original_color_folder, result_color_folder, region=None):
    """
    Calcola SSIM, PSNR, LPIPS per coppie di immagini e FID per l'intera cartella.
    Se 'region' è specificato (dizionario con center_x, center_y, rect_width, rect_height)
    allora le metriche SSIM, PSNR, LPIPS sono calcolate solo sulla porzione ritagliata post inferenza.
    FID è sempre calcolato sull'intera immagine.
    """
    print(f"Calcolo metriche (SSIM, PSNR, LPIPS) tra {original_color_folder} e {result_color_folder}")
    ssim_values = []
    psnr_values = []
    lpips_values = []

    # Inizializza LPIPS
    lpips_model = None
    try:
        import torch
        import lpips
        # Usa la GPU specificata tramite CUDA_VISIBLE_DEVICES se disponibile
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)
        lpips_model = lpips.LPIPS(net='alex').to(device)
        print(f"LPIPS model caricato su: {device_str}")
    except ImportError:
        print("Errore: La libreria lpips non è installata. Salto calcolo LPIPS.")
        print("Puoi installarla con: pip install lpips")
    except Exception as e:
        print(f"Errore nell'importazione o inizializzazione di lpips: {e}. Calcolo LPIPS saltato.")

    original_files = sorted([f for f in os.listdir(original_color_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    if not os.path.exists(result_color_folder):
        print(f"Errore: La cartella dei risultati {result_color_folder} non esiste dopo l'inferenza.")
        return None, None, None, None
    result_files = sorted([f for f in os.listdir(result_color_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

    if len(original_files) != len(result_files):
        print(f"Attenzione: Numero di file non corrispondente tra {original_color_folder} ({len(original_files)}) e {result_color_folder} ({len(result_files)})")

    print(f"Trovati {len(original_files)} file originali e {len(result_files)} file risultato.")

    matched_files = 0
    for filename in original_files:
        original_path = os.path.join(original_color_folder, filename)
        result_path = os.path.join(result_color_folder, filename)

        if not os.path.exists(result_path):
            print(f"Attenzione: File risultato corrispondente non trovato per {filename} in {result_color_folder}. Salto.")
            continue

        try:
            img_orig_pil = Image.open(original_path).convert('RGB')
            img_orig = np.array(img_orig_pil)
            img_result_pil = Image.open(result_path).convert('RGB')
            img_result = np.array(img_result_pil)

            # Se è stata specificata una regione, eseguo il crop per SSIM, PSNR, LPIPS
            img_orig_metric = img_orig
            img_result_metric = img_result
            if region is not None:
                img_orig_metric = crop_image(img_orig, region)
                img_result_metric = crop_image(img_result, region)
                # Verifica che le dimensioni siano coerenti dopo il crop
                if img_orig_metric.shape != img_result_metric.shape:
                    print(f"Attenzione: Dimensioni non corrispondenti dopo il crop per {filename} (Originale: {img_orig_metric.shape}, Risultato: {img_result_metric.shape}). Salto metriche per questa immagine.")
                    continue
                if img_orig_metric.size == 0 or img_result_metric.size == 0:
                    print(f"Attenzione: Immagine vuota dopo il crop per {filename}. Salto metriche per questa immagine.")
                    continue


            # Calcola SSIM e PSNR sulla porzione (o intera se no crop)
            try:
                s = ssim(img_orig_metric, img_result_metric, data_range=255, channel_axis=-1 if img_orig_metric.ndim == 3 else None, multichannel=True if img_orig_metric.ndim == 3 else False)
                ssim_values.append(s)

                # PSNR gestisce automaticamente le immagini grayscale o a colori
                p = psnr(img_orig_metric, img_result_metric, data_range=255)
                psnr_values.append(p)
            except ValueError as ve:
                print(f"Errore nel calcolo SSIM/PSNR per {filename}: {ve}. Salto.")
                continue

            # Calcolo LPIPS su coppia di immagini (sulla porzione se crop) (se lpips_model è disponibile)
            if lpips_model is not None and 'torch' in locals() and 'device' in locals():
                try:
                    # Converti numpy array a tensor e riorganizza in CxHxW, aggiungi batch dimension
                    img_orig_tensor = torch.from_numpy(img_orig_metric).permute(2, 0, 1).unsqueeze(0).float().to(device)
                    img_result_tensor = torch.from_numpy(img_result_metric).permute(2, 0, 1).unsqueeze(0).float().to(device)
                    # Normalizza in intervallo [-1, 1] come atteso da LPIPS
                    img_orig_tensor = img_orig_tensor / 127.5 - 1
                    img_result_tensor = img_result_tensor / 127.5 - 1
                    with torch.no_grad(): # Inferenza non richiede gradienti
                        lp = lpips_model(img_orig_tensor, img_result_tensor)
                    lpips_values.append(lp.item())
                except Exception as e:
                    print(f"Errore nel calcolo LPIPS per {filename}: {e}. Salto LPIPS per questa immagine.")

            matched_files += 1
        except Exception as e:
            print(f"Errore nell'apertura o processamento dell'immagine {filename}: {e}")

    # Calcola medie per SSIM, PSNR e LPIPS
    average_ssim = np.mean(ssim_values) if ssim_values else None
    if average_ssim is not None:
        print(f"Calcolata SSIM su {matched_files} coppie. Media SSIM: {average_ssim:.4f}")
    else:
        print("Attenzione: Nessun valore SSIM valido calcolato per questa cartella.")

    average_psnr = np.mean(psnr_values) if psnr_values else None
    # Gestisce PSNR infiniti (immagini identiche)
    if average_psnr is not None:
        if np.isinf(average_psnr):
             print(f"Calcolata PSNR su {matched_files} coppie. Media PSNR: Inf (immagini identiche)")
        else:
             print(f"Calcolata PSNR su {matched_files} coppie. Media PSNR: {average_psnr:.2f}")
    else:
        print("Attenzione: Nessun valore PSNR valido calcolato per questa cartella.")

    average_lpips = np.mean(lpips_values) if lpips_values else None
    if average_lpips is not None:
        print(f"Calcolato LPIPS su {matched_files} coppie. Media LPIPS: {average_lpips:.4f}")
    else:
        print("Attenzione: Nessun valore LPIPS valido calcolato per questa cartella.")

    # Calcola FID per la cartella (usando le cartelle INTERE, non croppate)
    fid_value = calculate_fid(original_color_folder, result_color_folder)

    return average_ssim, average_psnr, average_lpips, fid_value

# --- Main Execution Logic ---
def main(is_pre_finetuning):
    global INPUT_DIR, RESULT_DIR # Allow modification based on folder name if needed

    phase_log = "Pre-Finetuning" if is_pre_finetuning else "Post-Finetuning"
    print(f"Avvio pipeline di calcolo metriche ({phase_log})...")

    subfolders = sorted([f for f in os.listdir(TEST_SET_DIR) if os.path.isdir(os.path.join(TEST_SET_DIR, f))])
    if not subfolders:
        print(f"Errore: Nessuna sottocartella trovata in {TEST_SET_DIR}")
        return

    print(f"Trovate le seguenti sottocartelle nel test set: {', '.join(subfolders)}")

    results = []
    all_avg_ssims = []
    all_avg_psnrs = []
    all_avg_lpips = []
    all_avg_fids = []

    for folder_name in subfolders:
        # Elabora solo le cartelle definite in region_mapping
        if folder_name not in region_mapping:
            print(f"Salto la cartella {folder_name} perché non presente nel mapping delle regioni.")
            continue

        print(f"\n--- Elaborazione cartella: {folder_name} ---")
        original_frames_dir = os.path.join(TEST_SET_DIR, folder_name)

        # Aggiorna i path di input/output per usare sottocartelle specifiche se necessario
        # (in questo caso usiamo sempre "blackswan" come definito all'inizio, ma potresti
        # voler cambiare questo comportamento se test.py si aspetta nomi diversi)
        # INPUT_DIR = os.path.join(INPUT_DIR_BASE, folder_name) # Esempio se necessario
        # RESULT_DIR = os.path.join(RESULT_DIR_BASE, folder_name) # Esempio se necessario
        # os.makedirs(INPUT_DIR, exist_ok=True)
        # os.makedirs(RESULT_DIR, exist_ok=True)

        # 1. Pulisci directory (quelle fisse "blackswan")
        clean_directory(INPUT_DIR)
        clean_directory(RESULT_DIR)

        # 2. Prepara input
        if not prepare_input_images(original_frames_dir, INPUT_DIR):
            print(f"Errore nella preparazione input per {folder_name}. Salto questa cartella.")
            results.append({
                'Folder': f"{int(folder_name)}_alluminium",
                'Average_SSIM': 'Error_Input',
                'Average_PSNR': 'Error_Input',
                'Average_LPIPS': 'Error_Input',
                'Average_FID': 'Error_Input',
                'Phase': phase_log
            })
            continue

        # 3. Esegui inferenza
        if not run_inference(INFERENCE_SCRIPT, CUDA_DEVICE, is_pre_finetuning):
            print(f"Errore durante l'inferenza per {folder_name}. Salto questa cartella.")
            results.append({
                'Folder': f"{int(folder_name)}_alluminium",
                'Average_SSIM': 'Error_Inference',
                'Average_PSNR': 'Error_Inference',
                'Average_LPIPS': 'Error_Inference',
                'Average_FID': 'Error_Inference',
                'Phase': phase_log
            })
            continue

        # 4. Calcola Metriche (SSIM, PSNR, LPIPS, FID) passando la regione di crop specifica per la cartella
        region = region_mapping.get(folder_name) # Ottiene la regione specifica
        avg_ssim, avg_psnr, avg_lpips, fid_value = calculate_metrics_for_folder(original_frames_dir, RESULT_DIR, region)

        # Gestione valori None o Inf per il CSV
        ssim_val_str = f"{avg_ssim:.4f}" if avg_ssim is not None else 'N/A'
        if avg_psnr is not None:
             psnr_val_str = 'Inf' if np.isinf(avg_psnr) else f"{avg_psnr:.2f}"
        else:
             psnr_val_str = 'N/A'
        lpips_val_str = f"{avg_lpips:.4f}" if avg_lpips is not None else 'N/A'
        fid_val_str = f"{fid_value:.4f}" if fid_value is not None else 'N/A'


        # Controlla se almeno una metrica non è calcolabile e aggiorna le stringhe di errore se necessario
        if avg_ssim is None or avg_psnr is None or avg_lpips is None or fid_value is None:
            print(f"Almeno una metrica non calcolabile per {folder_name}.")
            if not os.path.exists(RESULT_DIR) or not os.listdir(RESULT_DIR): # Controlla anche se la cartella risultato è vuota
                 print("Causa probabile: Cartella risultati mancante o vuota.")
                 ssim_val_str = psnr_val_str = lpips_val_str = fid_val_str = 'Error_ResultDir_Missing/Empty'
            else:
                 # Se la cartella esiste ma ci sono problemi di calcolo specifici
                 if avg_ssim is None: ssim_val_str = 'N/A_Calculation'
                 if avg_psnr is None: psnr_val_str = 'N/A_Calculation'
                 if avg_lpips is None: lpips_val_str = 'N/A_Calculation'
                 if fid_value is None: fid_val_str = 'N/A_Calculation'


        # Modifica il nome da salvare nel CSV: ad es. da "001" diventa "1_alluminium"
        csv_folder_name = f"{int(folder_name)}_alluminium"

        results.append({
            'Folder': csv_folder_name,
            'Average_SSIM': ssim_val_str,
            'Average_PSNR': psnr_val_str,
            'Average_LPIPS': lpips_val_str,
            'Average_FID': fid_val_str,
            'Phase': phase_log
        })

        # Aggiungi alle liste per le medie totali solo se i valori sono numerici validi
        if avg_ssim is not None: all_avg_ssims.append(avg_ssim)
        if avg_psnr is not None and not np.isinf(avg_psnr): all_avg_psnrs.append(avg_psnr) # Escludi Inf dalla media
        if avg_lpips is not None: all_avg_lpips.append(avg_lpips)
        if fid_value is not None: all_avg_fids.append(fid_value)

    # 5. Calcola Medie Totali (escludendo eventuali Inf da PSNR)
    total_average_ssim = np.mean(all_avg_ssims) if all_avg_ssims else None
    total_average_psnr = np.mean(all_avg_psnrs) if all_avg_psnrs else None # Già filtrato Inf
    total_average_lpips = np.mean(all_avg_lpips) if all_avg_lpips else None
    total_average_fid = np.mean(all_avg_fids) if all_avg_fids else None

    ssim_avg_str = f"{total_average_ssim:.4f}" if total_average_ssim is not None else 'N/A'
    psnr_avg_str = f"{total_average_psnr:.2f}" if total_average_psnr is not None else 'N/A'
    lpips_avg_str = f"{total_average_lpips:.4f}" if total_average_lpips is not None else 'N/A'
    fid_avg_str = f"{total_average_fid:.4f}" if total_average_fid is not None else 'N/A'

    print(f"\n--- Risultato Complessivo ({phase_log}) ---")
    print(f"SSIM Media Totale: {ssim_avg_str}")
    print(f"PSNR Media Totale (esclusi Inf): {psnr_avg_str}")
    print(f"LPIPS Media Totale: {lpips_avg_str}")
    print(f"FID Media Totale: {fid_avg_str}")

    # Aggiungi riga per medie totali
    results.append({
        'Folder': 'Overall_Average',
        'Average_SSIM': ssim_avg_str,
        'Average_PSNR': psnr_avg_str,
        'Average_LPIPS': lpips_avg_str,
        'Average_FID': fid_avg_str,
        'Phase': phase_log
    })

    if not results:
        print("\nNessun risultato da salvare nel CSV.")
        print("\nPipeline completata.")
        return

    # --- Sezione di scrittura CSV Modificata ---
    try:
        df = pd.DataFrame(results)
        column_order = ['Folder', 'Average_SSIM', 'Average_PSNR', 'Average_LPIPS', 'Average_FID', 'Phase']
        df = df[column_order] # Assicura l'ordine delle colonne

        if is_pre_finetuning:
            # Modalità Pre-Finetuning: Sovrascrive il file CSV esistente (o lo crea)
            df.to_csv(OUTPUT_CSV, index=False, mode='w', header=True)
            print(f"\nRisultati (pre-finetuning) salvati sovrascrivendo: {OUTPUT_CSV}")
        else:
            # Modalità Post-Finetuning: Appende al file CSV esistente
            file_exists = os.path.exists(OUTPUT_CSV)

            # Scrivi due righe vuote PRIMA di appendere i nuovi dati, solo se il file esiste già
            if file_exists:
                try:
                    # Apri in modalità append ('a') per aggiungere le righe vuote
                    with open(OUTPUT_CSV, 'a', newline='') as f:
                        f.write('\n\n') # Scrive due caratteri newline
                    print(f"Aggiunte due righe vuote a {OUTPUT_CSV} prima di appendere.")
                except Exception as e:
                    print(f"Errore nell'aggiungere righe vuote a {OUTPUT_CSV}: {e}")
                    # Nonostante l'errore sulle righe vuote, proviamo comunque ad appendere i dati

            # Appende il DataFrame. Scrive l'header solo se il file NON esisteva prima.
            # Le righe vuote sono già state scritte (se il file esisteva).
            df.to_csv(OUTPUT_CSV, index=False, mode='a', header=not file_exists)

            if file_exists:
                # Il file esisteva, quindi abbiamo aggiunto righe vuote e poi appeso i dati senza header
                print(f"\nRisultati (post-finetuning) aggiunti a: {OUTPUT_CSV} dopo due righe vuote.")
            else:
                # Se il file non esisteva, df.to_csv lo ha creato con l'header e i dati
                print(f"\nRisultati (post-finetuning) salvati creando nuovo file: {OUTPUT_CSV}")

    except Exception as e:
        print(f"Errore durante il salvataggio del file CSV: {e}")
    # --- Fine Sezione Modificata ---

    print("\nPipeline completata.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline per calcolare SSIM, PSNR, FID e LPIPS per ColorMNet.")
    parser.add_argument(
        '--pre_finetuning',
        action='store_true',
        help="Indica se l'esecuzione è pre-finetuning (sovrascrive CSV, non usa checkpoint specifico)."
    )
    args = parser.parse_args()

    # Import torch qui per verificare la disponibilità di CUDA prima dell'esecuzione
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA disponibile. Verrà usata la GPU: {CUDA_DEVICE}")
            # Imposta la variabile d'ambiente qui così viene usata anche da LPIPS/FID se necessario
            os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE
        else:
            print("CUDA non disponibile. Verranno usate le CPU.")
            # Assicurati che CUDA_DEVICE non sia impostato se CUDA non è disponibile
            # o che gli script/librerie gestiscano correttamente il caso CPU.
            # CUDA_DEVICE = "" # Potrebbe essere necessario
    except ImportError:
        print("ATTENZIONE: PyTorch non è installato. LPIPS e FID (se usano PyTorch) non funzioneranno.")
        # Potresti voler uscire se PyTorch è strettamente necessario
        # exit(1)


    if args.pre_finetuning:
        print("Modalità operativa: PRE-FINETUNING")
    else:
        print("Modalità operativa: POST-FINETUNING")
        if not os.path.exists(POST_FINETUNING_CHECKPOINT):
            print(f"\n!!! ERRORE CRITICO !!! Checkpoint post-finetuning NON TROVATO: {POST_FINETUNING_CHECKPOINT}")
            print("L'inferenza in modalità post-finetuning non può procedere. Uscita.")
            exit(1) # Esci se il checkpoint è necessario ma non trovato


    # Controlli esistenza path critici
    if not os.path.isdir(BASE_DIR):
        print(f"Errore: BASE_DIR non trovata: {BASE_DIR}")
        exit(1)
    if not os.path.isdir(TEST_SET_DIR):
        print(f"Errore: TEST_SET_DIR non trovata: {TEST_SET_DIR}")
        exit(1)
    if not os.path.isfile(INFERENCE_SCRIPT):
        print(f"Errore: INFERENCE_SCRIPT non trovato: {INFERENCE_SCRIPT}")
        exit(1)

    # Crea directory di input/output se non esistono
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    main(args.pre_finetuning)
