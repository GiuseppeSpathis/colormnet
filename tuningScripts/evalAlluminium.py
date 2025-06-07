import os
import shutil
import subprocess
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import time
import glob
from typing import Dict, Tuple, Optional, List
import tempfile # Importato per gestire cartelle temporanee

# Importa metriche
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

# Import opzionali per LPIPS e FID
try:
    import torch
    import lpips
    LPIPS_AVAILABLE = True
    print("Librerie 'torch' e 'lpips' trovate. Calcolo LPIPS abilitato.")
except ImportError:
    LPIPS_AVAILABLE = False
    print("Attenzione: Librerie 'torch' o 'lpips' non trovate. Calcolo LPIPS non disponibile.")

try:
    from pytorch_fid.fid_score import calculate_fid_given_paths
    FID_AVAILABLE = True
    print("Libreria 'pytorch-fid' trovata. Calcolo FID abilitato.")
except ImportError:
    FID_AVAILABLE = False
    print("Attenzione: Libreria 'pytorch-fid' non trovata. Calcolo FID non disponibile.")

# --- Configurazione dei Path ---
# (Come prima)
SCRATCH = "/scratch.hpc/giuseppe.spathis/"
BASE_DIR = os.path.join(SCRATCH, "colormnet")
TEST_SET_DIR = os.path.join(BASE_DIR, "test_set")
RESULTS_BASE_DIR = os.path.join(SCRATCH, "results")
RESULTS_PRE_TUNING_BASE = os.path.join(RESULTS_BASE_DIR, "preTuning")
RESULTS_POST_TUNING_BASE = os.path.join(RESULTS_BASE_DIR, "postTuning")
FOLDERS_TO_PROCESS = ["001", "007"]
ALUMINIUM_REGIONS: Dict[str, Dict[str, int]] = {
    "001": {"center_x": 400, "center_y": 200, "rect_width": 600, "rect_height": 300},
    "007": {"center_x": 450, "center_y": 350, "rect_width": 650, "rect_height": 300}
}
OUTPUT_CSV = os.path.join(SCRATCH, "evalAlluminium_InsideVsOutside_FIDReg.csv") # Nuovo nome CSV
CUDA_DEVICE = "0"
MAX_PIXEL_VALUE = 255.0

# --- Funzioni Helper ---

def get_crop_slice_and_masks(coords: Dict[str, int], img_shape: Tuple[int, int]) -> \
    Optional[Tuple[Tuple[slice, slice], np.ndarray, np.ndarray]]:
    """Restituisce: (crop_slice, mask_inside, mask_outside) o None."""
    # (Implementazione come prima)
    h, w = img_shape
    cx, cy = coords["center_x"], coords["center_y"]
    rw, rh = coords["rect_width"], coords["rect_height"]
    x_min = max(0, cx - rw // 2)
    x_max = min(w, cx + (rw + 1) // 2)
    y_min = max(0, cy - rh // 2)
    y_max = min(h, cy + (rh + 1) // 2)
    if x_min >= x_max or y_min >= y_max: return None
    crop_slice = (slice(y_min, y_max), slice(x_min, x_max))
    mask_inside = np.zeros((h, w), dtype=bool)
    mask_inside[crop_slice] = True
    mask_outside = ~mask_inside
    return crop_slice, mask_inside, mask_outside

def create_cropped_images_folder(
    source_folder: str,
    target_crop_folder: str,
    region_coords: Dict[str, int],
    common_files: List[str] # Lista di nomi file da processare
    ) -> bool:
    """
    Crea una cartella con le versioni croppate delle immagini specificate.
    Restituisce True se ha successo (anche se 0 immagini sono state croppate),
    False se c'è un errore grave.
    """
    print(f"    Creazione crops in: {target_crop_folder}")
    cropped_count = 0
    try:
        os.makedirs(target_crop_folder, exist_ok=True)
        if not common_files:
             print(f"    Attenzione: Nessun file comune specificato per il crop da {source_folder}.")
             return True # Non è un errore, ma non c'è nulla da croppare

        for filename in common_files:
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(target_crop_folder, filename)
            try:
                with Image.open(source_path) as img:
                    img_rgb = img.convert('RGB') # Assicura RGB
                    img_np = np.array(img_rgb)
                    h, w = img_np.shape[:2]
                    slice_masks_tuple = get_crop_slice_and_masks(region_coords, (h, w))
                    if slice_masks_tuple:
                        crop_slice, _, _ = slice_masks_tuple
                        img_cropped_np = img_np[crop_slice]
                        if img_cropped_np.size > 0:
                            img_cropped_pil = Image.fromarray(img_cropped_np)
                            img_cropped_pil.save(target_path)
                            cropped_count += 1
                        else:
                            print(f"    Attenzione: Crop vuoto per {filename}, non salvato.")
                    else:
                        print(f"    Attenzione: Slice non valido per {filename}, non croppato.")

            except FileNotFoundError:
                 print(f"    Errore: File non trovato durante il crop: {source_path}. Salto file.")
                 continue # Continua con il prossimo file
            except Exception as e:
                print(f"    Errore durante il crop/salvataggio di {filename}: {e}. Salto file.")
                continue # Continua con il prossimo file

        print(f"    Croppate {cropped_count}/{len(common_files)} immagini da {os.path.basename(source_folder)}.")
        return True # Successo anche se 0 immagini sono state effettivamente croppate
    except Exception as e:
        print(f"Errore grave durante la creazione della cartella crops {target_crop_folder}: {e}")
        return False


# --- Funzioni per le Metriche ---

def calculate_fid(original_color_folder: str, result_color_folder: str, batch_size: int = 50, dims: int = 2048) -> Optional[float]:
    """Calcola FID tra due cartelle di immagini."""
    # (Implementazione FID come prima, ora usata sia per Full che Region)
    if not FID_AVAILABLE: return None
    # Path check
    if not os.path.isdir(original_color_folder) or not os.path.isdir(result_color_folder):
        print(f"Errore FID: Cartella non trovata - Orig: '{original_color_folder}', Res: '{result_color_folder}'")
        return None
    # Image check
    def has_images(folder):
        for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff'):
             if glob.glob(os.path.join(folder, f'*{ext}')): return True
        return False
    if not has_images(original_color_folder) or not has_images(result_color_folder):
        # Non stampare errore se le cartelle sono temporanee e potenzialmente vuote
        # print(f"Errore FID: Nessuna immagine trovata in '{original_color_folder}' o '{result_color_folder}'")
        return None

    try:
        paths = [original_color_folder, result_color_folder]
        fid_device = "cpu"
        if LPIPS_AVAILABLE and torch.cuda.is_available():
            try:
                cuda_device_idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(',')[0]
                fid_device = f"cuda:{cuda_device_idx}"
            except Exception: fid_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # print(f"    Calcolo FID con device '{fid_device}' tra: {os.path.basename(original_color_folder)} e {os.path.basename(result_color_folder)}")
        fid_value = calculate_fid_given_paths(paths, batch_size, fid_device, dims)
        if np.isnan(fid_value): return None
        # print(f"    FID calcolato: {fid_value:.4f}")
        return fid_value
    except Exception as e:
        # Non stampare l'errore completo ogni volta, potrebbe essere lungo
        print(f"    Errore durante il calcolo FID: {type(e).__name__}")
        return None

# Funzione per PSNR, SSIM, LPIPS (Inside/Outside)
def calculate_metrics_inside_vs_outside(
    original_path: str,
    result_path: str,
    region_coords: Dict[str, int],
    lpips_model: Optional[torch.nn.Module] = None,
    lpips_device: Optional[torch.device] = None
) -> Dict[str, Optional[float]]:
    """Calcola PSNR/SSIM (Region/Outside) e LPIPS (Region/Full)."""
    # (Implementazione come nello script precedente)
    metrics = {
        "PSNR_Region": None, "PSNR_Outside": None,
        "SSIM_Region": None, "SSIM_Outside_Avg": None,
        "LPIPS_Region": None, "LPIPS_Full": None,
    }
    try:
        img_orig_pil = Image.open(original_path).convert('RGB')
        img_result_pil = Image.open(result_path).convert('RGB')
        img_orig_np = np.array(img_orig_pil)
        img_result_np = np.array(img_result_pil)
        h, w, c = img_orig_np.shape
        if img_orig_np.shape != img_result_np.shape: return metrics

        slice_masks_tuple = get_crop_slice_and_masks(region_coords, (h, w))
        if not slice_masks_tuple: return metrics
        crop_slice, mask_inside, mask_outside = slice_masks_tuple

        # PSNR
        try:
            squared_error_map = (img_orig_np.astype(np.float64) - img_result_np.astype(np.float64)) ** 2
            img_orig_region_np = img_orig_np[crop_slice]; img_result_region_np = img_result_np[crop_slice]
            if img_orig_region_np.size > 0:
                 metrics["PSNR_Region"] = psnr_metric(img_orig_region_np, img_result_region_np, data_range=MAX_PIXEL_VALUE)
                 if np.isinf(metrics["PSNR_Region"]): metrics["PSNR_Region"] = 100.0
            outside_pixels_sq_error = squared_error_map[mask_outside]
            if outside_pixels_sq_error.size > 0:
                mse_outside = np.mean(outside_pixels_sq_error)
                if mse_outside <= 1e-10: metrics["PSNR_Outside"] = 100.0
                else:
                     metrics["PSNR_Outside"] = 10.0 * np.log10((MAX_PIXEL_VALUE**2) / mse_outside)
                     if np.isinf(metrics["PSNR_Outside"]): metrics["PSNR_Outside"] = 100.0
        except Exception as e: print(f"Err PSNR {os.path.basename(original_path)}: {e}")

        # SSIM
        try:
            min_dim = min(h, w); win_size = min(11, min_dim)
            if win_size % 2 == 0: win_size = max(1, win_size - 1)
            # Region
            if win_size >= 3 and img_orig_region_np.size > 0:
                 min_dim_region = min(img_orig_region_np.shape[:2]); win_size_region = min(win_size, min_dim_region)
                 if win_size_region % 2 == 0: win_size_region = max(1, win_size_region -1)
                 if win_size_region >= 3:
                      try: metrics["SSIM_Region"] = ssim_metric(img_orig_region_np, img_result_region_np, data_range=MAX_PIXEL_VALUE, channel_axis=-1, win_size=win_size_region)
                      except ValueError: pass # Ignore win_size errors here
            # Outside
            if win_size >= 3:
                try:
                    ssim_full_val, ssim_map = ssim_metric(img_orig_np, img_result_np, data_range=MAX_PIXEL_VALUE, channel_axis=-1, win_size=win_size, full=True)
                    if ssim_map.ndim == 3: ssim_map_mean = np.mean(ssim_map, axis=-1)
                    else: ssim_map_mean = ssim_map
                    outside_ssim_values = ssim_map_mean[mask_outside]
                    if outside_ssim_values.size > 0: metrics["SSIM_Outside_Avg"] = np.mean(outside_ssim_values)
                except ValueError: pass # Ignore win_size errors here
        except Exception as e: print(f"Err SSIM {os.path.basename(original_path)}: {e}")

        # LPIPS
        if lpips_model is not None and lpips_device is not None:
            try:
                img_orig_tensor = lpips.im2tensor(img_orig_np).to(lpips_device)
                img_result_tensor = lpips.im2tensor(img_result_np).to(lpips_device)
                with torch.no_grad(): metrics["LPIPS_Full"] = lpips_model(img_orig_tensor, img_result_tensor).item()
                if img_orig_region_np.size > 0:
                    img_orig_region_tensor = lpips.im2tensor(img_orig_region_np).to(lpips_device)
                    img_result_region_tensor = lpips.im2tensor(img_result_region_np).to(lpips_device)
                    with torch.no_grad(): metrics["LPIPS_Region"] = lpips_model(img_orig_region_tensor, img_result_region_tensor).item()
            except Exception as e: print(f"Err LPIPS {os.path.basename(original_path)}: {e}")

    except FileNotFoundError: return metrics
    except Exception as e: print(f"Err Generic {os.path.basename(original_path)}: {e}")
    return metrics


# --- Main Execution Logic ---
def main():
    print(f"\n{'='*60}")
    print(f"AVVIO CALCOLO METRICHE ALLUMINIO (Inside vs Outside + FID Regionale)")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"{'='*60}\n")

    # --- Inizializza LPIPS ---
    lpips_model = None; lpips_device = None
    # Usa la variabile LPIPS_AVAILABLE definita a livello globale
    if LPIPS_AVAILABLE:
        try:
            # Setup device (simplified fallback)
            lpips_device = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu")
            print(f"Utilizzo device '{lpips_device}' per LPIPS.")
            lpips_model = lpips.LPIPS(net='alex').to(lpips_device); lpips_model.eval()
            print("Modello LPIPS caricato.")
        except Exception as e:
            print(f"Errore inizializzazione LPIPS: {e}. Calcolo LPIPS non sarà possibile se il modello non è caricato.")
            lpips_model = None # Assicura che sia None se l'inizializzazione fallisce
            # RIMOSSA: LPIPS_AVAILABLE = False # Rimuovi questa riga che causava l'errore

    results_list = []
    phases = ["preTuning", "postTuning"]

    for phase in phases:
        print(f"\n--- FASE: {phase} ---")
        result_base_dir = RESULTS_PRE_TUNING_BASE if phase == "preTuning" else RESULTS_POST_TUNING_BASE

        for folder_name in FOLDERS_TO_PROCESS:
            print(f"\n-- Elaborazione cartella: {folder_name} --")

            original_color_folder = os.path.join(TEST_SET_DIR, folder_name)
            result_color_folder = os.path.join(result_base_dir, folder_name)
            region_coords = ALUMINIUM_REGIONS.get(folder_name)

            # Controlli preliminari (come prima)
            if not os.path.isdir(original_color_folder) or not os.path.isdir(result_color_folder) or not region_coords:
                 print(f"ERRORE: Path o coordinate mancanti. Salto {folder_name}/{phase}.")
                 continue

            # Trova file comuni (come prima)
            try:
                 original_files = sorted([f for f in os.listdir(original_color_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
                 result_files = sorted([f for f in os.listdir(result_color_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
                 common_files = sorted(list(set(original_files) & set(result_files)))
            except FileNotFoundError as e: print(f"Errore lettura dir: {e}. Salto."); continue

            if not common_files:
                print(f"Attenzione: Nessun file comune. Salto {folder_name}/{phase}.")
                continue
            print(f"Trovati {len(common_files)} file comuni.")

            # Accumulatori per metriche PSNR/SSIM/LPIPS
            metrics_accumulator: Dict[str, List[float]] = {k: [] for k in [
                 "PSNR_Region", "PSNR_Outside", "SSIM_Region", "SSIM_Outside_Avg",
                 "LPIPS_Region", "LPIPS_Full"
            ]}

            # Calcola metriche per coppia (PSNR/SSIM/LPIPS)
            for filename in common_files:
                original_path = os.path.join(original_color_folder, filename)
                result_path = os.path.join(result_color_folder, filename)
                # Passa lpips_model (che sarà None se l'init è fallita)
                pair_metrics = calculate_metrics_inside_vs_outside(
                    original_path, result_path, region_coords, lpips_model, lpips_device
                )
                for key, value in pair_metrics.items():
                    if value is not None and key in metrics_accumulator:
                        metrics_accumulator[key].append(value)

            # Calcola Medie PSNR/SSIM/LPIPS
            avg_metrics: Dict[str, Optional[float]] = {}
            num_samples: Dict[str, int] = {}
            for key, values in metrics_accumulator.items():
                 avg_metrics[key] = np.mean(values) if values else None
                 num_samples[key] = len(values)

            # --- Calcolo FID (Full e Regionale) ---
            fid_full_value = None
            fid_region_value = None
            temp_orig_crop_dir = None
            temp_res_crop_dir = None

            # Usa la variabile FID_AVAILABLE definita globalmente
            if FID_AVAILABLE:
                # 1. FID Full (su immagini intere)
                print("  Calcolo FID Full...")
                fid_full_value = calculate_fid(original_color_folder, result_color_folder)
                print(f"  FID Full: {fid_full_value:.4f}" if fid_full_value is not None else "  FID Full: N/A")

                # 2. FID Regionale (su crops)
                print("  Preparazione per FID Regionale...")
                try:
                    prefix = f"fid_crop_{phase}_{folder_name}_"
                    temp_orig_crop_dir = tempfile.mkdtemp(prefix=prefix + "orig_")
                    temp_res_crop_dir = tempfile.mkdtemp(prefix=prefix + "res_")

                    orig_cropped_ok = create_cropped_images_folder(original_color_folder, temp_orig_crop_dir, region_coords, common_files)
                    res_cropped_ok = create_cropped_images_folder(result_color_folder, temp_res_crop_dir, region_coords, common_files)

                    if orig_cropped_ok and res_cropped_ok:
                        print("  Calcolo FID Regionale...")
                        fid_region_value = calculate_fid(temp_orig_crop_dir, temp_res_crop_dir)
                        print(f"  FID Regionale: {fid_region_value:.4f}" if fid_region_value is not None else "  FID Regionale: N/A")
                    else:
                        print("  Errore durante la creazione delle cartelle crop, FID Regionale saltato.")

                except Exception as e:
                    print(f"  Errore generale durante preparazione/calcolo FID Regionale: {e}")
                finally:
                    print("  Pulizia cartelle temporanee FID Regionale...")
                    if temp_orig_crop_dir and os.path.isdir(temp_orig_crop_dir):
                        shutil.rmtree(temp_orig_crop_dir, ignore_errors=True)
                    if temp_res_crop_dir and os.path.isdir(temp_res_crop_dir):
                        shutil.rmtree(temp_res_crop_dir, ignore_errors=True)
                    print("  Pulizia completata.")
            else:
                 print("  Calcolo FID saltato (libreria non disponibile).")


            # Stampa medie PSNR/SSIM/LPIPS
            print(f"\nRisultati medi metriche per {folder_name} ({phase}):")
            print(f"  PSNR  | Region: {avg_metrics['PSNR_Region']:.2f} ({num_samples['PSNR_Region']} imgs) | Outside: {avg_metrics['PSNR_Outside']:.2f} ({num_samples['PSNR_Outside']} imgs)" if avg_metrics['PSNR_Region'] is not None else "  PSNR  | Region: N/A | Outside: N/A")
            print(f"  SSIM  | Region: {avg_metrics['SSIM_Region']:.4f} ({num_samples['SSIM_Region']} imgs) | Outside_Avg: {avg_metrics['SSIM_Outside_Avg']:.4f} ({num_samples['SSIM_Outside_Avg']} imgs)" if avg_metrics['SSIM_Region'] is not None else "  SSIM  | Region: N/A | Outside_Avg: N/A")
            # La formattazione qui userà LPIPS_AVAILABLE globale per N/A_Unavailable
            print(f"  LPIPS | Region: {avg_metrics['LPIPS_Region']:.4f} ({num_samples['LPIPS_Region']} imgs) | Full: {avg_metrics['LPIPS_Full']:.4f} ({num_samples['LPIPS_Full']} imgs)" if avg_metrics['LPIPS_Region'] is not None else "  LPIPS | Region: N/A | Full: N/A")
            print("-" * 20)

            # Formatta valori per CSV
            def format_metric(value, precision):
                 if value is None: return 'N/A'
                 return f"{value:.{precision}f}"

            # Aggiungi risultati alla lista per il CSV
            result_data = {
                'Folder': folder_name, 'Phase': phase,
                'PSNR_Region': format_metric(avg_metrics['PSNR_Region'], 2),
                'PSNR_Outside': format_metric(avg_metrics['PSNR_Outside'], 2),
                'SSIM_Region': format_metric(avg_metrics['SSIM_Region'], 4),
                'SSIM_Outside_Avg': format_metric(avg_metrics['SSIM_Outside_Avg'], 4),
                # Qui usiamo la LPIPS_AVAILABLE globale per decidere il tipo di N/A
                'LPIPS_Region': format_metric(avg_metrics['LPIPS_Region'], 4) if LPIPS_AVAILABLE else 'N/A_Unavailable',
                'LPIPS_Full': format_metric(avg_metrics['LPIPS_Full'], 4) if LPIPS_AVAILABLE else 'N/A_Unavailable',
                 # Usa FID_AVAILABLE globale
                'FID_Full': format_metric(fid_full_value, 4) if FID_AVAILABLE else 'N/A_Unavailable',
                'FID_Region': format_metric(fid_region_value, 4) if FID_AVAILABLE else 'N/A_Unavailable',
                'Num_Images_Processed': len(common_files)
            }
            for key, count in num_samples.items(): result_data[f'Num_Imgs_{key}'] = count
            results_list.append(result_data)


    # --- Salva Risultati su CSV ---
    # (Come prima)
    if not results_list:
        print("\nNessun risultato valido calcolato. Il file CSV non verrà creato.")
    else:
        try:
            df = pd.DataFrame(results_list)
            column_order = [
                'Folder', 'Phase',
                'PSNR_Region', 'PSNR_Outside',
                'SSIM_Region', 'SSIM_Outside_Avg',
                'LPIPS_Region', 'LPIPS_Full',
                'FID_Full', 'FID_Region',
                'Num_Images_Processed',
                'Num_Imgs_PSNR_Region', 'Num_Imgs_PSNR_Outside',
                'Num_Imgs_SSIM_Region', 'Num_Imgs_SSIM_Outside_Avg',
                'Num_Imgs_LPIPS_Region', 'Num_Imgs_LPIPS_Full',
            ]
            existing_cols = [col for col in column_order if col in df.columns]
            df = df[existing_cols]
            print(f"\nSalvataggio risultati in: {OUTPUT_CSV}")
            df.to_csv(OUTPUT_CSV, index=False, mode='w', header=True)
            print("Salvataggio CSV completato.")
        except Exception as e:
            print(f"ERRORE durante il salvataggio del file CSV ({OUTPUT_CSV}): {e}")

    print("\n" + "="*60)
    print(f"CALCOLO METRICHE COMPLETATO (Inside vs Outside + FID Regionale)")
    print("="*60 + "\n")

# --- Blocco di Esecuzione Principale ---
# (Nessuna modifica necessaria qui)
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'Non impostato')}")
    essential_paths = {"GT": TEST_SET_DIR, "Pre": RESULTS_PRE_TUNING_BASE, "Post": RESULTS_POST_TUNING_BASE}
    abort = False
    for name, path in essential_paths.items():
        if not os.path.isdir(path): print(f"ERRORE: Path {name} non trovato: '{path}'"); abort=True
    if abort: exit(1)
    main()
