import os
import glob
import random
import math
import cv2 # Importa OpenCV
import sys # Per uscire in caso di errori critici

# --- Configurazione dei Percorsi ---
# Assicurati che questi percorsi siano corretti per il tuo sistema
DATASET_BASE = '/scratch.hpc/giuseppe.spathis//dataset' # La cartella contenente le sottocartelle come bright2dark, context, etc.
# Usa i percorsi assoluti che hai fornito
TEST_SET_BASE = '/scratch.hpc/giuseppe.spathis/colormnet/test_set'
TRAIN_SET_BASE = '/scratch.hpc/giuseppe.spathis/colormnet/train_set'
VAL_SET_BASE = '/scratch.hpc/giuseppe.spathis/colormnet/val_set'

# --- Configurazione Dimensioni Frame ---
TARGET_WIDTH = 832
TARGET_HEIGHT = 480
# --- Fine Configurazione ---

def get_next_subdir_name(set_path):
    """
    Trova il prossimo nome di sottocartella numerica (es. "001", "002")
    all'interno della directory del set specificato.
    """
    # Assicura che la directory base del set esista
    os.makedirs(set_path, exist_ok=True)
    
    # Trova le directory esistenti che corrispondono al pattern "ddd" (tre cifre)
    existing_dirs = glob.glob(os.path.join(set_path, '[0-9][0-9][0-9]'))
    existing_nums = []
    for d in existing_dirs:
        # Verifica che sia una directory e che il nome sia un numero intero
        if os.path.isdir(d): 
            try:
                num = int(os.path.basename(d))
                existing_nums.append(num)
            except ValueError:
                # Ignora le directory con nomi non numerici nel formato atteso
                pass
    
    # Determina il prossimo numero
    if not existing_nums:
        next_num = 1
    else:
        next_num = max(existing_nums) + 1
    
    # Formatta come stringa a 3 cifre con zero iniziali (es. "001", "015", "123")
    return f"{next_num:03d}" 

def process_video(video_path, set_base_path, width, height):
    """
    Elabora un singolo file video: crea una nuova sottocartella numerata nel set di destinazione,
    estrae i frame, li ridimensiona e li salva come PNG.
    """
    video_filename = os.path.basename(video_path)
    print(f"  Processing video: {video_filename} -> {set_base_path}")
    
    # Ottieni il nome della prossima sottocartella (es. "005")
    next_subdir_name = get_next_subdir_name(set_base_path)
    target_dir_path = os.path.join(set_base_path, next_subdir_name)
    
    success = False
    cap = None # Inizializza cap a None
    try:
        # Crea la sottocartella di destinazione (es. .../test_set/005/)
        os.makedirs(target_dir_path, exist_ok=True)
        
        # Apri il file video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"    Error: Impossibile aprire il file video {video_path}")
            return success # Restituisce False

        frame_count = 0
        processed_frame_count = 0
        while True:
            # Leggi un frame
            ret, frame = cap.read()
            
            # Se ret è False, il video è finito o c'è stato un errore
            if not ret:
                break 
            
            frame_count += 1
            
            # Ridimensiona il frame
            try:
                # Usa INTER_AREA per il ridimensionamento, generalmente buono per ridurre
                resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            except Exception as e:
                print(f"    Warning: Errore nel ridimensionare il frame {frame_count-1} da {video_filename}: {e}. Salto questo frame.")
                continue # Salta al prossimo frame

            # Costruisci il nome del file di output per il frame (es. 00000.png, 00001.png)
            frame_filename = os.path.join(target_dir_path, f"{processed_frame_count:05d}.png")
            
            # Salva il frame ridimensionato come PNG
            try:
                cv2.imwrite(frame_filename, resized_frame)
                processed_frame_count += 1
            except Exception as e:
                 print(f"    Warning: Errore nel salvare il frame {processed_frame_count} su {frame_filename}: {e}. Salto questo frame.")
                 continue # Salta al prossimo frame

        # Se siamo qui, il loop è finito (fine video o break anticipato)
        if processed_frame_count > 0:
             print(f"    Successo: Estratti e salvati {processed_frame_count} frame in {target_dir_path}")
             success = True
        elif frame_count > 0:
             print(f"    Warning: Letto {frame_count} frame da {video_filename}, ma nessuno è stato salvato con successo in {target_dir_path}.")
        else:
             print(f"    Warning: Nessun frame letto o salvato da {video_filename} in {target_dir_path}.")


    except Exception as e:
        print(f"    Errore generale durante l'elaborazione di {video_filename} in {target_dir_path}: {e}")
        success = False

    finally:
        # Assicurati di rilasciare la risorsa video, anche se ci sono stati errori
        if cap is not None and cap.isOpened():
            cap.release()
            
    return success # Restituisce True se almeno un frame è stato salvato, False altrimenti

# --- Logica Principale dello Script ---
print("Avvio dello script per dividere ed elaborare il dataset...")
print(f"Directory dataset sorgente: {DATASET_BASE}")
print(f"Directory Test set destinazione: {TEST_SET_BASE}")
print(f"Directory Train set destinazione: {TRAIN_SET_BASE}")
print(f"Directory Validation set destinazione: {VAL_SET_BASE}")
print(f"Dimensioni target frame: {TARGET_WIDTH}x{TARGET_HEIGHT}")

# Assicurati che le directory di destinazione base esistano
# (Anche get_next_subdir_name lo fa, ma è buona norma essere espliciti)
os.makedirs(TEST_SET_BASE, exist_ok=True)
os.makedirs(TRAIN_SET_BASE, exist_ok=True)
os.makedirs(VAL_SET_BASE, exist_ok=True)

# Ottieni l'elenco delle sottocartelle nella directory base del dataset
try:
    # Lista solo le directory effettive
    source_subdirs = [d for d in os.listdir(DATASET_BASE) if os.path.isdir(os.path.join(DATASET_BASE, d))]
except FileNotFoundError:
    print(f"Errore critico: La directory dataset sorgente '{DATASET_BASE}' non è stata trovata.")
    print("Script terminato.")
    sys.exit(1) # Esce dallo script con un codice di errore

if not source_subdirs:
    print(f"Attenzione: Nessuna sottocartella trovata in '{DATASET_BASE}'.")
    print("Script terminato.")
    sys.exit(0) # Esce normalmente se non c'è nulla da fare

# Elabora ogni sottocartella sorgente trovata
total_videos_processed = 0
total_errors = 0

for subdir_name in source_subdirs:
    current_source_path = os.path.join(DATASET_BASE, subdir_name)
    print(f"\nElaborazione sottocartella sorgente: {current_source_path}")

    # Trova tutti i file .mp4 nella sottocartella corrente
    # Usa recursive=False se non vuoi cercare in eventuali sottocartelle dentro subdir_name
    video_files = glob.glob(os.path.join(current_source_path, '*.mp4')) 
    
    n = len(video_files)
    print(f"Trovati {n} file video (.mp4).")

    if n == 0:
        print("  Nessun file video .mp4 trovato, salto questa sottocartella.")
        continue
        
    # Mescola l'elenco dei file per un'assegnazione casuale
    random.shuffle(video_files)

    # Determina il numero di file per ciascun set in base a 'n'
    num_test, num_train, num_val = 0, 0, 0
    
    if n == 1:
        print("  Attenzione: Trovato solo 1 video. Assegnato a Train Set.")
        num_train = 1
    elif n == 2:
        print("  Attenzione: Trovati solo 2 video. Assegnati 1 a Train Set e 1 a Validation Set.")
        num_train = 1
        num_val = 1 
    elif n == 3:
        print("  Trovati 3 video. Assegnati 1 a Test, 1 a Train, 1 a Validation Set.")
        num_test = 1
        num_train = 1
        num_val = 1
    else: # n > 3
        # Applica la logica: minimo 1 per set, poi distribuisci il resto
        num_test = 1
        num_train = 1
        num_val = 1
        remaining = n - 3
        
        # Distribuisci il rimanente: ~50% test, ~40% train, ~10% val del *rimanente*
        # Calcola gli addizionali arrotondando, assicurati che la somma sia corretta
        add_test = round(remaining * 0.5)
        add_train = round(remaining * 0.4)
        # Il resto va a validation per far quadrare i conti
        add_val = remaining - add_test - add_train 
        
        # Somma i minimi agli addizionali calcolati
        num_test += add_test
        num_train += add_train
        num_val += add_val
        
        print(f"  Assegnazione per {n} video: {num_test} a Test, {num_train} a Train, {num_val} a Validation.")

    # Controlla che la somma faccia ancora 'n' (debug)
    if num_test + num_train + num_val != n:
         print(f"  ERRORE INTERNO: La somma delle assegnazioni ({num_test+num_train+num_val}) non corrisponde a n ({n}). Controllare la logica di divisione.")
         # Potresti voler fermare lo script qui o solo segnalarlo
         total_errors += n # Considera tutti i video di questa cartella come errori
         continue # Salta l'elaborazione di questa cartella

    # Esegui l'assegnazione e l'elaborazione
    files_assigned_count = 0
    successful_processing_count = 0
    
    # 1. Assegna e processa per Test Set
    print(f"  -> Assegnazione {num_test} video a Test Set...")
    for i in range(num_test):
        video_to_process = video_files[files_assigned_count]
        if process_video(video_to_process, TEST_SET_BASE, TARGET_WIDTH, TARGET_HEIGHT):
            successful_processing_count += 1
        files_assigned_count += 1

    # 2. Assegna e processa per Train Set
    print(f"  -> Assegnazione {num_train} video a Train Set...")
    for i in range(num_train):
        video_to_process = video_files[files_assigned_count]
        if process_video(video_to_process, TRAIN_SET_BASE, TARGET_WIDTH, TARGET_HEIGHT):
             successful_processing_count += 1
        files_assigned_count += 1
             
    # 3. Assegna e processa per Validation Set
    print(f"  -> Assegnazione {num_val} video a Validation Set...")
    for i in range(num_val):
        video_to_process = video_files[files_assigned_count]
        if process_video(video_to_process, VAL_SET_BASE, TARGET_WIDTH, TARGET_HEIGHT):
             successful_processing_count += 1
        files_assigned_count += 1

    # Verifica finale per la sottocartella corrente
    print(f"  Elaborazione completata per {subdir_name}: {successful_processing_count}/{n} video elaborati con successo.")
    total_videos_processed += successful_processing_count
    if successful_processing_count < n:
        total_errors += (n - successful_processing_count)


print("\n--- Riepilogo Esecuzione Script ---")
print(f"Elaborazione completata.")
print(f"Sottocartelle sorgente analizzate: {len(source_subdirs)}")
print(f"Video totali elaborati con successo (almeno 1 frame salvato): {total_videos_processed}")
if total_errors > 0:
    print(f"Errori o video non elaborati completamente: {total_errors}")
print("----------------------------------")
