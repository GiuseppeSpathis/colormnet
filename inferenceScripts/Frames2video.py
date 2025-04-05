import cv2
import os
import argparse

def images_to_video(image_folder, output_video, fps=30):
    # Ottieni la lista di immagini ordinate per numero
    images = sorted(
        [img for img in os.listdir(image_folder) if img.endswith(".png")],
        key=lambda x: int(''.join(filter(str.isdigit, x)))  # Ordina per numero
    )
    
    if not images:
        print("Nessuna immagine trovata nella cartella.")
        return
    
    # Leggi la prima immagine per ottenere le dimensioni
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape
    
    # Inizializza il video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Aggiungi le immagini al video
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)
    
    # Rilascia il writer
    video.release()
    print(f"Video salvato come {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True, help="Percorso del file video di output")
    parser.add_argument("--fps", type=int, default=30, help="Frame per secondo (FPS) del video")
    args = parser.parse_args()
    
    image_folder = "/scratch.hpc/giuseppe.spathis/colormnet/result/blackswan/"
    images_to_video(image_folder, args.output, args.fps)

