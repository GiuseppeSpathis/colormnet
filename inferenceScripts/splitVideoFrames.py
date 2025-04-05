import cv2
import os
import argparse

def extract_frames(video_path, output_folder, width=832, height=480):
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Errore: impossibile aprire il video.")
        return
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (width, height))
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    
    cap.release()
    print(f"Estrazione completata. {frame_count} frame salvati in {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estrai frame da un video e ridimensionali.")
    parser.add_argument("--video", required=True, help="Percorso del file video MP4")
    
    args = parser.parse_args()
    output_path = "/scratch.hpc/giuseppe.spathis/colormnet/input/blackswan"
    extract_frames(args.video, output_path)
