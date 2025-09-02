# train_main.py
import os
from datetime import datetime

# Lee rutas y parámetros del pipeline
from config import SOURCE_DIR, DEST_DIR, ARTIF_DIR
from config import OUTPUT_SIZE, HUE_BINS
try:
    from config import V_BINS
except ImportError:
    V_BINS = 8

# Funciones de entrenamiento desde tu procesamiento
from procesamiento2 import train_from_directory

def main():
    print("=== ENTRENAMIENTO K-MEANS ===")
    print(f"Fecha/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Origen de datos: {SOURCE_DIR}")
    print(f"Destino de imágenes procesadas: {DEST_DIR}")
    print(f"Carpeta de artefactos: {ARTIF_DIR}")
    print(f"OUTPUT_SIZE={OUTPUT_SIZE} | HUE_BINS={HUE_BINS} | V_BINS={V_BINS}")
    print("init_strategy='seeded' (determinista por clase), k=4\n")

    os.makedirs(DEST_DIR, exist_ok=True)
    os.makedirs(ARTIF_DIR, exist_ok=True)

    acc, clases, paths, _bundle = train_from_directory(
        source_dir=SOURCE_DIR,
        dest_dir=DEST_DIR,
        artifact_dir=ARTIF_DIR,
        mode="flat",
        k=4,                   #clases (berenjena, camote, papa, zanahoria)
        random_state=42,
        init_strategy="seeded",
        max_iter=100,
        tol=1e-6
    )

    print("\n--- RESUMEN ---")
    print(f"Imágenes: {len(paths)}")
    print(f"Clases detectadas: {clases}")
    print(f"Acierto (voto mayoritario): {acc*100:.1f}%")
    print(f"Artefactos guardados en: {ARTIF_DIR}")

if __name__ == "__main__":
    main()
