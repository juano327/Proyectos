# config.py
import os

# Ajustá estas rutas a tu entorno:
SOURCE_DIR = r"C:\Users\gabri\OneDrive\Escritorio\Juano Uncuyo\IA\Proyecto Final\base de datos"
DEST_DIR   = r"C:\Users\gabri\OneDrive\Escritorio\Juano Uncuyo\IA\Proyecto Final\imagenes procesadas"
ARTIF_DIR  = r"C:\Users\gabri\OneDrive\Escritorio\Juano Uncuyo\IA\Proyecto Final\modelos"

AUDIO_DIR       = r"C:\Users\gabri\OneDrive\Escritorio\Juano Uncuyo\IA\Proyecto Final\Audios"     # carpeta con subcarpetas por clase
AUDIO_ARTIF_DIR = r"C:\Users\gabri\OneDrive\Escritorio\Juano Uncuyo\IA\Proyecto Final\modelos"  # donde guardamos mu/sigma y dataset escalado
AUDIO_K         = 4   # k por defecto para KNN de audio
# --- Audio (preproc) ---
AUDIO_SPLIT_TOP_DB = 25     # 20–30 típico; más alto = quita más silencio/ruido
AUDIO_TARGET_DUR   = 1.0    # segundos tras quitar silencios (0.8–1.2 es buen rango)
AUDIO_BAND_LOW     = 80     # Hz
AUDIO_BAND_HIGH    = 6500   # Hz (bajá a 5500 si hay “s” muy fuertes)

# Parámetros comunes del pipeline
OUTPUT_SIZE = 256
HUE_BINS    = 3   # usá el mismo valor en todo el proyecto
V_BINS      = 8
#que son los hue_bins y v_bins? 
# V_BINS es el número de bins para el canal V (valor) en HSV, usado en procesamiento de imágenes.
# Hue_bins es el número de bins para el canal H (matiz) en HSV, usado para cuantizar colores.
# Estos parámetros afectan cómo se extraen las características de las imágenes.