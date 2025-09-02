# main.py
# ------------------------------------------------------------
# GUI para:
# - IMÁGENES: entrenar / clasificar (con PCA opcional de la imagen nueva)
# - AUDIO: entrenar KNN y graficar LDA del dataset (SIN graficar audio nuevo)
# - Grabación manual START/STOP con filtro "sin voz, no predigo"
# ------------------------------------------------------------

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import sounddevice as sd
import soundfile as sf

# ---------------- Config ----------------
from config import SOURCE_DIR, DEST_DIR, ARTIF_DIR
from config import AUDIO_DIR, AUDIO_ARTIF_DIR, AUDIO_K

# Umbrales anti-silencio (opcionales; podés moverlos a config.py)
try:
    from config import AUDIO_MIN_VOICED_DUR
except Exception:
    AUDIO_MIN_VOICED_DUR = 0.25
try:
    from config import AUDIO_MIN_RMS
except Exception:
    AUDIO_MIN_RMS = 0.01

# -------------- Pipelines --------------
from procesamiento2 import train_from_directory, classify_new_image
from audio_pipeline import (
    train_audio_knn,
    predict_audio_file,
    plot_audio_lda_dataset,   # << mantenemos LDA del dataset al entrenar
    has_speech,
)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Proyecto Visión + Audio - Clasificación")
        self.geometry("1000x680")

        # ---------- Estado de grabación ----------
        self._audio_stream = None
        self._audio_frames = []
        self._audio_sr = 44100

        # ---------- Barra superior ----------
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # --- IMAGEN ---
        self.btn_train = ttk.Button(top, text="Entrenar (Imágenes)", command=self.on_train_images)
        self.btn_train.pack(side=tk.LEFT, padx=5)

        self.btn_classify = ttk.Button(top, text="Clasificar imagen…", command=self.on_classify_image)
        self.btn_classify.pack(side=tk.LEFT, padx=5)

        # toggle para mostrar/ocultar PCA al clasificar imagen
        self.var_plot = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="PCA imagen al clasificar", variable=self.var_plot).pack(side=tk.LEFT, padx=14)

        # --- AUDIO ---
        self.btn_train_audio = ttk.Button(top, text="Entrenar Audio (KNN)", command=self.on_train_audio)
        self.btn_train_audio.pack(side=tk.LEFT, padx=12)

        self.btn_pred_audio_file = ttk.Button(top, text="Clasificar audio…", command=self.on_pred_audio_file)
        self.btn_pred_audio_file.pack(side=tk.LEFT, padx=5)

        # Grabación manual
        self.btn_rec_start = ttk.Button(top, text="Comenzar grabación", command=self.on_rec_start)
        self.btn_rec_start.pack(side=tk.LEFT, padx=12)

        self.btn_rec_stop_pred = ttk.Button(
            top, text="Detener y predecir", command=self.on_rec_stop_and_predict, state=tk.DISABLED
        )
        self.btn_rec_stop_pred.pack(side=tk.LEFT, padx=5)

        # ---------- Info rutas ----------
        info = ttk.LabelFrame(self, text="Rutas y parámetros")
        info.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        ttk.Label(info, text=f"SOURCE_DIR     : {SOURCE_DIR}").pack(anchor="w")
        ttk.Label(info, text=f"DEST_DIR       : {DEST_DIR}").pack(anchor="w")
        ttk.Label(info, text=f"ARTIF_DIR      : {ARTIF_DIR}").pack(anchor="w")
        ttk.Label(info, text=f"AUDIO_DIR      : {AUDIO_DIR}").pack(anchor="w")
        ttk.Label(info, text=f"AUDIO_ARTIF_DIR: {AUDIO_ARTIF_DIR}").pack(anchor="w")
        ttk.Label(info, text=f"AUDIO_K={AUDIO_K}").pack(anchor="w")

        # ---------- Log ----------
        logf = ttk.LabelFrame(self, text="Log")
        logf.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.txt = tk.Text(logf, height=20, wrap="word")
        self.txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scr = ttk.Scrollbar(logf, command=self.txt.yview)
        scr.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt.configure(yscrollcommand=scr.set)

    # -------- Utils GUI --------
    def log(self, msg: str):
        self.txt.insert(tk.END, str(msg) + "\n")
        self.txt.see(tk.END)
        self.update_idletasks()

    def _set_buttons_state(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        for b in (
            self.btn_train,
            self.btn_classify,
            self.btn_train_audio,
            self.btn_pred_audio_file,
            self.btn_rec_start,
            self.btn_rec_stop_pred,
        ):
            b.configure(state=state)

    # --------------- IMÁGENES ---------------
    def on_train_images(self):
        def _job():
            try:
                self._set_buttons_state(False)
                self.log(">> Entrenando pipeline de IMÁGENES…")
                train_from_directory(SOURCE_DIR, DEST_DIR, ARTIF_DIR)
                self.log(">> Entrenamiento de imágenes COMPLETADO.")
                self.after(0, lambda: messagebox.showinfo("Entrenamiento imágenes", f"Listo. Artefactos en:\n{ARTIF_DIR}"))
            except Exception as e:
                self.log(f"[ERROR] {e}")
                self.after(0, lambda: messagebox.showerror("Error entrenamiento imágenes", str(e)))
            finally:
                self._set_buttons_state(True)

        threading.Thread(target=_job, daemon=True).start()

    def on_classify_image(self):
        path = filedialog.askopenfilename(
            title="Elegí una imagen",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.heic *.heif *.bmp *.tif *.tiff")],
        )
        if not path:
            return

        def _job():
            try:
                self._set_buttons_state(False)
                self.log(f">> Clasificando imagen: {path}")

                # 1) Predigo SIN plot (si el pipeline lo permite)
                try:
                    pred = classify_new_image(path, ARTIF_DIR, DEST_DIR, plot=False)
                except TypeError:
                    try:
                        pred = classify_new_image(path, ARTIF_DIR, DEST_DIR, False)
                    except TypeError:
                        # no acepta flag de plot -> predigo normal
                        pred = classify_new_image(path, ARTIF_DIR, DEST_DIR)

                self.log(f">> Predicción (imagen): {pred}")

                # 2) Dibujo el PCA en el hilo principal
                def _plot_and_msg():
                    try:
                        try:
                            classify_new_image(path, ARTIF_DIR, DEST_DIR, plot=True)
                        except TypeError:
                            try:
                                classify_new_image(path, ARTIF_DIR, DEST_DIR, True)
                            except TypeError:
                                # si tu función no dibuja, no hacemos nada
                                pass
                    finally:
                        messagebox.showinfo("Predicción imagen", f"Clase predicha: {pred}")

                self.after(0, _plot_and_msg)

            except Exception as e:
                self.log(f"[ERROR] {e}")
                self.after(0, lambda: messagebox.showerror("Error de clasificación", str(e)))
            finally:
                self._set_buttons_state(True)

        threading.Thread(target=_job, daemon=True).start()


    # --------------- AUDIO ---------------
    def on_train_audio(self):
        def _job():
            try:
                self._set_buttons_state(False)
                self.log(">> Entrenando KNN de AUDIO…")
                acc, clases = train_audio_knn(
                    audio_dir=AUDIO_DIR,
                    artifact_dir=AUDIO_ARTIF_DIR,
                    k=AUDIO_K,
                    do_pca_plot=False,  # no dibujar PCA interno
                    pca_components=2,
                )
                self.log(f">> Audio listo. LOO: {acc*100:.1f}% | Clases: {list(clases)}")

                # Graficar LDA del dataset en hilo principal
                self.after(0, lambda: plot_audio_lda_dataset(AUDIO_ARTIF_DIR, weighted=True, n_components=2))
                self.after(0, lambda: messagebox.showinfo(
                    "Entrenamiento audio", f"Acierto LOO: {acc*100:.1f}%\nArtefactos: {AUDIO_ARTIF_DIR}"
                ))

            except Exception as e:
                self.log(f"[ERROR] {e}")
                self.after(0, lambda: messagebox.showerror("Error entrenamiento audio", str(e)))
            finally:
                self._set_buttons_state(True)

        threading.Thread(target=_job, daemon=True).start()

    def on_pred_audio_file(self):
        path = filedialog.askopenfilename(
            title="Elegí un archivo de audio",
            filetypes=[("Audio", "*.wav *.flac *.ogg *.mp3 *.m4a")],
        )
        if not path:
            return

        def _job():
            try:
                self._set_buttons_state(False)
                self.log(f">> Clasificando audio: {path}")
                pred = predict_audio_file(path, AUDIO_ARTIF_DIR, k=AUDIO_K)
                self.log(f">> Predicción (audio): {pred}")
                # SIN gráfica del audio nuevo
                self.after(0, lambda: messagebox.showinfo("Predicción audio", f"Palabra predicha: {pred}"))
            except Exception as e:
                self.log(f"[ERROR] {e}")
                self.after(0, lambda: messagebox.showerror("Error predicción audio", str(e)))
            finally:
                self._set_buttons_state(True)

        threading.Thread(target=_job, daemon=True).start()

    # ---- Grabación manual START / STOP ----
    def _audio_callback(self, indata, frames, time, status):
        if status:
            self.log(f"[AUDIO] {status}")  # warnings del driver
        self._audio_frames.append(indata.copy())

    def on_rec_start(self):
        try:
            if self._audio_stream is not None:
                self.log("[WARN] Ya se está grabando.")
                return
            self._audio_frames = []
            self._audio_stream = sd.InputStream(
                samplerate=self._audio_sr, channels=1, dtype="float32", callback=self._audio_callback
            )
            self._audio_stream.start()
            self.btn_rec_start.configure(state=tk.DISABLED)
            self.btn_rec_stop_pred.configure(state=tk.NORMAL)
            self.log(">> GRABANDO… (hablá ahora). Luego presioná 'Detener y predecir'.")
        except Exception as e:
            self.log(f"[ERROR] {e}")
            messagebox.showerror("Error al iniciar grabación", str(e))

    def on_rec_stop_and_predict(self):
        def _job():
            try:
                if self._audio_stream is None:
                    self.log("[WARN] No hay grabación en curso.")
                    return
                self._audio_stream.stop()
                self._audio_stream.close()
                self._audio_stream = None
                self.btn_rec_start.configure(state=tk.NORMAL)
                self.btn_rec_stop_pred.configure(state=tk.DISABLED)

                if not self._audio_frames:
                    self.log("[WARN] Sin datos de audio.")
                    return

                y = np.concatenate(self._audio_frames, axis=0).flatten()
                sr = self._audio_sr

                ok, dur_eff, rms = has_speech(
                    y, sr, min_voiced_dur=AUDIO_MIN_VOICED_DUR, min_rms=AUDIO_MIN_RMS
                )
                self.log(f">> Chequeo voz: ok={ok} | dur_efectiva={dur_eff:.2f}s | rms={rms:.3f}")
                if not ok:
                    self.after(
                        0,
                        lambda: messagebox.showinfo(
                            "Sin voz detectada",
                            f"No se detectó voz suficiente.\nDuración efectiva: {dur_eff:.2f}s | RMS: {rms:.3f}",
                        ),
                    )
                    return

                tmp_path = os.path.join(AUDIO_ARTIF_DIR, "_tmp_rec.wav")
                sf.write(tmp_path, y, sr)
                self.log(">> Prediciendo…")
                pred = predict_audio_file(tmp_path, AUDIO_ARTIF_DIR, k=AUDIO_K)
                self.log(f">> Predicción (mic): {pred}")
                # SIN gráfica del audio nuevo
                self.after(0, lambda: messagebox.showinfo("Predicción mic", f"Palabra predicha: {pred}"))

            except Exception as e:
                self.log(f"[ERROR] {e}\n¿Permisos de mic y drivers OK?")
                self.after(0, lambda: messagebox.showerror("Error micrófono", str(e)))

        threading.Thread(target=_job, daemon=True).start()


# ------------------------ MAIN ------------------------
if __name__ == "__main__":
    os.makedirs(DEST_DIR, exist_ok=True)
    os.makedirs(ARTIF_DIR, exist_ok=True)
    os.makedirs(AUDIO_ARTIF_DIR, exist_ok=True)

    app = App()
    app.mainloop()
