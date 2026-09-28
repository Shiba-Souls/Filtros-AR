"""Captura el audio que esta sonando en la PC (lo que sale por los
parlantes/auriculares) usando el modo "loopback" de WASAPI a traves de
PyAudioWPatch, y lo resume en un par de numeros por frame que geometry.py
usa para modular la animacion del portal "inestable".

Importante: esto NO usa el microfono. WASAPI loopback abre el dispositivo
de RENDER por defecto (el parlante) en modo de captura: lo que se lee es
una copia de lo que ese dispositivo esta reproduciendo, no lo que entra
por ningun microfono. Es exclusivo de Windows (WASAPI), que es la razon
por la que este proyecto depende de "pyaudiowpatch" (el fork de PyAudio
con soporte de loopback) en lugar del PyAudio comun.

El analisis corre en un hilo aparte leyendo el stream en chunks chicos, y
main.py solo lee level/pulse desde el hilo principal cada frame de video
(get_level / consume_pulse), sin bloquearse nunca esperando audio.
"""

import threading
import time

import numpy as np

try:
    import pyaudiowpatch as pyaudio
    PYAUDIO_AVAILABLE = True
except Exception:
    pyaudio = None
    PYAUDIO_AVAILABLE = False


class SystemAudioAnalyzer:
    """Lee el loopback del dispositivo de salida por defecto y expone:

    - get_level(): float 0.0-1.0, volumen suavizado (envolvente RMS) del
      audio del sistema en este instante. Pensado para modular parametros
      continuos (velocidad de rotacion, jitter, amplitud del "latido").
    - consume_pulse(): bool, True una sola vez por cada "golpe" (onset)
      detectado desde la ultima llamada -- un salto brusco de energia
      respecto del promedio reciente, como un beat de musica o un bajonazo
      en un video. Pensado para eventos puntuales (ej. forzar un cambio de
      cantidad de esquinas en el portal). Se consume al leerlo, como una
      bandera, para no disparar el mismo golpe dos veces.

    Si PyAudioWPatch no esta instalado o no se encuentra un dispositivo de
    loopback (ej. no-Windows, o sin salida de audio activa), el analizador
    queda en modo "inerte": get_level() siempre devuelve 0.0 y
    consume_pulse() siempre False, para que main.py pueda usarlo igual sin
    checks especiales y la app no dependa de tener audio para funcionar.
    """

    def __init__(
        self,
        smoothing=0.99,        # 0-1, mas alto = envolvente mas suave/lenta
        pulse_sensitivity=2.3,  # cuantas veces el promedio reciente hace falta para contar como "golpe"
        pulse_cooldown=1.5,    # segundos minimos entre dos golpes consecutivos
    ):
        self.smoothing = smoothing
        self.pulse_sensitivity = pulse_sensitivity
        self.pulse_cooldown = pulse_cooldown

        self._level = 0.0
        self._level_low = 0.0
        self._level_mid = 0.0
        self._level_high = 0.0
        self._pulse_flag = False
        self._running = False
        self._lock = threading.Lock()
        self._thread = None
        self._pa = None
        self._stream = None
        self._last_pulse_time = 0.0
        self._recent_avg = 0.0
        self.available = False
        self.device_name = None
        self._error = None

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def start(self):
        """Intenta abrir el loopback del dispositivo de salida por defecto
        y arranca el hilo de lectura. Si algo falla (no-Windows, no hay
        PyAudioWPatch instalado, no hay dispositivo de salida activo), no
        lanza excepcion: deja el analizador inerte y guarda el motivo en
        self._error para que main.py pueda mostrar un aviso si quiere."""
        if not PYAUDIO_AVAILABLE:
            self._error = "pyaudiowpatch no esta instalado"
            return False

        try:
            self._pa = pyaudio.PyAudio()
            device_info = self._find_loopback_device()
        except Exception as exc:
            self._error = f"No se encontro loopback WASAPI: {exc}"
            if self._pa is not None:
                self._pa.terminate()
                self._pa = None
            return False

        if device_info is None:
            self._error = "No se encontro un dispositivo de loopback WASAPI"
            self._pa.terminate()
            self._pa = None
            return False

        try:
            self.device_name = device_info.get("name", "output device")
            channels = int(device_info.get("maxInputChannels", 2)) or 2
            rate = int(device_info.get("defaultSampleRate", 44100))

            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=device_info["index"],
                frames_per_buffer=1024,
            )
        except Exception as exc:
            self._error = f"No se pudo abrir el stream de loopback: {exc}"
            self._stream = None
            if self._pa is not None:
                self._pa.terminate()
                self._pa = None
            return False

        self._channels = channels
        self._rate = rate
        self._running = True
        self.available = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        return True

    def _find_loopback_device(self):
        """Devuelve el dict de info del dispositivo de loopback WASAPI para
        el altavoz por defecto, o None si no se encuentra ninguno.

        Se intenta primero `get_default_wasapi_loopback()`, el atajo que
        trae PyAudioWPatch para esto; si esa version de la libreria no lo
        tiene (metodo mas nuevo) o falla, se cae al modo manual: ubicar el
        dispositivo de salida por defecto via el host API WASAPI y
        despues buscar, entre todos los dispositivos marcados como
        loopback, el que coincide con ese mismo nombre.
        """
        get_default = getattr(self._pa, "get_default_wasapi_loopback", None)
        if get_default is not None:
            try:
                return get_default()
            except Exception:
                pass  # cae al modo manual de abajo

        wasapi_info = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_output = self._pa.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

        if default_output.get("isLoopbackDevice"):
            return default_output

        for loopback in self._pa.get_loopback_device_info_generator():
            if default_output["name"] in loopback["name"]:
                return loopback

        return None

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None
        self.available = False

    # ------------------------------------------------------------------
    # Lectura (corre en el hilo de audio)
    # ------------------------------------------------------------------

    def _read_loop(self):
        while self._running:
            try:
                raw = self._stream.read(1024, exception_on_overflow=False)
            except Exception:
                # El dispositivo de salida por defecto puede cambiar (el
                # usuario desconecta los auriculares, etc). En vez de matar
                # el hilo, esperamos un toque y reintentamos leer del mismo
                # stream; si sigue fallando, simplemente no actualizamos
                # nivel y el portal vuelve a comportarse como si no hubiera
                # audio hasta que se resuelva.
                time.sleep(0.05)
                continue

            samples = np.frombuffer(raw, dtype=np.int16)
            if samples.size == 0:
                continue
            if self._channels > 1:
                samples = samples.reshape(-1, self._channels).mean(axis=1)

            samples_f = samples.astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(samples_f ** 2))) if samples_f.size else 0.0

            # FFT para separar bandas de frecuencia (bajos, medios, altos)
            fft_vals = np.abs(np.fft.rfft(samples_f))
            fft_freqs = np.fft.rfftfreq(len(samples_f), 1.0 / float(getattr(self, '_rate', 44100)))

            low_mask = (fft_freqs >= 20) & (fft_freqs < 250)
            mid_mask = (fft_freqs >= 250) & (fft_freqs < 4000)
            high_mask = (fft_freqs >= 4000) & (fft_freqs <= 20000)

            rms_low = float(np.mean(fft_vals[low_mask])) if np.any(low_mask) else 0.0
            rms_mid = float(np.mean(fft_vals[mid_mask])) if np.any(mid_mask) else 0.0
            rms_high = float(np.mean(fft_vals[high_mask])) if np.any(high_mask) else 0.0

            with self._lock:
                # Envolvente exponencial: sube/baja suave en vez de saltar
                # de frame a frame con cada chunk de audio.
                self._level = self.smoothing * self._level + (1 - self.smoothing) * min(rms * 4.0, 1.0)
                self._level_low = self.smoothing * self._level_low + (1 - self.smoothing) * min(rms_low * 8.0, 1.0)
                self._level_mid = self.smoothing * self._level_mid + (1 - self.smoothing) * min(rms_mid * 8.0, 1.0)
                self._level_high = self.smoothing * self._level_high + (1 - self.smoothing) * min(rms_high * 8.0, 1.0)

                # Deteccion simple de "golpe": el RMS actual supera por
                # bastante el promedio reciente (que se mueve mas lento
                # todavia que la envolvente de nivel).
                self._recent_avg = 0.95 * self._recent_avg + 0.05 * rms
                now = time.time()
                if (
                    rms > self._recent_avg * self.pulse_sensitivity
                    and rms > 0.02
                    and (now - self._last_pulse_time) > self.pulse_cooldown
                ):
                    self._pulse_flag = True
                    self._last_pulse_time = now

    # ------------------------------------------------------------------
    # Lectura (llamado desde el hilo principal / loop de video)
    # ------------------------------------------------------------------

    def get_level(self):
        """Nivel de audio suavizado, 0.0-1.0. Devuelve 0.0 si el
        analizador esta inerte (sin loopback disponible) o detenido."""
        if not self.available:
            return 0.0
        with self._lock:
            return self._level

    def get_level_low(self):
        if not self.available:
            return 0.0
        with self._lock:
            return self._level_low

    def get_level_mid(self):
        if not self.available:
            return 0.0
        with self._lock:
            return self._level_mid

    def get_level_high(self):
        if not self.available:
            return 0.0
        with self._lock:
            return self._level_high

    def consume_pulse(self):
        """True si hubo un golpe de audio desde la ultima vez que se llamo
        a esto; lo resetea al leerlo (por eso "consume")."""
        if not self.available:
            return False
        with self._lock:
            flag = self._pulse_flag
            self._pulse_flag = False
            return flag
