import cv2
import numpy as np


def portal_width(p1, p2, p3, p4):
    top_w = np.hypot(p3[0] - p1[0], p3[1] - p1[1])
    bottom_w = np.hypot(p4[0] - p2[0], p4[1] - p2[1])
    return (top_w + bottom_w) / 2.0


def group_hands_by_person(hand_anchor_points, person_anchor_points, max_distance):
    """Asigna cada mano detectada a la persona mas cercana, en modo 2
    personas, para que las 4 manos en pantalla (2 personas x 2 manos) no se
    mezclen entre si al armar los portales.

    hand_anchor_points: lista de (x, y) en pixeles, un punto por mano
        detectada (tipicamente la muneca de cada mano).
    person_anchor_points: lista de (x, y) en pixeles, un punto por persona
        detectada por BodyTracker (centro entre hombros).
    max_distance: distancia maxima en pixeles entre una mano y una persona
        para aceptar la asignacion. Si la mano mas cercana igual queda mas
        lejos que esto, se descarta esa mano (None) en vez de pegarla a la
        persona equivocada -- mejor perder una mano un frame que bugear el
        portal de otra persona.

    Devuelve: lista del mismo largo que hand_anchor_points, con el indice de
    persona asignado a cada mano (o None si no hay ninguna persona lo
    bastante cerca).
    """
    assignments = []
    for hx, hy in hand_anchor_points:
        best_idx = None
        best_dist = None
        for p_idx, (px, py) in enumerate(person_anchor_points):
            dist = np.hypot(hx - px, hy - py)
            if dist <= max_distance and (best_dist is None or dist < best_dist):
                best_idx = p_idx
                best_dist = dist
        assignments.append(best_idx)
    return assignments


class ClosingGestureDetector:

    def __init__(self, close_ratio=0.18, open_ratio=0.32):
        self.base_close_ratio = close_ratio
        self.base_open_ratio = open_ratio
        self.sensitivity_multiplier = 1.0
        self.is_closed = False

    def set_sensitivity(self, level):
        # Level 1 (low sensitivity) to 5 (ultra sensitivity)
        multipliers = {1: 0.6, 2: 0.8, 3: 1.0, 4: 1.3, 5: 1.6}
        self.sensitivity_multiplier = multipliers.get(level, 1.0)

    def update(self, width, frame_w):
        effective_close = self.base_close_ratio * self.sensitivity_multiplier
        effective_open = self.base_open_ratio * self.sensitivity_multiplier
        close_threshold = effective_close * frame_w
        open_threshold = effective_open * frame_w

        triggered = False
        if not self.is_closed and width < close_threshold:
            self.is_closed = True
            triggered = True
        elif self.is_closed and width > open_threshold:
            self.is_closed = False

        return triggered


class PinchGestureDetector:

    def __init__(self, base_threshold=0.065):
        self.base_threshold = base_threshold
        self.sensitivity_multiplier = 1.0
        self.is_pinched = False

    def set_sensitivity(self, level):
        multipliers = {1: 0.6, 2: 0.8, 3: 1.0, 4: 1.3, 5: 1.6}
        self.sensitivity_multiplier = multipliers.get(level, 1.0)

    def update(self, hand_landmarks):
        t_tip = hand_landmarks[4]  # THUMB_TIP
        i_tip = hand_landmarks[8]  # INDEX_TIP
        dist = np.hypot(t_tip.x - i_tip.x, t_tip.y - i_tip.y)

        effective_threshold = self.base_threshold * self.sensitivity_multiplier
        triggered = False
        if not self.is_pinched and dist < effective_threshold:
            self.is_pinched = True
            triggered = True
        elif self.is_pinched and dist > effective_threshold * 1.7:
            self.is_pinched = False

        return triggered


def paint_filter_in_polygon(frame, polygon_pts, filtro_func):
    h, w = frame.shape[:2]

    x, y, bw, bh = cv2.boundingRect(polygon_pts)
    x, y = max(x, 0), max(y, 0)
    bw = min(bw, w - x)
    bh = min(bh, h - y)
    if bw <= 1 or bh <= 1:
        return frame

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_pts], 255)
    mask_roi = mask[y : y + bh, x : x + bw]

    roi = frame[y : y + bh, x : x + bw]
    filtered_roi = filtro_func(roi)

    mask_3ch = cv2.cvtColor(mask_roi, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    blended = (filtered_roi * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)
    frame[y : y + bh, x : x + bw] = blended
    return frame


class UnstablePortalAnimator:
    """Mantiene el estado de una animacion 'inestable': el poligono gira
    continuamente y su cantidad de vertices (esquinas) aparece/desaparece
    de forma periodica, dando la sensacion de un objeto que no logra
    mantener una forma fija.
    """

    def __init__(
        self,
        min_corners=3,
        max_corners=9,
        pulse_speed=0.15,           # velocidad del latido
        # --- Reactividad al audio del sistema (ver audio_reactive.py) ---
        # El modo inestable ahora se rige por completo por audio_level: sin
        # sonido el portal queda practicamente quieto (near-static), y toda
        # la rotacion/temblor/frecuencia de cambio de esquinas crece con el
        # volumen. Ya no existe un nivel de velocidad manual: audio_level
        # 0.0-1.0 es la unica fuente de "velocidad" del portal.
        audio_reactive=False,           # si False, animate() ignora audio_level/audio_pulse por completo
        idle_rotation_speed=0.15,       # grados/frame con audio_level=0 (casi estatico, no muerto del todo)
        audio_rotation_boost=9.0,       # grados/frame extra de rotacion a audio_level maximo (1.0)
        idle_corner_change_interval=90,  # frames entre cambios de esquinas con audio_level=0 (lento)
        min_corner_change_interval=3,    # frames entre cambios de esquinas a audio_level maximo (rapido)
        idle_radius_jitter=0.02,        # jitter casi nulo con audio_level=0
        audio_jitter_boost=0.55,        # jitter extra (0-1) a audio_level maximo
        pulse_amplitude=0.02,           # "latido" base casi nulo sin audio
        audio_pulse_boost=0.35,         # amplitud de "latido" extra a audio_level maximo
        audio_corner_pulse_boost=6,     # esquinas extra (encima de max_corners) que un "golpe" de audio puede agregar por un instante
    ):
        self.min_corners = max(3, min_corners)
        self.max_corners = max(self.min_corners, max_corners)
        self.pulse_speed = pulse_speed

        self.angle_deg = 0.0
        self.frame_count = 0
        self.current_corners = self.max_corners
        self._rng = np.random.default_rng()

        self.audio_reactive = audio_reactive
        self.idle_rotation_speed = idle_rotation_speed
        self.audio_rotation_boost = audio_rotation_boost
        self.idle_corner_change_interval = max(1, idle_corner_change_interval)
        self.min_corner_change_interval = max(1, min_corner_change_interval)
        self.idle_radius_jitter = idle_radius_jitter
        self.audio_jitter_boost = audio_jitter_boost
        self.pulse_amplitude = pulse_amplitude
        self.audio_pulse_boost = audio_pulse_boost
        self.audio_corner_pulse_boost = audio_corner_pulse_boost
        # Nivel de audio suavizado que se ve en el proximo animate(), para
        # que un "golpe" puntual (consume_pulse) pueda sumar unas cuantas
        # esquinas de mas por un ratito en vez de solo por un frame.
        self._pulse_corner_frames_left = 0

    def reset(self):
        self.angle_deg = 0.0
        self.frame_count = 0
        self.current_corners = self.max_corners
        self._pulse_corner_frames_left = 0

    def set_audio_reactive(self, enabled):
        self.audio_reactive = enabled
        if not enabled:
            self._pulse_corner_frames_left = 0

    def _maybe_change_corners(self, corner_change_interval, extra_corners=0):
        if self.frame_count % corner_change_interval == 0:
            self.current_corners = int(
                self._rng.integers(self.min_corners, self.max_corners + 1)
            )
        if extra_corners > 0:
            # Un "golpe" de audio agrega esquinas de mas por unos frames,
            # sin pisar el ciclo normal de cambio de esquinas de arriba.
            self.current_corners = min(self.current_corners + extra_corners, 24)

    def animate(self, p1, p2, p3, p4, audio_level=0.0, audio_pulse=False):
        """Toma los 4 puntos base rastreados (dedos) y devuelve un array
        (N, 2) int32 con un poligono irregular, rotado y con N variable,
        centrado y escalado segun ese mismo poligono base.

        audio_level: float 0.0-1.0, volumen actual del audio del sistema
        (ver audio_reactive.SystemAudioAnalyzer.get_level). Solo tiene
        efecto si self.audio_reactive es True; a mas volumen, el portal
        gira mas rapido, tiembla mas y late mas fuerte.
        audio_pulse: bool, True si justo ahora se detecto un "golpe" de
        audio (beat/onset). Agrega esquinas extra por un puñado de frames
        para que el portal "salte" visiblemente con el golpe.
        """
        base_pts = np.array([p1, p3, p4, p2], dtype=np.float32)
        center = base_pts.mean(axis=0)
        avg_radius = float(np.mean(np.linalg.norm(base_pts - center, axis=1)))
        avg_radius = max(avg_radius, 1.0)

        level = 0.0
        if self.audio_reactive:
            level = max(0.0, min(1.0, audio_level))
            if audio_pulse:
                self._pulse_corner_frames_left = 6

        extra_corners = 0
        if self._pulse_corner_frames_left > 0:
            extra_corners = self.audio_corner_pulse_boost
            self._pulse_corner_frames_left -= 1

        # Todo lo "inestable" del portal (rotacion, frecuencia de cambio de
        # esquinas, temblor de radio, latido) escala linealmente con el
        # nivel de audio del sistema. Con level=0 el portal queda casi
        # estatico (idle_*); a level=1.0 (musica fuerte) llega a su maxima
        # inestabilidad. Ya no hay un control manual de velocidad.
        effective_rotation_speed = self.idle_rotation_speed + level * self.audio_rotation_boost
        effective_corner_interval = max(
            self.min_corner_change_interval,
            int(round(self.idle_corner_change_interval - level * (
                self.idle_corner_change_interval - self.min_corner_change_interval
            ))),
        )

        self.frame_count += 1
        self._maybe_change_corners(effective_corner_interval, extra_corners=extra_corners)

        self.angle_deg = (self.angle_deg + effective_rotation_speed) % 360.0

        n = self.current_corners
        base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        angles = np.deg2rad(self.angle_deg) + base_angles

        effective_pulse_amplitude = self.pulse_amplitude + level * self.audio_pulse_boost
        pulse = 1.0 + effective_pulse_amplitude * np.sin(self.frame_count * self.pulse_speed)

        effective_jitter = min(0.95, self.idle_radius_jitter + level * self.audio_jitter_boost)
        jitter = 1.0 + self._rng.uniform(-effective_jitter, effective_jitter, size=n)
        radii = avg_radius * pulse * jitter

        pts = np.stack(
            [center[0] + radii * np.cos(angles), center[1] + radii * np.sin(angles)],
            axis=1,
        )
        return pts.astype(np.int32)


class PersonPortalState:
    """Agrupa todo el estado que en modo 1 persona es global (filtro activo,
    detectores de gestos, animador del portal inestable) para que, en modo 2
    personas, cada persona tenga el suyo propio y controle su portal sin
    afectar el del otro.
    """

    def __init__(self, num_filtros, sensitivity_level=3):
        self.filtro_index = 0
        self.num_filtros = num_filtros
        self.closing_detector = ClosingGestureDetector()
        self.pinch_detector = PinchGestureDetector()
        self.portal_animator = UnstablePortalAnimator()
        self.set_sensitivity(sensitivity_level)

    def set_sensitivity(self, level):
        self.closing_detector.set_sensitivity(level)
        self.pinch_detector.set_sensitivity(level)

    def set_audio_reactive(self, enabled):
        self.portal_animator.set_audio_reactive(enabled)

    def next_filter(self):
        self.filtro_index = (self.filtro_index + 1) % self.num_filtros

    def reset_gestures(self):
        """Reinicia los detectores (no el animador) al cambiar de modo, para
        que un pellizco o clap que quedo 'a medias' en el modo anterior no
        dispare un cambio de filtro fantasma en el nuevo modo."""
        self.closing_detector.is_closed = False
        self.pinch_detector.is_pinched = False


def render_portal(frame, p1, p2, p3, p4, filtro_func, scale_factor=1.0, animator=None,
                   audio_level=0.0, audio_pulse=False, portal_color=(255, 255, 255), portal_text=""):
    """Dibuja el portal. Si se pasa un `animator` (UnstablePortalAnimator),
    el poligono se dibuja girando y con numero de esquinas variable en vez
    del cuadrilatero fijo definido por p1..p4. audio_level/audio_pulse se
    pasan directo a animator.animate() (ver ahi el efecto de cada uno).
    """
    if animator is not None:
        full_polygon = animator.animate(p1, p2, p3, p4, audio_level=audio_level, audio_pulse=audio_pulse)
        if scale_factor != 1.0:
            center = full_polygon.mean(axis=0)
            full_polygon = (center + (full_polygon - center) * scale_factor).astype(np.int32)
    else:
        pts = np.array([p1, p3, p4, p2], dtype=np.float32)
        if scale_factor != 1.0:
            center = np.mean(pts, axis=0)
            pts = center + (pts - center) * scale_factor
        full_polygon = pts.astype(np.int32)

    paint_filter_in_polygon(frame, full_polygon, filtro_func)

    cv2.polylines(frame, [full_polygon], isClosed=True, color=portal_color, thickness=2)
    if portal_text:
        center = full_polygon.mean(axis=0).astype(int)
        # Calcular tamaño del portal para escalar el texto
        # Usamos el ancho aproximado del portal para definir un fontScale proporcional
        portal_w = cv2.boundingRect(full_polygon)[2]
        font_scale = max(0.3, portal_w / 200.0)

        # Calcular tamaño del texto para centrarlo correctamente
        text_size = cv2.getTextSize(portal_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        text_x = center[0] - text_size[0] // 2
        text_y = center[1] + text_size[1] // 2

        cv2.putText(frame, portal_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, portal_color, 2)
    return frame