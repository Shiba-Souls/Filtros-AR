import cv2
import numpy as np


def portal_width(p1, p2, p3, p4):
    top_w = np.hypot(p3[0] - p1[0], p3[1] - p1[1])
    bottom_w = np.hypot(p4[0] - p2[0], p4[1] - p2[1])
    return (top_w + bottom_w) / 2.0


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
        rotation_speed=3.0,      # grados que gira por frame
        min_corners=3,
        max_corners=9,
        corner_change_interval=12,  # frames entre cada cambio de nro. de esquinas
        radius_jitter=0.15,         # variacion aleatoria del radio por vertice (0-1)
        pulse_amplitude=0.08,       # variacion global de tamano tipo "latido"
        pulse_speed=0.15,           # velocidad del latido
    ):
        self.rotation_speed = rotation_speed
        self.min_corners = max(3, min_corners)
        self.max_corners = max(self.min_corners, max_corners)
        self.corner_change_interval = max(1, corner_change_interval)
        self.radius_jitter = radius_jitter
        self.pulse_amplitude = pulse_amplitude
        self.pulse_speed = pulse_speed

        self.angle_deg = 0.0
        self.frame_count = 0
        self.current_corners = self.max_corners
        self._rng = np.random.default_rng()

        self._base_rotation_speed = rotation_speed
        self._base_corner_change_interval = self.corner_change_interval
        self._base_radius_jitter = radius_jitter
        self.speed_level = 3

    def set_speed_level(self, level):
        # Nivel 1 (lento/casi estable) a 5 (frenetico)
        multipliers = {1: 0.35, 2: 0.65, 3: 1.0, 4: 1.6, 5: 2.4}
        mult = multipliers.get(level, 1.0)
        self.speed_level = level
        self.rotation_speed = self._base_rotation_speed * mult
        # Mas velocidad = cambia de esquinas mas seguido (intervalo mas chico)
        self.corner_change_interval = max(1, int(round(self._base_corner_change_interval / mult)))
        self.radius_jitter = min(0.9, self._base_radius_jitter * mult)

    def reset(self):
        self.angle_deg = 0.0
        self.frame_count = 0
        self.current_corners = self.max_corners

    def _maybe_change_corners(self):
        if self.frame_count % self.corner_change_interval == 0:
            self.current_corners = int(
                self._rng.integers(self.min_corners, self.max_corners + 1)
            )

    def animate(self, p1, p2, p3, p4):
        """Toma los 4 puntos base rastreados (dedos) y devuelve un array
        (N, 2) int32 con un poligono irregular, rotado y con N variable,
        centrado y escalado segun ese mismo poligono base.
        """
        base_pts = np.array([p1, p3, p4, p2], dtype=np.float32)
        center = base_pts.mean(axis=0)
        avg_radius = float(np.mean(np.linalg.norm(base_pts - center, axis=1)))
        avg_radius = max(avg_radius, 1.0)

        self.frame_count += 1
        self._maybe_change_corners()
        self.angle_deg = (self.angle_deg + self.rotation_speed) % 360.0

        n = self.current_corners
        base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        angles = np.deg2rad(self.angle_deg) + base_angles

        pulse = 1.0 + self.pulse_amplitude * np.sin(self.frame_count * self.pulse_speed)
        jitter = 1.0 + self._rng.uniform(-self.radius_jitter, self.radius_jitter, size=n)
        radii = avg_radius * pulse * jitter

        pts = np.stack(
            [center[0] + radii * np.cos(angles), center[1] + radii * np.sin(angles)],
            axis=1,
        )
        return pts.astype(np.int32)


def render_portal(frame, p1, p2, p3, p4, filtro_func, scale_factor=1.0, animator=None):
    """Dibuja el portal. Si se pasa un `animator` (UnstablePortalAnimator),
    el poligono se dibuja girando y con numero de esquinas variable en vez
    del cuadrilatero fijo definido por p1..p4.
    """
    if animator is not None:
        full_polygon = animator.animate(p1, p2, p3, p4)
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

    cv2.polylines(frame, [full_polygon], isClosed=True, color=(255, 255, 255), thickness=2)
    return frame