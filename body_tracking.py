import cv2
import mediapipe as mp

# MediaPipe Tasks aliases (mismo patron que hand_tracking.py)
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Indices de los 33 landmarks del modelo de pose (BlazePose).
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

# Landmarks de torso, brazos y manos para filtrado
VALID_POSE_LANDMARKS = {
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
}

# Conexiones del esqueleto para dibujo, provistas por MediaPipe (mismo
# mecanismo que usa hand_tracking.py con HandLandmarksConnections).
POSE_CONNECTIONS = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS

# Un color por persona (BGR) para diferenciarlas al dibujar el esqueleto.
PERSON_COLORS = [
    (255, 140, 0),   # Persona 1
    (0, 140, 255),   # Persona 2
]

MIN_VISIBILITY = 0.3


def _visible(landmark, threshold=MIN_VISIBILITY):
    vis = landmark.visibility
    return vis is None or vis >= threshold


class BodyTracker:
    """Envuelve PoseLandmarker de MediaPipe Tasks para detectar el torso de
    hasta `max_num_people` personas dentro del cuadro.

    Este tracker no dibuja portales ni maneja gestos: su unico proposito es
    darle a main.py, en modo 2 personas, un punto de referencia estable (el
    centro entre los dos hombros) por cada persona presente en camara. Ese
    punto se usa despues para decidir a que persona pertenece cada mano
    detectada por HandTracker, comparando distancias (ver
    geometry.group_hands_by_person). Sin esta referencia, con 4 manos en
    pantalla (2 personas x 2 manos) no hay forma confiable de saber cuales
    dos manos son de la misma persona.
    """

    def __init__(self, model_path="pose_landmarker_lite.task", max_num_people=2, min_detection_confidence=0.5):
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=max_num_people,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    def process_frame(self, frame_bgr):
        """Convierte el frame BGR al formato de MediaPipe y corre la deteccion
        de poses. Misma convencion que HandTracker.process_frame."""
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.landmarker.detect(mp_image)

    def get_person_anchors(self, result, frame_w, frame_h):
        """Devuelve una lista de puntos (x, y) en pixeles, uno por persona
        detectada con suficiente confianza en ambos hombros. El orden de la
        lista define el "indice de persona" (0 = Persona 1, 1 = Persona 2)
        que se usa en el resto de la app (colores, HUD, estado de portal).

        Si a alguien no se le detectan los hombros con buena visibilidad
        (por ejemplo, esta de espaldas o muy pegado al borde del cuadro), se
        descarta esa persona en ese frame en vez de arriesgarse a asignarle
        manos ajenas.
        """
        anchors = []
        if not result.pose_landmarks:
            return anchors

        for person_landmarks in result.pose_landmarks:
            l_sh = person_landmarks[LEFT_SHOULDER]
            r_sh = person_landmarks[RIGHT_SHOULDER]
            if not (_visible(l_sh) and _visible(r_sh)):
                continue
            cx = (l_sh.x + r_sh.x) / 2.0 * frame_w
            cy = (l_sh.y + r_sh.y) / 2.0 * frame_h
            anchors.append((cx, cy))
        return anchors

    def draw_landmarks(self, frame_bgr, result):
        """Dibuja el esqueleto de cada persona detectada, con un color propio
        por indice de persona para que se puedan distinguir a simple vista."""
        if not result.pose_landmarks:
            return frame_bgr

        h, w, _ = frame_bgr.shape
        for idx, person_landmarks in enumerate(result.pose_landmarks):
            color = PERSON_COLORS[idx % len(PERSON_COLORS)]

            for conn in POSE_CONNECTIONS:
                if conn.start not in VALID_POSE_LANDMARKS or conn.end not in VALID_POSE_LANDMARKS:
                    continue
                pt1 = person_landmarks[conn.start]
                pt2 = person_landmarks[conn.end]
                if not (_visible(pt1) and _visible(pt2)):
                    continue
                x1, y1 = int(pt1.x * w), int(pt1.y * h)
                x2, y2 = int(pt2.x * w), int(pt2.y * h)
                cv2.line(frame_bgr, (x1, y1), (x2, y2), color, 2)

            for idx, lm in enumerate(person_landmarks):
                if idx not in VALID_POSE_LANDMARKS:
                    continue
                if not _visible(lm):
                    continue
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame_bgr, (cx, cy), 3, color, -1)

        return frame_bgr

    def close(self):
        self.landmarker.close()
