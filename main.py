import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from hand_tracking import HandTracker, INDEX_TIP, THUMB_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP, WRIST
from body_tracking import BodyTracker, PERSON_COLORS
from geometry import (
    render_portal,
    portal_width,
    ClosingGestureDetector,
    PinchGestureDetector,
    paint_filter_in_polygon,
    UnstablePortalAnimator,
    PersonPortalState,
    group_hands_by_person,
)
from filters import FILTROS
from audio_reactive import SystemAudioAnalyzer

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Que tan lejos (en fraccion del ancho del frame) puede estar una mano de la
# persona mas cercana para seguir considerandose "de esa persona" en modo 2
# personas. Si las dos personas se paran muy juntas y esto da problemas,
# achicar este numero; si quedan manos "huerfanas" (no se les pinta portal)
# porque la persona esta lejos de camara, agrandarlo.
PERSON_HAND_MAX_DISTANCE_RATIO = 0.45


def build_trackers(mode, hand_model_path="hand_landmarker.task",
                    body_model_path="pose_landmarker_lite.task",
                    detection_confidence=0.3):
    """Crea el/los tracker(s) necesarios para el modo pedido.

    Modo 1 persona: igual que siempre, HandTracker buscando hasta 2 manos.
    Modo 2 personas: HandTracker buscando hasta 4 manos (2 personas x 2
    manos) + BodyTracker (PoseLandmarker) para saber, por la posicion del
    torso, cual mano es de cual persona.
    """
    if mode == "2_person":
        hand_tracker = HandTracker(
            model_path=hand_model_path, max_num_hands=4,
            min_detection_confidence=detection_confidence,
        )
        body_tracker = BodyTracker(
            model_path=body_model_path, max_num_people=2,
            min_detection_confidence=0.5,
        )
    else:
        hand_tracker = HandTracker(
            model_path=hand_model_path, max_num_hands=2,
            min_detection_confidence=detection_confidence,
        )
        body_tracker = None
    return hand_tracker, body_tracker


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "No se pudo abrir la camara. Revisa el indice de camara o los permisos."
        )

    # Request HD camera resolution if supported
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "Filters AR Portal"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # --- Modo de tracking: "1_person" (comportamiento original, intacto) o
    # "2_person" (portales independientes con body tracking para no mezclar
    # las manos de las dos personas). Arranca siempre en 1 persona.
    tracking_mode = "1_person"
    hand_tracker, body_tracker = build_trackers(tracking_mode)

    # --- Estado modo 1 persona (identico a como ya funcionaba) ---
    filtro_index = 0
    portal_text = ""
    text_input_active = False
    closing_detector = ClosingGestureDetector()
    pinch_detector = PinchGestureDetector()
    portal_animator = UnstablePortalAnimator()

    sensitivity_level = 3  # Range: 1 (Low) to 5 (Ultra)
    closing_detector.set_sensitivity(sensitivity_level)
    pinch_detector.set_sensitivity(sensitivity_level)

    # --- Estado modo 2 personas: una PersonPortalState por persona (filtro,
    # detectores de gestos y animador propios, para que cada quien controle
    # su portal de forma independiente) ---
    person_states = [
        PersonPortalState(len(FILTROS), sensitivity_level)
        for _ in range(2)
    ]

    # --- Audio del sistema (lo que suena por los parlantes, sin pasar por
    # el microfono) para modular la animacion "inestable". Si no hay
    # loopback disponible (falta pyaudiowpatch, no es Windows, no hay
    # dispositivo de salida activo), el analizador queda inerte y
    # get_level()/consume_pulse() devuelven siempre 0.0/False, asi que el
    # resto del codigo no necesita ramas especiales para ese caso.
    audio_analyzer = SystemAudioAnalyzer()
    audio_started = audio_analyzer.start()
    audio_reactive_enabled = audio_started  # arranca activo solo si se pudo abrir el loopback
    portal_animator.set_audio_reactive(audio_reactive_enabled)
    for s in person_states:
        s.set_audio_reactive(audio_reactive_enabled)

    # Settings Toggles
    gestures_enabled = True         # Master toggle for all gestures
    pinch_gesture_enabled = True    # Toggle for single-hand pinch gesture
    clap_gesture_enabled = True     # Toggle for two-hand clap/closing gesture
    draw_skeleton_enabled = True    # Toggle for drawing hand skeleton dots/lines
    draw_body_enabled = True        # Toggle for drawing body/pose skeleton (modo 2 personas)
    show_settings_menu = False      # Toggle for showing on-screen settings menu panel
    unstable_portal_enabled = True  # Toggle for the rotating/unstable-corners animation
    is_3d_mode_enabled = False      # Toggle for 3D multi-portal mode

    is_fullscreen = True
    scale_factor = 1.3  # Default filter size multiplier
    banners = []  # Lista de banners de feedback activos: {"msg": str, "timer": int}

    if not audio_started:
        banners.append({
            "msg": "Audio del sistema no disponible (falta pyaudiowpatch o loopback)",
            "timer": 90,
        })

    def audio_status_text():
        if not audio_started:
            return "N/A (sin loopback)"
        return "ENABLED" if audio_reactive_enabled else "DISABLED"

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Nivel de audio del sistema para este frame (0.0-1.0) y si
            # justo hubo un "golpe" (beat/onset). Se leen una sola vez por
            # frame y se reparten a todos los animadores para que esten
            # sincronizados entre si.
            audio_level = audio_analyzer.get_level() if audio_reactive_enabled else 0.0
            audio_level_low = audio_analyzer.get_level_low() if audio_reactive_enabled else 0.0
            audio_level_mid = audio_analyzer.get_level_mid() if audio_reactive_enabled else 0.0
            audio_level_high = audio_analyzer.get_level_high() if audio_reactive_enabled else 0.0
            audio_pulse = audio_analyzer.consume_pulse() if audio_reactive_enabled else False

            # Process detection
            results = hand_tracker.process_frame(frame)
            pose_result = body_tracker.process_frame(frame) if body_tracker is not None else None

            # Draw skeletons if enabled
            if draw_skeleton_enabled:
                frame = hand_tracker.draw_landmarks(frame, results)
            if tracking_mode == "2_person" and draw_body_enabled and pose_result is not None:
                frame = body_tracker.draw_landmarks(frame, pose_result)

            # ================= MODO 1 PERSONA (sin cambios) =================
            if tracking_mode == "1_person":
                left_hand = None
                right_hand = None
                gesture_triggered = False

                if gestures_enabled and pinch_gesture_enabled and results.hand_landmarks:
                    for hand in results.hand_landmarks:
                        if pinch_detector.update(hand):
                            gesture_triggered = True
                            break

                if results.hand_landmarks and len(results.hand_landmarks) >= 2:
                    sorted_hands = sorted(results.hand_landmarks[:2], key=lambda hand: hand[WRIST].x)
                    left_hand = sorted_hands[0]
                    right_hand = sorted_hands[1]

                if left_hand is not None and right_hand is not None:
                    portal_configs = [(INDEX_TIP, THUMB_TIP)]
                    if is_3d_mode_enabled:
                        portal_configs.append((INDEX_TIP, MIDDLE_TIP))
                        portal_configs.append((MIDDLE_TIP, RING_TIP))
                        portal_configs.append((RING_TIP, PINKY_TIP))

                    active_animator = portal_animator if unstable_portal_enabled else None
                    gesture_triggered_portal_idx = 0 # Use the first portal for gesture detection

                    # Mapping de niveles de audio para cada portal (bateria, bajo, medio, alto)
                    portal_audio_map = [audio_level, audio_level_low, audio_level_mid, audio_level_high]

                    for i, (tip1, tip2) in enumerate(portal_configs):
                        p1 = (left_hand[tip1].x * w, left_hand[tip1].y * h)
                        p2 = (left_hand[tip2].x * w, left_hand[tip2].y * h)
                        p3 = (right_hand[tip1].x * w, right_hand[tip1].y * h)
                        p4 = (right_hand[tip2].x * w, right_hand[tip2].y * h)

                        if i == gesture_triggered_portal_idx:
                            width = portal_width(p1, p2, p3, p4)
                            if gestures_enabled and clap_gesture_enabled and closing_detector.update(width, w):
                                gesture_triggered = True

                        # Obtener nivel de audio específico para este portal
                        current_audio_level = portal_audio_map[i] if i < len(portal_audio_map) else audio_level

                        frame = render_portal(
                            frame, p1, p2, p3, p4, FILTROS[filtro_index],
                            scale_factor=scale_factor, animator=active_animator,
                            audio_level=current_audio_level, audio_pulse=audio_pulse,
                            portal_text=portal_text
                        )

                    unstable_tag = " [INESTABLE]" if unstable_portal_enabled else " [ESTABLE]"
                    cv2.putText(
                        frame,
                        f"Portal Mode: {unstable_tag.strip()}",
                        (20, 40), FONT, 0.8, (0, 255, 0), 2,
                    )
                elif results.hand_landmarks and len(results.hand_landmarks) == 1:
                    single_hand = results.hand_landmarks[0]
                    ix, iy = single_hand[INDEX_TIP].x * w, single_hand[INDEX_TIP].y * h
                    tx, ty = single_hand[THUMB_TIP].x * w, single_hand[THUMB_TIP].y * h
                    cx, cy = (ix + tx) / 2.0, (iy + ty) / 2.0
                    box_w = max(abs(ix - tx) * 2.5 * scale_factor, 140 * scale_factor)
                    box_h = box_w

                    corner_tl = (cx - box_w / 2, cy - box_h / 2)
                    corner_tr = (cx + box_w / 2, cy - box_h / 2)
                    corner_br = (cx + box_w / 2, cy + box_h / 2)
                    corner_bl = (cx - box_w / 2, cy + box_h / 2)

                    active_animator = portal_animator if unstable_portal_enabled else None

                    frame = render_portal(
                        frame, corner_tl, corner_bl, corner_tr, corner_br, FILTROS[filtro_index],
                        scale_factor=scale_factor, animator=active_animator,
                        audio_level=audio_level, audio_pulse=audio_pulse,
                        portal_text=portal_text
                    )

                    unstable_tag = " [INESTABLE]" if unstable_portal_enabled else " [ESTABLE]"
                    cv2.putText(
                        frame,
                        f"Portal Mode: {unstable_tag.strip()}",
                        (20, 40), FONT, 0.8, (0, 255, 0), 2,
                    )
                else:
                    pass

                if gesture_triggered:
                    filtro_index = (filtro_index + 1) % len(FILTROS)

            # ================= MODO 2 PERSONAS =================
            else:
                person_anchors = (
                    body_tracker.get_person_anchors(pose_result, w, h)
                    if pose_result is not None else []
                )

                hand_list = results.hand_landmarks or []
                hand_anchor_points = [(hand[WRIST].x * w, hand[WRIST].y * h) for hand in hand_list]
                max_dist = PERSON_HAND_MAX_DISTANCE_RATIO * w
                assignments = group_hands_by_person(hand_anchor_points, person_anchors, max_dist)

                person_hands = {0: [], 1: []}
                for hand, person_idx in zip(hand_list, assignments):
                    if person_idx is not None and person_idx < 2:
                        person_hands[person_idx].append(hand)

                if not person_anchors:
                    cv2.putText(
                        frame,
                        "Modo 2 Personas: no se detecto a nadie - ubicate frente a la camara",
                        (20, 40), FONT, 0.65, (0, 165, 255), 2,
                    )

                for p_idx in range(2):
                    if p_idx >= len(person_anchors):
                        continue  # esta persona no esta (o no se detecto bien) en este frame

                    state = person_states[p_idx]
                    hands_p = person_hands[p_idx]
                    anchor = person_anchors[p_idx]
                    color = PERSON_COLORS[p_idx % len(PERSON_COLORS)]
                    label_pos = (max(int(anchor[0] - 100), 10), max(int(anchor[1] - 120), 30))

                    gesture_triggered_p = False
                    if gestures_enabled and pinch_gesture_enabled:
                        for hand in hands_p:
                            if state.pinch_detector.update(hand):
                                gesture_triggered_p = True
                                break

                    if len(hands_p) >= 2:
                        two_hands = sorted(hands_p[:2], key=lambda hand: hand[WRIST].x)
                        left_hand, right_hand = two_hands[0], two_hands[1]
                        p1 = (left_hand[INDEX_TIP].x * w, left_hand[INDEX_TIP].y * h)
                        p2 = (left_hand[THUMB_TIP].x * w, left_hand[THUMB_TIP].y * h)
                        p3 = (right_hand[INDEX_TIP].x * w, right_hand[INDEX_TIP].y * h)
                        p4 = (right_hand[THUMB_TIP].x * w, right_hand[THUMB_TIP].y * h)

                        width = portal_width(p1, p2, p3, p4)
                        if gestures_enabled and clap_gesture_enabled and state.closing_detector.update(width, w):
                            gesture_triggered_p = True

                        active_animator = state.portal_animator if unstable_portal_enabled else None
                        frame = render_portal(
                            frame, p1, p2, p3, p4, FILTROS[state.filtro_index],
                            scale_factor=scale_factor, animator=active_animator,
                            audio_level=audio_level, audio_pulse=audio_pulse,
                        )
                        unstable_tag = " [INESTABLE]" if unstable_portal_enabled else ""
                        cv2.putText(
                            frame,
                            f"P{p_idx + 1} - Filter {state.filtro_index + 1}/{len(FILTROS)}{unstable_tag}",
                            label_pos, FONT, 0.6, color, 2,
                        )

                    elif len(hands_p) == 1:
                        single_hand = hands_p[0]
                        ix, iy = single_hand[INDEX_TIP].x * w, single_hand[INDEX_TIP].y * h
                        tx, ty = single_hand[THUMB_TIP].x * w, single_hand[THUMB_TIP].y * h
                        cx, cy = (ix + tx) / 2.0, (iy + ty) / 2.0
                        box_w = max(abs(ix - tx) * 2.5 * scale_factor, 140 * scale_factor)
                        box_h = box_w

                        corner_tl = (cx - box_w / 2, cy - box_h / 2)
                        corner_tr = (cx + box_w / 2, cy - box_h / 2)
                        corner_br = (cx + box_w / 2, cy + box_h / 2)
                        corner_bl = (cx - box_w / 2, cy + box_h / 2)

                        if unstable_portal_enabled:
                            pts = state.portal_animator.animate(
                                corner_tl, corner_bl, corner_tr, corner_br,
                                audio_level=audio_level, audio_pulse=audio_pulse,
                            )
                        else:
                            pts = np.array([corner_tl, corner_tr, corner_br, corner_bl], dtype=np.int32)

                        paint_filter_in_polygon(frame, pts, FILTROS[state.filtro_index])
                        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

                        unstable_tag = " [INESTABLE]" if unstable_portal_enabled else " [ESTABLE]"
                        cv2.putText(
                            frame,
                            f"P{p_idx + 1} Portal: {unstable_tag.strip()}",
                            label_pos, FONT, 0.6, color, 2,
                        )
                    else:
                        pass

                    if gesture_triggered_p:
                        state.next_filter()

            # Draw active feedback banners (stack if more than one triggers at once)
            for i, banner in enumerate(banners):
                y0 = 65 + i * 55
                cv2.rectangle(frame, (w // 2 - 280, y0), (w // 2 + 280, y0 + 50), (0, 200, 0), -1)
                cv2.putText(frame, banner["msg"], (w // 2 - 260, y0 + 35), FONT, 0.7, (255, 255, 255), 2)
                banner["timer"] -= 1
            banners = [b for b in banners if b["timer"] > 0]

            # --- Limpiado el HUD ---
            cv2.putText(
                frame,
                "Menu [S]",
                (20, h - 20), FONT, 0.7, (255, 255, 255), 2,
            )

            # Settings Overlay Menu Panel
            if show_settings_menu:
                menu_box = frame.copy()
                cv2.rectangle(menu_box, (40, 80), (640, 560), (20, 20, 20), -1)
                frame = cv2.addWeighted(menu_box, 0.85, frame, 0.15, 0)
                cv2.rectangle(frame, (40, 80), (640, 560), (0, 255, 255), 2)

                sens_names = {1: "1-Low", 2: "2-Med", 3: "3-High", 4: "4-V.High", 5: "5-Ultra"}
                mode_name = "1 PERSON" if tracking_mode == "1_person" else "2 PERSONS"
                menu_lines = [
                    "--- AR PORTAL SETTINGS ---",
                    f"[M] Tracking Mode:      {mode_name}",
                    f"[G] Master Gestures:     {'ENABLED' if gestures_enabled else 'DISABLED'}",
                    f"[P] Single-Hand Pinch:   {'ENABLED' if pinch_gesture_enabled else 'DISABLED'}",
                    f"[C] Two-Hand Clap:       {'ENABLED' if clap_gesture_enabled else 'DISABLED'}",
                    f"[L] Draw Hand Skeleton:  {'ENABLED' if draw_skeleton_enabled else 'DISABLED'}",
                    f"[B] Draw Body Skeleton:  {'ENABLED' if draw_body_enabled else 'DISABLED'} (modo 2P)",
                    f"[U] Unstable Portal:     {'ENABLED' if unstable_portal_enabled else 'DISABLED'} (velocidad = audio)",
                    f"[3] 3D Mode (Multi):     {'ENABLED' if is_3d_mode_enabled else 'DISABLED'}",
                    f"[R] Edit Portal Text:    {portal_text if portal_text else 'None'}",
                    f"[A] Audio Reactive:      {audio_status_text()}",
                    f"[1-5 or []] Sensitivity: {sens_names[sensitivity_level]}",
                    f"[+/-] Filter Size:       {scale_factor:.1f}x",
                    f"[F] Fullscreen Mode:     {'ON' if is_fullscreen else 'OFF'}",
                    "-----------------------------------",
                    "Press [S] to Close Settings Menu",
                ]
                for idx, line in enumerate(menu_lines):
                    color = (0, 255, 255) if idx == 0 or idx == len(menu_lines) - 1 else (255, 255, 255)
                    cv2.putText(frame, line, (60, 115 + idx * 30), FONT, 0.6, color, 1)

            # Text input handler (when R is active)
            if text_input_active:
                cv2.rectangle(frame, (w // 2 - 200, h // 2 - 40), (w // 2 + 200, h // 2 + 40), (50, 50, 50), -1)
                cv2.rectangle(frame, (w // 2 - 200, h // 2 - 40), (w // 2 + 200, h // 2 + 40), (255, 255, 255), 2)
                cv2.putText(frame, f"Text: {portal_text}", (w // 2 - 180, h // 2 + 10), FONT, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Type to change, Enter to save, R to close", (w // 2 - 180, h // 2 + 30), FONT, 0.4, (200, 200, 200), 1)


            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 255:  # No key pressed
                continue

            if text_input_active:
                if key == 13:  # Enter
                    text_input_active = False
                elif key == 8:  # Backspace
                    portal_text = portal_text[:-1]
                elif 32 <= key <= 126:  # Printable characters
                    portal_text += chr(key)
                elif key == ord("r"):  # R: Toggle text input (close)
                    text_input_active = False
            else:
                if key in (ord("q"), 27):  # Q or ESC
                    break
                elif key in (ord("n"), 32):  # N or Space
                    if tracking_mode == "1_person":
                        filtro_index = (filtro_index + 1) % len(FILTROS)
                    else:
                        for s in person_states:
                            s.next_filter()
                elif key == ord("s"):  # S: Toggle Settings Overlay Menu
                    show_settings_menu = not show_settings_menu
                elif key == ord("g"):  # G: Toggle Master Gestures ON/OFF
                    gestures_enabled = not gestures_enabled
                elif key == ord("p"):  # P: Toggle Pinch Gesture ON/OFF
                    pinch_gesture_enabled = not pinch_gesture_enabled
                elif key == ord("c"):  # C: Toggle Clap Gesture ON/OFF
                    clap_gesture_enabled = not clap_gesture_enabled
                elif key == ord("l"):  # L: Toggle Hand Skeleton drawing ON/OFF
                    draw_skeleton_enabled = not draw_skeleton_enabled
                elif key == ord("b"):  # B: Toggle Body Skeleton drawing ON/OFF (modo 2 personas)
                    draw_body_enabled = not draw_body_enabled
                elif key == ord("m"):  # M: Toggle 1 Person / 2 Person tracking mode
                    new_mode = "2_person" if tracking_mode == "1_person" else "1_person"
                    try:
                        candidate_hand, candidate_body = build_trackers(new_mode)
                    except Exception:
                        candidate_hand, candidate_body = None, None
                        banners.append({
                            "msg": "No se pudo activar 2 Personas: falta pose_landmarker_lite.task",
                            "timer": 60,
                        })

                    if candidate_hand is not None:
                        hand_tracker.close()
                        if body_tracker is not None:
                            body_tracker.close()
                        hand_tracker = candidate_hand
                        body_tracker = candidate_body
                        tracking_mode = new_mode

                        # Evita que un pellizco/clap "a medio hacer" en el modo
                        # anterior dispare un cambio de filtro fantasma apenas
                        # arranca el nuevo modo.
                        closing_detector.is_closed = False
                        pinch_detector.is_pinched = False
                        for s in person_states:
                            s.reset_gestures()
                elif key == ord("u"):  # U: Toggle rotating/unstable-corners portal animation
                    unstable_portal_enabled = not unstable_portal_enabled
                    if unstable_portal_enabled:
                        portal_animator.reset()
                        for s in person_states:
                            s.portal_animator.reset()
                elif key == ord("3"):  # 3: Toggle 3D Mode
                    is_3d_mode_enabled = not is_3d_mode_enabled
                elif key == ord("r"):  # R: Toggle text input
                    text_input_active = not text_input_active
                elif key == ord("a"):  # A: Toggle audio-reactive unstable portal
                    if audio_started:
                        audio_reactive_enabled = not audio_reactive_enabled
                        portal_animator.set_audio_reactive(audio_reactive_enabled)
                        for s in person_states:
                            s.set_audio_reactive(audio_reactive_enabled)
                    else:
                        banners.append({
                            "msg": "Audio del sistema no disponible (falta pyaudiowpatch o loopback)",
                            "timer": 60,
                        })
                elif key in (ord("+"), ord("=")):  # Increase filter size
                    scale_factor = min(scale_factor + 0.2, 4.0)
                elif key in (ord("-"), ord("_")):  # Decrease filter size
                    scale_factor = max(scale_factor - 0.2, 0.4)
                elif key in (ord("]"), ord("}")):  # Increase gesture sensitivity
                    sensitivity_level = min(sensitivity_level + 1, 5)
                    closing_detector.set_sensitivity(sensitivity_level)
                    pinch_detector.set_sensitivity(sensitivity_level)
                    for s in person_states:
                        s.set_sensitivity(sensitivity_level)
                elif key in (ord("["), ord("{")):  # Decrease gesture sensitivity
                    sensitivity_level = max(sensitivity_level - 1, 1)
                    closing_detector.set_sensitivity(sensitivity_level)
                    pinch_detector.set_sensitivity(sensitivity_level)
                    for s in person_states:
                        s.set_sensitivity(sensitivity_level)
                elif ord("1") <= key <= ord("5"):  # Direct level selection 1-5
                    sensitivity_level = key - ord("0")
                    closing_detector.set_sensitivity(sensitivity_level)
                    pinch_detector.set_sensitivity(sensitivity_level)
                    for s in person_states:
                        s.set_sensitivity(sensitivity_level)
                elif key == ord("f"):  # F to toggle fullscreen
                    is_fullscreen = not is_fullscreen
                    prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)
    finally:
        hand_tracker.close()
        if body_tracker is not None:
            body_tracker.close()
        audio_analyzer.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()