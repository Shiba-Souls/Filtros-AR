import cv2
import numpy as np
from hand_tracking import HandTracker, INDEX_TIP, THUMB_TIP, WRIST
from geometry import (
    render_portal,
    portal_width,
    ClosingGestureDetector,
    PinchGestureDetector,
    paint_filter_in_polygon,
    UnstablePortalAnimator,
)
from filters import FILTROS


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

    # High sensitivity hand tracking (min_detection_confidence=0.3)
    tracker = HandTracker(model_path="hand_landmarker.task", min_detection_confidence=0.3)
    filtro_index = 0
    closing_detector = ClosingGestureDetector()
    pinch_detector = PinchGestureDetector()
    portal_animator = UnstablePortalAnimator()

    sensitivity_level = 3  # Range: 1 (Low) to 5 (Ultra)
    closing_detector.set_sensitivity(sensitivity_level)
    pinch_detector.set_sensitivity(sensitivity_level)

    portal_speed_level = 3  # Range: 1 (Slow) to 5 (Frenetic) - rotation/corner-change speed
    portal_animator.set_speed_level(portal_speed_level)

    # Settings Toggles
    gestures_enabled = True         # Master toggle for all gestures
    pinch_gesture_enabled = True    # Toggle for single-hand pinch gesture
    clap_gesture_enabled = True     # Toggle for two-hand clap/closing gesture
    draw_skeleton_enabled = True    # Toggle for drawing hand skeleton dots/lines
    show_settings_menu = False      # Toggle for showing on-screen settings menu panel
    unstable_portal_enabled = True  # Toggle for the rotating/unstable-corners animation

    is_fullscreen = True
    scale_factor = 1.3  # Default filter size multiplier
    gesture_banner_timer = 0  # Frame counter for showing gesture feedback banner
    feedback_msg = ""

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Process detection
            results = tracker.process_frame(frame)

            # Draw hand landmarks if enabled
            if draw_skeleton_enabled:
                frame = tracker.draw_landmarks(frame, results)

            left_hand = None
            right_hand = None
            gesture_triggered = False

            # Check single-hand pinch gesture if enabled
            if gestures_enabled and pinch_gesture_enabled and results.hand_landmarks:
                for hand in results.hand_landmarks:
                    if pinch_detector.update(hand):
                        gesture_triggered = True
                        break

            # Sort detected hands by X-coordinate to reliably get Left & Right hands in screen space
            if results.hand_landmarks and len(results.hand_landmarks) >= 2:
                sorted_hands = sorted(results.hand_landmarks[:2], key=lambda hand: hand[WRIST].x)
                left_hand = sorted_hands[0]
                right_hand = sorted_hands[1]

            if left_hand is not None and right_hand is not None:
                p1 = (left_hand[INDEX_TIP].x * w, left_hand[INDEX_TIP].y * h)
                p2 = (left_hand[THUMB_TIP].x * w, left_hand[THUMB_TIP].y * h)
                p3 = (right_hand[INDEX_TIP].x * w, right_hand[INDEX_TIP].y * h)
                p4 = (right_hand[THUMB_TIP].x * w, right_hand[THUMB_TIP].y * h)

                width = portal_width(p1, p2, p3, p4)

                # Check two-hand clap/closing gesture if enabled
                if gestures_enabled and clap_gesture_enabled and closing_detector.update(width, w):
                    gesture_triggered = True

                active_animator = portal_animator if unstable_portal_enabled else None
                frame = render_portal(
                    frame, p1, p2, p3, p4, FILTROS[filtro_index],
                    scale_factor=scale_factor, animator=active_animator,
                )
                unstable_tag = " [INESTABLE]" if unstable_portal_enabled else ""
                cv2.putText(
                    frame,
                    f"AR Portal Active - Filter {filtro_index + 1}/{len(FILTROS)} (Size: {scale_factor:.1f}x){unstable_tag}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            elif results.hand_landmarks and len(results.hand_landmarks) == 1:
                # 1 Hand detected: render a centered filter box around the hand
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

                if unstable_portal_enabled:
                    # render_portal expects p1..p4 as (left-index, left-thumb,
                    # right-index, right-thumb); reusar el mismo orden con las
                    # 4 esquinas del recuadro alcanza para centrar/escalar el
                    # poligono inestable.
                    pts = portal_animator.animate(corner_tl, corner_bl, corner_tr, corner_br)
                else:
                    pts = np.array([corner_tl, corner_tr, corner_br, corner_bl], dtype=np.int32)

                paint_filter_in_polygon(frame, pts, FILTROS[filtro_index])
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

                unstable_tag = " [INESTABLE]" if unstable_portal_enabled else ""
                cv2.putText(
                    frame,
                    f"Single Hand Active - Filter {filtro_index + 1}/{len(FILTROS)} (Size: {scale_factor:.1f}x){unstable_tag}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Manos no detectadas - Levanta la(s) mano(s) para aplicar el filtro",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                )

            # Handle gesture trigger event
            if gesture_triggered:
                filtro_index = (filtro_index + 1) % len(FILTROS)
                gesture_banner_timer = 20  # Display feedback banner
                feedback_msg = "Cambio de Filtro!"

            if gesture_banner_timer > 0:
                cv2.rectangle(frame, (w // 2 - 270, 65), (w // 2 + 270, 115), (0, 200, 0), -1)
                cv2.putText(
                    frame,
                    feedback_msg,
                    (w // 2 - 250, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )
                gesture_banner_timer -= 1

            # Settings Overlay Menu Panel
            if show_settings_menu:
                menu_box = frame.copy()
                cv2.rectangle(menu_box, (40, 80), (620, 420), (20, 20, 20), -1)
                frame = cv2.addWeighted(menu_box, 0.85, frame, 0.15, 0)
                cv2.rectangle(frame, (40, 80), (620, 420), (0, 255, 255), 2)
                
                sens_names = {1: "1-Low", 2: "2-Med", 3: "3-High", 4: "4-V.High", 5: "5-Ultra"}
                speed_names = {1: "1-Slow", 2: "2-Med", 3: "3-Normal", 4: "4-Fast", 5: "5-Frenetic"}
                menu_lines = [
                    "--- AR PORTAL SETTINGS ---",
                    f"[G] Gestos Generales:     {'ENABLED' if gestures_enabled else 'DISABLED'}",
                    f"[P] Pellizcar:   {'ENABLED' if pinch_gesture_enabled else 'DISABLED'}",
                    f"[C] Aplauso:       {'ENABLED' if clap_gesture_enabled else 'DISABLED'}",
                    f"[L] Tracking:  {'ENABLED' if draw_skeleton_enabled else 'DISABLED'}",
                    f"[U] Unstable Portal:     {'ENABLED' if unstable_portal_enabled else 'DISABLED'}",
                    f"[,/.] Unstable Speed:    {speed_names[portal_speed_level]}",
                    f"[1-5 or []] Sensivilidad: {sens_names[sensitivity_level]}",
                    f"[+/-] Filtro Tamaño:       {scale_factor:.1f}x",
                    f"[F] Pantalla Completa:     {'ON' if is_fullscreen else 'OFF'}",
                    "-----------------------------------",
                    "Presiona [S] para cerrar el Menu"
                ]
                for idx, line in enumerate(menu_lines):
                    color = (0, 255, 255) if idx == 0 or idx == len(menu_lines)-1 else (255, 255, 255)
                    cv2.putText(frame, line, (60, 115 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

            # HUD Status & Controls
            sens_names = {1: "1-Low", 2: "2-Med", 3: "3-High", 4: "4-V.High", 5: "5-Ultra"}
            speed_names = {1: "1-Slow", 2: "2-Med", 3: "3-Normal", 4: "4-Fast", 5: "5-Frenetic"}
            gest_status = "ALL OFF" if not gestures_enabled else (
                f"Pinch:{'ON' if pinch_gesture_enabled else 'OFF'} Clap:{'ON' if clap_gesture_enabled else 'OFF'}"
            )
            cv2.putText(
                frame,
                f"Gestures: {gest_status} | Sens: {sens_names[sensitivity_level]} [1-5/[] | Size: {scale_factor:.1f}x [+/-] | Speed: {speed_names[portal_speed_level]} [,/.]",
                (20, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 255),
                1,
            )
            cv2.putText(
                frame,
                "Controls: [S] Config Menu | [G] Gestos ON/OFF | [P] Pellizcar | [C] Clap | [L] Tracking | [U] Unstable | [,/.] Speed | [F] Pantalla | [Q] Salir",
                (20, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # Q or ESC
                break
            elif key in (ord("n"), 32):  # N or Space
                filtro_index = (filtro_index + 1) % len(FILTROS)
            elif key == ord("s"):  # S: Toggle Settings Overlay Menu
                show_settings_menu = not show_settings_menu
            elif key == ord("g"):  # G: Toggle Master Gestures ON/OFF
                gestures_enabled = not gestures_enabled
            elif key == ord("p"):  # P: Toggle Pinch Gesture ON/OFF
                pinch_gesture_enabled = not pinch_gesture_enabled
            elif key == ord("c"):  # C: Toggle Clap Gesture ON/OFF
                clap_gesture_enabled = not clap_gesture_enabled
            elif key == ord("l"):  # L: Toggle Skeleton Landmarks drawing ON/OFF
                draw_skeleton_enabled = not draw_skeleton_enabled
            elif key == ord("u"):  # U: Toggle rotating/unstable-corners portal animation
                unstable_portal_enabled = not unstable_portal_enabled
                if unstable_portal_enabled:
                    portal_animator.reset()
            elif key == ord(","):  # , : Decrease unstable portal speed
                portal_speed_level = max(portal_speed_level - 1, 1)
                portal_animator.set_speed_level(portal_speed_level)
            elif key == ord("."):  # . : Increase unstable portal speed
                portal_speed_level = min(portal_speed_level + 1, 5)
                portal_animator.set_speed_level(portal_speed_level)
            elif key in (ord("+"), ord("=")):  # Increase filter size
                scale_factor = min(scale_factor + 0.2, 4.0)
            elif key in (ord("-"), ord("_")):  # Decrease filter size
                scale_factor = max(scale_factor - 0.2, 0.4)
            elif key in (ord("]"), ord("}")):  # Increase gesture sensitivity
                sensitivity_level = min(sensitivity_level + 1, 5)
                closing_detector.set_sensitivity(sensitivity_level)
                pinch_detector.set_sensitivity(sensitivity_level)
            elif key in (ord("["), ord("{")):  # Decrease gesture sensitivity
                sensitivity_level = max(sensitivity_level - 1, 1)
                closing_detector.set_sensitivity(sensitivity_level)
                pinch_detector.set_sensitivity(sensitivity_level)
            elif ord("1") <= key <= ord("5"):  # Direct level selection 1-5
                sensitivity_level = key - ord("0")
                closing_detector.set_sensitivity(sensitivity_level)
                pinch_detector.set_sensitivity(sensitivity_level)
            elif key == ord("f"):  # F to toggle fullscreen
                is_fullscreen = not is_fullscreen
                prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()