# AR Portal Filters 🌀✨

An interactive Real-Time Augmented Reality (AR) Portal application that tracks hands via webcam to render dynamic visual filters in mid-air.

> **👤 Original Project by**: [mishu006](https://github.com/mishu006)  
> **🛠️ Fixed, Modernized & Upgraded by**: [baraamallah](https://github.com/baraamallah)  
> **Modded by**: [Shiba-Souls](https://github.com/Shiba-Souls)
> Upgraded to the latest **MediaPipe Tasks API** (`HandLandmarker`), resolving legacy `mp.solutions` deprecation errors, adding Python 3.13 compatibility, custom gesture controls, dynamic filter sizing, multi-level sensitivity adjustment, and an interactive settings menu.

---

## 🌟 Key Features

- **⚡ High-Sensitivity Hand Detection**: Powered by MediaPipe Tasks `HandLandmarker` API with optimized detection thresholds (`0.3`) for fast, responsive tracking even in low light or at long distances.
- **🌀 Dual-Hand AR Portal**: Pinch your thumb and index fingers on both hands to form a dynamic 4-corner AR portal.
- **✋ Single-Hand Filter Mode**: Raise just one hand to automatically anchor a scaled filter box to your hand.
- **🎨 8 Real-Time Visual Filters**:
  - `filtro_grid`: Cyberpunk neon grid overlay.
  - `filtro_1`: Segmented luminosity duotone.
  - `filtro_2`: Classic black & white halftone dot matrix.
  - `filtro_3`: RGB channel-shift chromatic aberration.
  - `filtro_5`: Thermal camera simulation (Jet colormap).
  - `filtro_6`: Vintage sepia with vignette and film noise.
  - `filtro_blanco`: Frosted glass blur effect.
  - `filtro_rosa`: Pink/magenta duotone halftone.
- **🤌 Multi-Gesture Filter Switching**:
  - **Single-Hand Pinch**: Pinch index & thumb tips on *either hand* to change filters.
  - **Two-Hand Clap**: Bring both hands close together to cycle filters.
- **🎚️ 5-Level Adjustable Sensitivity**: Tune gesture responsiveness from `1-Low` to `5-Ultra`.
- **🔍 Dynamic Filter Sizing**: Dynamically shrink (`0.4x`) or enlarge (`4.0x`) the filter size in real time.
- **⚙️ On-Screen Interactive Settings Menu**: Press `S` to toggle a live HUD menu to manage all gesture toggles, skeleton drawing, and visual settings.
- **🧍🧍 1 Person / 2 Person Mode**: Press `M` to switch tracking modes.
  - **1 Person** (default): exactly the original behavior above — one portal, up to 2 hands.
  - **2 People**: up to 2 people can each open their own independent AR portal at the same time, each with its own active filter, its own gestures, and its own unstable-portal animation. Powered by MediaPipe's **`PoseLandmarker`** (body tracking): each person's body position (shoulder center) is used to figure out which of the up to 4 hands on screen belong to which person, based on hand-to-body distance, so the two people's hands don't get crossed into a single "bugged" portal. If a hand ends up too far from both people's bodies, it's simply ignored for that frame instead of risking a wrong pairing.
  - **`B`**: toggle drawing the body skeleton (only relevant in 2-person mode; each person is drawn in a different color).
- **🔊 Audio-Reactive Unstable Portal**: the "unstable" portal animation (rotation, jitter, pulse) reacts live to whatever is playing on your PC — music, a video, a game — without touching the microphone at all.
  - Uses **`PyAudioWPatch`**'s WASAPI **loopback** capture: it opens the *default output device* (speakers/headphones) in capture mode, which reads a copy of what that device is playing. No microphone permission or input is ever used.
  - Louder audio → faster rotation, more jitter/wobble, and a stronger "pulse". A sudden volume spike (a beat, a bass hit) briefly adds extra corners to the polygon for a visible "kick".
  - **`A`**: toggle audio reactivity on/off. Only available on Windows (WASAPI loopback is a Windows-only API); on other platforms, or if `pyaudiowpatch` isn't installed, or if no output device is currently active, the app shows a one-time warning banner and audio reactivity simply stays unavailable — everything else works exactly as before.
  - Applies in both 1-person and 2-person mode, per-portal (each person's portal reacts to the same system audio independently of the other's filter/gesture state).

---

## 🎮 Controls & Shortcuts Cheat Sheet

| Key / Gesture | Action |
| :--- | :--- |
| **Pinch Finger** (Index + Thumb) | Cycle to the next filter (Gesture) |
| **Bring Hands Close** (Clap) | Cycle to the next filter (Gesture) |
| **`N`** or **`Space`** | Cycle to the next filter (Keyboard) |
| **`S`** | Open / Close the On-Screen Settings & Help Panel |
| **`G`** | Master Toggle: Turn **ALL** Gestures ON or OFF |
| **`P`** | Toggle **Single-Hand Pinch Gesture** ON / OFF |
| **`C`** | Toggle **Two-Hand Clap Gesture** ON / OFF |
| **`L`** | Toggle **Hand Skeleton Landmarks** Drawing ON / OFF |
| **`M`** | Switch **1 Person** ⇄ **2 People** tracking mode |
| **`B`** | Toggle **Body Skeleton** Drawing ON / OFF (2-person mode) |
| **`U`** | Toggle **Unstable Portal** Animation ON / OFF |
| **`A`** | Toggle **Audio-Reactive** Unstable Portal ON / OFF (Windows only, needs `pyaudiowpatch`) |
| **`,`** / **`.`** | Decrease (`,`) or Increase (`.`) Unstable Portal Speed |
| **`1`** – **`5`** or **`[`** / **`]`** | Set Gesture Sensitivity (`1-Low`, `2-Med`, `3-High`, `4-V.High`, `5-Ultra`) |
| **`+`** / **`-`** | Enlarge (`+`) or Shrink (`-`) the Filter Size |
| **`F`** | Toggle Full-Screen Camera View |
| **`Q`** or **`ESC`** | Quit Application |

> In 2-person mode, each person cycles **their own** filter with their own pinch/clap gestures. The `N` / `Space` keyboard shortcut advances both people's filters at once (there's no keyboard way to target just one person — that's what the gestures are for).

---

## 🛠️ Major Fixes & Enhancements Made

1. **Fixed Legacy MediaPipe Deprecation (`AttributeError: module 'mediapipe' has no attribute 'solutions'`)**:
   - Replaced removed `mp.solutions.hands` with `mp.tasks.vision.HandLandmarker` using `hand_landmarker.task`.
2. **Python 3.13 Windows `ctypes` Compatibility Fix**:
   - Added automatic Windows C-bindings patch for `ctypes.cdll.msvcrt.free` on Python 3.13+.
3. **100% Reliable 2-Hand Sorting**:
   - Replaced unreliable `handedness` classification loops with horizontal X-coordinate sorting, guaranteeing correct `left_hand` and `right_hand` assignment.
4. **OpenCV Full-Screen Attribute Fix**:
   - Fixed `AttributeError: module 'cv2' has no attribute 'WNDPROP_FULLSCREEN'` by using standard `cv2.WND_PROP_FULLSCREEN`.
5. **Interactive UI / HUD**:
   - Added live visual feedback banners (`"GESTURE DETECTED! FILTER CHANGED"`), hand count indicators, size counters, and an interactive settings menu overlay (`S`).
6. **Body Tracking & Independent 2-Person Portals**:
   - Added `body_tracking.py` (`PoseLandmarker`) to detect up to 2 people's body position.
   - Added `geometry.group_hands_by_person()`, which assigns each of the up to 4 detected hands to the nearest person's body (by distance) instead of naive left/right sorting, so two people's hands don't get crossed into the same portal.
   - Added `geometry.PersonPortalState`, giving each person their own filter, gesture detectors, and portal animation state, so both portals run fully independently.
   - 1-person mode's code path is untouched — 2-person mode is fully additive and only active while toggled on with `M`.
7. **Audio-Reactive Unstable Portal**:
   - Added `audio_reactive.py` (`SystemAudioAnalyzer`), which captures **system output audio** (speakers/headphones) via `PyAudioWPatch`'s WASAPI loopback mode, running in its own background thread so it never blocks the video loop.
   - `geometry.UnstablePortalAnimator.animate()` now takes an `audio_level`/`audio_pulse` pair that modulates rotation speed, corner jitter, pulse amplitude, and briefly spikes the corner count on a detected "beat" — on top of the existing manual speed levels.
   - Fully optional and additive: if `pyaudiowpatch` isn't installed, the platform isn't Windows, or no output device is active, the app shows a one-time warning and everything falls back to the original (non-audio) unstable animation with zero behavior change.

---

## 🚀 Quick Start & Installation

### 1. Clone the repository
```bash
git clone https://github.com/mishu006/Filters.git
cd Filters
```

### 2. Set up a virtual environment (Recommended)
```bash
python -m venv venv
```

Activate it:
- **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **macOS / Linux**:
  ```bash
  source venv/bin/activate
  ```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Only for 2-person mode) Download the body-tracking model
1-person mode works out of the box with just `hand_landmarker.task`. To use **2-person mode** (`M` key) you also need MediaPipe's pose model, `pose_landmarker_lite.task`, placed in the project root next to `hand_landmarker.task`:

```bash
curl -L -o pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
```

```powershell
# PowerShell equivalent
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task" -OutFile "pose_landmarker_lite.task"
```

If this file is missing, 1-person mode still works normally — pressing `M` to switch to 2-person mode will just show an on-screen warning and stay in 1-person mode instead of crashing.

### 5. (Only for audio-reactive portal, Windows only) Install PyAudioWPatch
`PyAudioWPatch` is already listed in `requirements.txt` (Windows-only marker, so it's skipped automatically on macOS/Linux). It's what lets the app read **system audio via WASAPI loopback** — a copy of whatever is currently playing through your speakers/headphones, never the microphone.

If it's missing or fails to install, or you're not on Windows, the app still runs fine: audio reactivity (`A` key) just stays unavailable and everything else works exactly as before.

### 6. Run the app
```powershell
python main.py
```

---

## 📁 Project Structure

```
Filters/
├── main.py                     # Main entry point: capture loop, mode switch, gesture handlers & UI HUD
├── hand_tracking.py            # HandTracker class using MediaPipe Tasks HandLandmarker
├── body_tracking.py            # BodyTracker class using MediaPipe Tasks PoseLandmarker (2-person mode)
├── geometry.py                 # Portal math, gesture detectors, per-person state & hand/person grouping
├── audio_reactive.py           # SystemAudioAnalyzer: WASAPI loopback capture (PyAudioWPatch), no microphone
├── filters.py                  # Definition of 8 custom OpenCV filters
├── hand_landmarker.task        # MediaPipe HandLandmarker TFLite model file
├── pose_landmarker_lite.task   # MediaPipe PoseLandmarker model file (only needed for 2-person mode)
├── requirements.txt            # Project dependencies (opencv-python, mediapipe, numpy)
└── README.md                   # Documentation & credits
```

---

## 🎨 Adding New Filters

To create your own custom filter, add a function in `filters.py` that takes a BGR image crop (`np.ndarray`) and returns a modified crop of the same shape:

```python
def filtro_custom(roi: np.ndarray) -> np.ndarray:
    # Your image processing logic here (e.g. cv2.applyColorMap, blurring, tinting)
    return roi
```

Then append your new filter function to the `FILTROS` list at the bottom of `filters.py`. The application will automatically include it in the gesture rotation cycle!

---

## 💻 Tech Stack

- **Language**: Python 3.10 – 3.13
- **Computer Vision**: OpenCV (`opencv-python`)
- **ML Hand Tracking**: MediaPipe Tasks `HandLandmarker` (`mediapipe>=0.10.20`)
- **ML Body Tracking**: MediaPipe Tasks `PoseLandmarker` (2-person mode)
- **System Audio Capture**: `PyAudioWPatch` (WASAPI loopback, Windows only, audio-reactive portal)
- **Array Math**: NumPy

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

**Original Project by [mishu006](https://github.com/mishu006) — Fixed & Upgraded by [baraamallah](https://github.com/baraamallah) — Modded by [Shiba-Souls](https://github.com/Shiba-Souls)**
