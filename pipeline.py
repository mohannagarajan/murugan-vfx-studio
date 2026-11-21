import os, json, glob, subprocess
import cv2
import numpy as np
from faster_whisper import WhisperModel

os.makedirs("media", exist_ok=True)
os.makedirs("public/layers", exist_ok=True)

FOLDER_URL = "https://drive.google.com/drive/folders/1vzixPqNYooLgMKDKdc3fLresRsiLQxKF"
subprocess.run(["gdown", "--folder", FOLDER_URL, "-O", "media"], check=True)

wav_files = glob.glob("media/**/*.wav", recursive=True) + glob.glob("media/*.wav")
wav_file = wav_files[0]
subprocess.run(["ffmpeg", "-y", "-i", wav_file, "-ar", "44100", "-ac", "2", "public/voice.mp3"], check=True)

dur = float(subprocess.check_output([
    "ffprobe", "-v", "error", "-show_entries", "format=duration",
    "-of", "default=noprint_wrappers=1:nokey=1", "public/voice.mp3"
]).decode().strip())

whisper = WhisperModel("small", device="cpu", compute_type="int8")
segments, _ = whisper.transcribe("public/voice.mp3", language="ta")
segs = list(segments)

if len(segs) >= 5:
    step = len(segs) / 5.0
    cuts = [0.0] + [segs[int(round(i * step))].start for i in range(1, 5)] + [dur]
else:
    cuts = [round(dur * i / 5.0, 2) for i in range(6)]

all_imgs = (
    glob.glob("media/**/*.webp", recursive=True)
    + glob.glob("media/**/*.jpg", recursive=True)
    + glob.glob("media/**/*.png", recursive=True)
)
images = sorted(all_imgs, key=lambda p: os.path.basename(p))[:5]
manifest = []

for idx, img_path in enumerate(images, start=1):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Detect white paper-cutout sticker borders & foreground subjects
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white_border = cv2.inRange(hsv, (0, 0, 215), (180, 45, 255))

    # Central subject mask via GrabCut
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.08), int(h * 0.12), int(w * 0.84), int(h * 0.78))
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 4, cv2.GC_INIT_WITH_RECT)
    fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8") * 255

    # Clean & smooth sticker edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.GaussianBlur(fg_mask, (7, 7), 0)

    # Inpaint background cleanly where characters were lifted
    inpaint_mask = cv2.dilate(fg_mask, kernel, iterations=2)
    bg_clean = cv2.inpaint(img, inpaint_mask, 7, cv2.INPAINT_TELEA)
    bg_local = f"public/layers/scene{idx}_bg.png"
    cv2.imwrite(bg_local, bg_clean)

    # Split distinct character cutouts (left-to-right)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((fg_mask > 127).astype(np.uint8))
    comps = []
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < (w * h * 0.025):
            continue
        comps.append((stats[lbl, cv2.CC_STAT_LEFT], lbl))
    comps.sort(key=lambda x: x[0])

    char_paths = []
    for c_i, (_, lbl) in enumerate(comps, start=1):
        comp_alpha = np.where(labels == lbl, 255, 0).astype(np.uint8)
        comp_alpha = cv2.GaussianBlur(comp_alpha, (5, 5), 0)
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = comp_alpha
        c_local = f"public/layers/scene{idx}_char{c_i}.png"
        cv2.imwrite(c_local, rgba)
        char_paths.append(f"layers/scene{idx}_char{c_i}.png")

    manifest.append({
        "start": round(cuts[idx - 1], 2),
        "end": round(cuts[idx], 2),
        "bg": f"layers/scene{idx}_bg.png",
        "chars": char_paths
    })

with open("public/manifest.json", "w") as f:
    json.dump({"durationInFrames": int(dur * 60), "scenes": manifest}, f, indent=2)
    
