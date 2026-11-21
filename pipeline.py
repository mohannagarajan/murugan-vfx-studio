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

def save_rgba(path, bgr, alpha):
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    cv2.imwrite(path, rgba)

for idx, img_path in enumerate(images, start=1):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 1. Exact White Sticker Outline Detection for Characters
    white_mask = cv2.inRange(hsv, (0, 0, 222), (180, 42, 255))
    # Exclude outer paper frame border (top/bottom/left/right 3%)
    border_pad = int(min(w, h) * 0.035)
    white_mask[:border_pad, :] = 0
    white_mask[-border_pad:, :] = 0
    white_mask[:, :border_pad] = 0
    white_mask[:, -border_pad:] = 0

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed_white = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, k_close, iterations=2)

    contours, _ = cv2.findContours(closed_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_mask_total = np.zeros((h, w), dtype=np.uint8)
    valid_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (w * h * 0.015) or area > (w * h * 0.55):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        # Skip upper sky clouds (top 32% & wide/short)
        if y < h * 0.28 and bw > bh * 1.4:
            continue
        valid_contours.append((x, cnt))
        cv2.drawContours(char_mask_total, [cnt], -1, 255, thickness=cv2.FILLED)

    # Fallback to GrabCut if any image lacks white sticker outline
    if len(valid_contours) == 0:
        gc_mask = np.zeros((h, w), np.uint8)
        rect = (int(w * 0.10), int(h * 0.18), int(w * 0.80), int(h * 0.72))
        cv2.grabCut(img, gc_mask, rect, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64), 4, cv2.GC_INIT_WITH_RECT)
        char_mask_total = np.where((gc_mask == 2) | (gc_mask == 0), 0, 255).astype(np.uint8)
        char_mask_total = cv2.morphologyEx(char_mask_total, cv2.MORPH_CLOSE, k_close)
        n_l, lbls, stats, _ = cv2.connectedComponentsWithStats(char_mask_total)
        for l in range(1, n_l):
            if stats[l, cv2.CC_STAT_AREA] > w * h * 0.025:
                c_m = np.where(lbls == l, 255, 0).astype(np.uint8)
                cnts, _ = cv2.findContours(c_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    valid_contours.append((stats[l, cv2.CC_STAT_LEFT], cnts[0]))

    valid_contours.sort(key=lambda item: item[0])

    # Save individual sharp character cutouts
    char_paths = []
    for c_i, (_, cnt) in enumerate(valid_contours, start=1):
        single_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(single_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        single_mask = cv2.GaussianBlur(single_mask, (3, 3), 0)
        c_file = f"public/layers/scene{idx}_char{c_i}.png"
        save_rgba(c_file, img, single_mask)
        char_paths.append(f"layers/scene{idx}_char{c_i}.png")

    # Clean background without characters
    inpaint_mask = cv2.dilate(char_mask_total, np.ones((11, 11), np.uint8), iterations=2)
    bg_clean = cv2.inpaint(img, inpaint_mask, 5, cv2.INPAINT_TELEA)

    # 2. Split Background into 3 Depth Stages (Sky/Sun, Clouds/Mid, Foreground Hills)
    # Stage A: Full Sky + Sun base
    sky_file = f"public/layers/scene{idx}_sky.png"
    cv2.imwrite(sky_file, bg_clean)

    # Stage B: Upper Clouds layer (top 42% white/cream paper clouds)
    cloud_mask = cv2.inRange(hsv, (0, 0, 200), (180, 55, 255))
    cloud_mask[int(h * 0.42):, :] = 0
    cloud_mask[:border_pad, :] = 0
    cloud_mask = cv2.bitwise_and(cloud_mask, cv2.bitwise_not(inpaint_mask))
    cloud_mask = cv2.GaussianBlur(cv2.dilate(cloud_mask, np.ones((5, 5), np.uint8)), (5, 5), 0)
    clouds_file = f"public/layers/scene{idx}_clouds.png"
    save_rgba(clouds_file, bg_clean, cloud_mask)

    # Stage C: Lower Hills / Terraced Paper Ground (bottom 58% feathered paper horizon)
    ground_mask = np.zeros((h, w), dtype=np.uint8)
    for x_col in range(w):
        wave = int(h * 0.44 + np.sin(x_col / 95.0 + idx) * (h * 0.035))
        ground_mask[wave:, x_col] = 255
    ground_mask = cv2.GaussianBlur(ground_mask, (21, 21), 0)
    hills_file = f"public/layers/scene{idx}_hills.png"
    save_rgba(hills_file, bg_clean, ground_mask)

    manifest.append({
        "start": round(cuts[idx - 1], 2),
        "end": round(cuts[idx], 2),
        "sky": f"layers/scene{idx}_sky.png",
        "clouds": f"layers/scene{idx}_clouds.png",
        "hills": f"layers/scene{idx}_hills.png",
        "chars": char_paths
    })

with open("public/manifest.json", "w") as f:
    json.dump({"durationInFrames": int(dur * 60), "scenes": manifest}, f, indent=2)
    
