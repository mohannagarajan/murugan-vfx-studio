import os, json, glob, base64, subprocess, requests
import fal_client
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

def to_data_uri(path):
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{ext};base64,{b64}"

all_imgs = (
    glob.glob("media/**/*.webp", recursive=True)
    + glob.glob("media/**/*.jpg", recursive=True)
    + glob.glob("media/**/*.png", recursive=True)
)
images = sorted(all_imgs, key=lambda p: os.path.basename(p))[:5]
manifest = []

for idx, img_path in enumerate(images, start=1):
    img_url = to_data_uri(img_path)

    seg_res = fal_client.subscribe("fal-ai/sam2/auto-segment", arguments={
        "image_url": img_url,
        "points_per_side": 32,
        "pred_iou_thresh": 0.88
    })

    bg_res = fal_client.subscribe("fal-ai/lama", arguments={
        "image_url": img_url,
        "mask_url": seg_res["combined_mask"]["url"]
    })
    bg_local = f"public/layers/scene{idx}_bg.png"
    with open(bg_local, "wb") as f:
        f.write(requests.get(bg_res["image"]["url"]).content)

    char_paths = []
    masks_sorted = sorted(seg_res["masks"], key=lambda m: m["bbox"][0])
    for c_i, m in enumerate(masks_sorted, start=1):
        if m["area_ratio"] < 0.02 or m["area_ratio"] > 0.50:
            continue
        c_local = f"public/layers/scene{idx}_char{c_i}.png"
        with open(c_local, "wb") as f:
            f.write(requests.get(m["cutout_url"]).content)
        char_paths.append(f"layers/scene{idx}_char{c_i}.png")

    manifest.append({
        "start": round(cuts[idx - 1], 2),
        "end": round(cuts[idx], 2),
        "bg": f"layers/scene{idx}_bg.png",
        "chars": char_paths
    })

with open("public/manifest.json", "w") as f:
    json.dump({"durationInFrames": int(dur * 60), "scenes": manifest}, f, indent=2)
    
