import os, json, glob, subprocess
import numpy as np
from PIL import Image
from faster_whisper import WhisperModel
from transformers import pipeline as hf_pipeline

os.makedirs("media", exist_ok=True)
os.makedirs("public/scenes", exist_ok=True)

FOLDER_URL = "https://drive.google.com/drive/folders/1vzixPqNYooLgMKDKdc3fLresRsiLQxKF"
subprocess.run(["gdown", "--folder", FOLDER_URL, "-O", "media"], check=True)

# 1. Convert audio from Drive
audio_files = (
    glob.glob("media/**/*.wav", recursive=True)
    + glob.glob("media/**/*.mp3", recursive=True)
    + glob.glob("media/**/*.m4a", recursive=True)
)
subprocess.run(["ffmpeg", "-y", "-i", audio_files[0], "-ar", "44100", "-ac", "2", "public/voice.mp3"], check=True)

dur = float(subprocess.check_output([
    "ffprobe", "-v", "error", "-show_entries", "format=duration",
    "-of", "default=noprint_wrappers=1:nokey=1", "public/voice.mp3"
]).decode().strip())

# 2. Find all images uploaded in Drive folder in sorted order (1.webp, 2.webp, ...)
img_files = sorted(
    glob.glob("media/**/*.webp", recursive=True)
    + glob.glob("media/**/*.png", recursive=True)
    + glob.glob("media/**/*.jpg", recursive=True)
    + glob.glob("media/**/*.jpeg", recursive=True)
)

num_scenes = len(img_files)
if num_scenes == 0:
    raise RuntimeError("No images found in Google Drive folder!")

# 3. Transcribe Tamil voiceover for scene timing
whisper = WhisperModel("small", device="cpu", compute_type="int8")
segments, _ = whisper.transcribe("public/voice.mp3", language="ta")
segs = list(segments)

if len(segs) >= num_scenes:
    step = len(segs) / float(num_scenes)
    cuts = [0.0] + [segs[int(round(i * step))].start for i in range(1, num_scenes)] + [dur]
else:
    cuts = [round(dur * i / float(num_scenes), 2) for i in range(num_scenes + 1)]

# 4. Generate 3D Depth Map using Depth-Anything-V2 for each image
depth_estimator = hf_pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=-1)

manifest = []
for idx, src_path in enumerate(img_files):
    img_out = f"public/scenes/scene_{idx+1}.jpg"
    depth_out_path = f"public/scenes/depth_{idx+1}.png"

    pil_img = Image.open(src_path).convert("RGB").resize((1080, 1920), Image.Resampling.LANCZOS)
    pil_img.save(img_out, quality=95)

    depth_pred = depth_estimator(pil_img)["depth"]
    depth_arr = np.array(depth_pred, dtype=np.float32)
    depth_norm = ((depth_arr - depth_arr.min()) / (depth_arr.max() - depth_arr.min() + 1e-6) * 255).astype(np.uint8)
    Image.fromarray(depth_norm).resize((1080, 1920), Image.Resampling.LANCZOS).save(depth_out_path)

    manifest.append({
        "start": round(cuts[idx], 2),
        "end": round(cuts[idx + 1], 2),
        "image": f"scenes/scene_{idx+1}.jpg",
        "depth": f"scenes/depth_{idx+1}.png"
    })

with open("public/manifest.json", "w") as f:
    json.dump({"durationInFrames": int(dur * 60), "scenes": manifest}, f, indent=2)
    
