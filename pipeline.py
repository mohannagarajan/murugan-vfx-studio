import os, json, glob, time, urllib.parse, subprocess
import requests
import numpy as np
from PIL import Image
from faster_whisper import WhisperModel
import torch
from transformers import pipeline as hf_pipeline

os.makedirs("media", exist_ok=True)
os.makedirs("public/scenes", exist_ok=True)

FOLDER_URL = "https://drive.google.com/drive/folders/1vzixPqNYooLgMKDKdc3fLresRsiLQxKF"
subprocess.run(["gdown", "--folder", FOLDER_URL, "-O", "media"], check=True)

# 1. Find voiceover audio in Drive folder
audio_files = (
    glob.glob("media/**/*.wav", recursive=True)
    + glob.glob("media/**/*.mp3", recursive=True)
    + glob.glob("media/**/*.m4a", recursive=True)
)
audio_file = audio_files[0]
subprocess.run(["ffmpeg", "-y", "-i", audio_file, "-ar", "44100", "-ac", "2", "public/voice.mp3"], check=True)

dur = float(subprocess.check_output([
    "ffprobe", "-v", "error", "-show_entries", "format=duration",
    "-of", "default=noprint_wrappers=1:nokey=1", "public/voice.mp3"
]).decode().strip())

# 2. Transcribe Tamil voiceover for exact timing
whisper = WhisperModel("small", device="cpu", compute_type="int8")
segments, _ = whisper.transcribe("public/voice.mp3", language="ta")
segs = list(segments)

# Cinematic visual prompts matching the story beats (or driven by segment count)
DEFAULT_PROMPTS = [
    "Cinematic epic realistic photography of Mount Kailash golden temple courtyard at sunrise, Lord Shiva with glowing crescent moon and Goddess Parvati seated on carved stone throne, Sage Narada offering a radiant glowing golden divine mango, volumetric god rays, 8k photorealistic movie still, shallow depth of field",
    "Cinematic epic realistic close-up of a glowing golden divine fruit held in hands with radiant sparkles, Lord Shiva announcing a challenge, young Lord Murugan and baby Ganesha listening intently, dramatic warm temple lighting, 8k photorealistic film still",
    "Cinematic epic realistic action shot of young Lord Murugan holding golden Vel spear flying swiftly across golden clouds and snow-capped Himalayas on a majestic iridescent peacock, dynamic motion, dramatic sunset rays, 8k photorealistic movie still",
    "Cinematic epic realistic scene of cute young Lord Ganesha walking with folded hands in pure devotion around Lord Shiva and Goddess Parvati seated in Mount Kailash, warm divine golden aura, petals falling, 8k photorealistic film still",
    "Cinematic epic realistic shot of Lord Murugan standing atop Palani hill temple steps at dramatic crimson dusk wearing simple ochre cloth and holding sacred wooden staff Dandayudham, misty mountains below, emotional dramatic lighting, 8k photorealistic movie still",
    "Cinematic epic realistic divine scene on Palani hilltop bathed in brilliant golden heaven rays, Lord Shiva and Goddess Parvati blessing smiling Lord Murugan, glowing divine light, rich cinematic color grading, 8k photorealistic film still"
]

num_scenes = 6
if len(segs) >= num_scenes:
    step = len(segs) / float(num_scenes)
    cuts = [0.0] + [segs[int(round(i * step))].start for i in range(1, num_scenes)] + [dur]
else:
    cuts = [round(dur * i / float(num_scenes), 2) for i in range(num_scenes + 1)]

# Check if user placed a custom prompts.txt in Drive (one English visual prompt per line)
txt_files = glob.glob("media/**/prompts.txt", recursive=True) + glob.glob("media/prompts.txt")
if txt_files:
    with open(txt_files[0], "r", encoding="utf-8") as tf:
        custom_lines = [l.strip() for l in tf.readlines() if l.strip()]
    if custom_lines:
        DEFAULT_PROMPTS = custom_lines
        num_scenes = len(DEFAULT_PROMPTS)
        cuts = [round(dur * i / float(num_scenes), 2) for i in range(num_scenes + 1)]

# 3. Load Depth-Anything-V2 for real 3D Depth Parallax Map
depth_estimator = hf_pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=-1)

manifest = []

def generate_ai_image(prompt, out_path, seed):
    encoded = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=1080&height=1920&seed={seed}&model=flux&nologo=true"
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=90)
            if r.status_code == 200 and len(r.content) > 15000:
                with open(out_path, "wb") as f:
                    f.write(r.content)
                return True
        except Exception as e:
            print(f"Retry {attempt+1} for {out_path}: {e}")
        time.sleep(4)
    raise RuntimeError(f"Failed to generate image for {out_path}")

for idx in range(num_scenes):
    prompt = DEFAULT_PROMPTS[idx % len(DEFAULT_PROMPTS)]
    img_file = f"public/scenes/scene_{idx+1}.jpg"
    depth_file = f"public/scenes/depth_{idx+1}.png"

    generate_ai_image(prompt, img_file, seed=1000 + idx * 77)

    # Compute 3D Depth Map (Bright = Near Foreground, Dark = Far Background)
    pil_img = Image.open(img_file).convert("RGB")
    depth_out = depth_estimator(pil_img)["depth"]
    depth_arr = np.array(depth_out, dtype=np.float32)
    depth_norm = ((depth_arr - depth_arr.min()) / (depth_arr.max() - depth_arr.min() + 1e-6) * 255).astype(np.uint8)
    Image.fromarray(depth_norm).resize((1080, 1920)).save(depth_file)

    manifest.append({
        "start": round(cuts[idx], 2),
        "end": round(cuts[idx + 1], 2),
        "image": f"scenes/scene_{idx+1}.jpg",
        "depth": f"scenes/depth_{idx+1}.png"
    })

with open("public/manifest.json", "w") as f:
    json.dump({"durationInFrames": int(dur * 60), "scenes": manifest}, f, indent=2)
