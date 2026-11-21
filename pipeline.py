import os, asyncio, subprocess, time, requests
import edge_tts
from huggingface_hub import InferenceClient
from faster_whisper import WhisperModel

os.makedirs("clips", exist_ok=True)
os.makedirs("out", exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN secret is missing! Add HF_TOKEN in GitHub Repo Settings -> Secrets -> Actions.")

# 1. Story Scenes: (Tamil Narration, Cinematic English AI Image Prompt)
SCENES = [
    (
        "கைலாயத்தில் சிவனும் பார்வதியும் வீற்றிருந்தபோது, நாரதர் ஒரு அபூர்வமான ஞானப்பழத்தைக் கொண்டு வந்து கொடுத்தார்.",
        "Cinematic epic realistic movie still, Mount Kailash golden temple courtyard, Sage Narada offering a glowing radiant golden mango to Lord Shiva and Goddess Parvati seated on stone throne, volumetric sun rays through mist, 8k photorealistic, vertical 9:16"
    ),
    (
        "உலகை யார் முதலில் மூன்று முறை சுற்றி வருகிறார்களோ, அவர்களுக்கே இந்த ஞானப்பழம் என்று சிவன் அறிவித்தார்.",
        "Cinematic epic realistic close-up shot of a glowing golden divine fruit radiating golden light particles in ancient temple hall, dramatic warm lighting, shallow depth of field, 8k photorealistic, vertical 9:16"
    ),
    (
        "உடனே முருகன் தன் மயில் வாகனத்தில் ஏறி உலகை வலம் வர விண்ணில் பறந்தார்.",
        "Cinematic epic realistic action shot of young Lord Murugan holding golden Vel spear riding a majestic peacock flying swiftly through golden clouds above snow-capped Himalayas, dramatic sunset rays, 8k photorealistic, vertical 9:16"
    ),
    (
        "ஆனால் விநாயகரோ, தாய் தந்தையே என் உலகம் என்று சிவனையும் பார்வதியையும் மூன்று முறை வலம் வந்து ஞானப்பழத்தைப் பெற்றார்.",
        "Cinematic epic realistic shot of cute young Lord Ganesha walking with folded hands in devotion around Lord Shiva and Goddess Parvati seated in Mount Kailash, golden flower petals falling in air, warm divine aura, 8k photorealistic, vertical 9:16"
    ),
    (
        "ஏமாற்றத்தால் கோபம் கொண்ட முருகன் கைலாயத்தை விட்டு நீங்கி, தெற்கே பழனி மலை மீது ஆண்டியாக நின்றார்.",
        "Cinematic epic realistic dramatic shot of Lord Murugan standing atop Palani hill temple steps at sunset wearing simple ochre cloth holding wooden staff Dandayudham, misty mountains below, emotional lighting, 8k photorealistic, vertical 9:16"
    ),
    (
        "ஞானப்பழம் நீயே அப்பா என்று சிவனும் பார்வதியும் அழைத்ததால்தான், அந்தத் தலம் பழனி என்று பெயர் பெற்றது!",
        "Cinematic epic realistic divine scene on Palani hilltop bathed in brilliant golden heaven rays, Lord Shiva and Goddess Parvati blessing smiling Lord Murugan, glowing aura particles floating upward, 8k photorealistic, vertical 9:16"
    )
]

# 2. Generate Realistic Tamil Voiceover via Edge-TTS
custom_story = os.environ.get("STORY_INPUT", "").strip()
full_tamil_text = custom_story if custom_story else " ".join([s[0] for s in SCENES])

async def make_tts(text, out_mp3):
    communicate = edge_tts.Communicate(text, voice="ta-IN-ValluvarNeural", rate="-5%")
    await communicate.save(out_mp3)

asyncio.run(make_tts(full_tamil_text, "out/voice.mp3"))

dur = float(subprocess.check_output([
    "ffprobe", "-v", "error", "-show_entries", "format=duration",
    "-of", "default=noprint_wrappers=1:nokey=1", "out/voice.mp3"
]).decode().strip())

# 3. Generate Brand-New Realistic AI Images (HF Direct Free Provider + AI Horde Free GPU Fallback)
client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)
HF_MODELS = [
    "black-forest-labs/FLUX.1-schnell",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "ByteDance/SDXL-Lightning"
]

def generate_via_aihorde(prompt, out_path):
    print("Trying 100% Free AI Horde GPU network...")
    headers = {"apikey": "0000000000", "Client-Agent": "murugan-vfx:1.0:github"}
    payload = {
        "prompt": prompt,
        "params": {"width": 576, "height": 1024, "steps": 20, "n": 1},
        "models": ["AlbedoBase XL (SDXL)", "SDXL 1.0", "DreamShaper XL"],
        "r2": True
    }
    res = requests.post("https://aihorde.net/api/v2/generate/async", json=payload, headers=headers, timeout=30)
    res.raise_for_status()
    job_id = res.json()["id"]
    for _ in range(45):
        time.sleep(4)
        chk = requests.get(f"https://aihorde.net/api/v2/generate/status/{job_id}", headers=headers, timeout=20).json()
        if chk.get("done") and chk.get("generations"):
            img_url = chk["generations"][0]["img"]
            img_bytes = requests.get(img_url, timeout=30).content
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            return True
    return False

def generate_ai_image(prompt, out_path):
    for model_id in HF_MODELS:
        for attempt in range(2):
            try:
                print(f"Generating AI image with {model_id} via hf-inference (attempt {attempt+1})...")
                img = client.text_to_image(prompt, model=model_id)
                img.save(out_path, quality=95)
                if os.path.getsize(out_path) > 15000:
                    return True
            except Exception as e:
                print(f"Model {model_id} error: {e}")
                time.sleep(3)
    if generate_via_aihorde(prompt, out_path):
        return True
    raise RuntimeError(f"All free AI image providers failed for prompt: {prompt}")

scene_dur = dur / len(SCENES)
fps = 30
frames_per_scene = int(scene_dur * fps)
clip_paths = []

for idx, (_, vis_prompt) in enumerate(SCENES):
    img_path = f"clips/scene_{idx+1}.jpg"
    norm_clip = f"clips/norm_{idx+1}.mp4"
    generate_ai_image(vis_prompt, img_path)

    # Smooth Cinematic Zoom-In / Zoom-Out Motion
    if idx % 2 == 0:
        zoom_expr = "min(zoom+0.0015,1.25)"
    else:
        zoom_expr = "if(lte(zoom,1.0),1.25,max(1.001,zoom-0.0015))"

    subprocess.run([
        "ffmpeg", "-y", "-loop", "1", "-i", img_path,
        "-vf", f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,zoompan=z='{zoom_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames_per_scene}:s=1080x1920:fps={fps},format=yuv420p",
        "-t", f"{scene_dur:.2f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-an",
        norm_clip
    ], check=True)
    clip_paths.append(norm_clip)

# 4. Generate Tamil Subtitles (SRT) via Faster-Whisper
whisper = WhisperModel("small", device="cpu", compute_type="int8")
segments, _ = whisper.transcribe("out/voice.mp3", language="ta")

def fmt_srt_time(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

with open("out/subs.srt", "w", encoding="utf-8") as srt:
    for i, seg in enumerate(segments, start=1):
        srt.write(f"{i}\n{fmt_srt_time(seg.start)} --> {fmt_srt_time(seg.end)}\n{seg.text.strip()}\n\n")

# 5. Combine AI Scenes + Tamil Voiceover + Tamil Subtitles
with open("clips/list.txt", "w") as lf:
    for cp in clip_paths:
        lf.write(f"file '{os.path.abspath(cp)}'\n")

subprocess.run([
    "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "clips/list.txt",
    "-i", "out/voice.mp3",
    "-vf", "subtitles=out/subs.srt:force_style='FontName=Noto Sans Tamil,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Alignment=2,MarginV=120'",
    "-c:v", "libx264", "-preset", "fast", "-crf", "19",
    "-c:a", "aac", "-b:a", "192k", "-shortest",
    "out/final_ai_video.mp4"
], check=True)

print("100% Free AI Story Video Generated Successfully!")
