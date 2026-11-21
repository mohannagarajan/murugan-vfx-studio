import os, asyncio, subprocess, time
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

# 3. Generate Brand-New Realistic AI Images via Free Hugging Face Models
client = InferenceClient(token=HF_TOKEN)
IMAGE_MODELS = [
    "black-forest-labs/FLUX.1-schnell",
    "stabilityai/stable-diffusion-xl-base-1.0"
]

def generate_ai_image(prompt, out_path):
    for model_id in IMAGE_MODELS:
        for attempt in range(3):
            try:
                print(f"Generating AI image with {model_id} (attempt {attempt+1})...")
                img = client.text_to_image(prompt, model=model_id, width=768, height=1344)
                img.save(out_path, quality=95)
                if os.path.getsize(out_path) > 20000:
                    return True
            except Exception as e:
                print(f"Model {model_id} error: {e}")
                time.sleep(4)
    raise RuntimeError(f"Image generation failed for prompt: {prompt}")

scene_dur = dur / len(SCENES)
fps = 30
frames_per_scene = int(scene_dur * fps)
clip_paths = []

for idx, (_, vis_prompt) in enumerate(SCENES):
    img_path = f"clips/scene_{idx+1}.jpg"
    norm_clip = f"clips/norm_{idx+1}.mp4"
    generate_ai_image(vis_prompt, img_path)

    # Smooth Cinematic Zoom-In / Zoom-Out + Pan Motion
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
