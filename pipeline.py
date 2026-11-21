import os, asyncio, subprocess, glob, time
import edge_tts
from huggingface_hub import InferenceClient
from faster_whisper import WhisperModel

os.makedirs("clips", exist_ok=True)
os.makedirs("out", exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN secret is missing! Add HF_TOKEN in GitHub Repo Settings -> Secrets -> Actions.")

# 1. Story Scenes: (Tamil Narration, Cinematic English AI Video Motion Prompt)
SCENES = [
    (
        "கைலாயத்தில் சிவனும் பார்வதியும் வீற்றிருந்தபோது, நாரதர் ஒரு அபூர்வமான ஞானப்பழத்தைக் கொண்டு வந்து கொடுத்தார்.",
        "Cinematic epic realistic movie shot, Mount Kailash golden temple courtyard, Sage Narada offering a glowing radiant golden mango to Lord Shiva and Goddess Parvati seated on stone throne, volumetric sun rays moving through mist, camera slowly pushing in, 4k photorealistic"
    ),
    (
        "உலகை யார் முதலில் மூன்று முறை சுற்றி வருகிறார்களோ, அவர்களுக்கே இந்த ஞானப்பழம் என்று சிவன் அறிவித்தார்.",
        "Cinematic epic realistic close-up shot of a glowing golden divine fruit radiating golden light particles in temple hall, dramatic warm shadows flickering, smooth cinematic camera orbit, 4k photorealistic movie scene"
    ),
    (
        "உடனே முருகன் தன் மயில் வாகனத்தில் ஏறி உலகை வலம் வர விண்ணில் பறந்தார்.",
        "Cinematic epic realistic action shot of young Lord Murugan holding golden Vel spear riding a majestic peacock flying swiftly through golden clouds above snow-capped Himalayas, wings flapping, wind blowing hair, dynamic tracking camera, 4k photorealistic"
    ),
    (
        "ஆனால் விநாயகரோ, தாய் தந்தையே என் உலகம் என்று சிவனையும் பார்வதியையும் மூன்று முறை வலம் வந்து ஞானப்பழத்தைப் பெற்றார்.",
        "Cinematic epic realistic shot of cute young Lord Ganesha walking with folded hands in devotion around Lord Shiva and Goddess Parvati seated in Mount Kailash, golden flower petals falling slowly in air, warm divine lighting, smooth camera pan, 4k photorealistic"
    ),
    (
        "ஏமாற்றத்தால் கோபம் கொண்ட முருகன் கைலாயத்தை விட்டு நீங்கி, தெற்கே பழனி மலை மீது ஆண்டியாக நின்றார்.",
        "Cinematic epic realistic dramatic shot of Lord Murugan standing atop Palani hill temple steps at sunset wearing simple ochre cloth holding wooden staff Dandayudham, clouds drifting past mountain peak, wind blowing cloth, emotional lighting, 4k photorealistic"
    ),
    (
        "ஞானப்பழம் நீயே அப்பா என்று சிவனும் பார்வதியும் அழைத்ததால்தான், அந்தத் தலம் பழனி என்று பெயர் பெற்றது!",
        "Cinematic epic realistic divine scene on Palani hilltop bathed in brilliant golden heaven rays, Lord Shiva and Goddess Parvati blessing smiling Lord Murugan, glowing aura particles floating upward, slow majestic crane shot, 4k photorealistic"
    )
]

# 2. Generate Realistic Tamil Voiceover via Edge-TTS (or use Drive audio if available)
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

# 3. Generate Real Moving AI Video Clips via Hugging Face Inference Providers
client = InferenceClient(token=HF_TOKEN)
VIDEO_MODELS = [
    "Wan-AI/Wan2.1-T2V-1.3B",
    "Lightricks/LTX-Video",
    "THUDM/CogVideoX-5b"
]

def generate_video_clip(prompt, out_path):
    for model_id in VIDEO_MODELS:
        for attempt in range(2):
            try:
                print(f"Generating AI video with {model_id} (attempt {attempt+1})...")
                video_bytes = client.text_to_video(prompt, model=model_id)
                with open(out_path, "wb") as f:
                    f.write(video_bytes)
                if os.path.getsize(out_path) > 50000:
                    return True
            except Exception as e:
                print(f"Model {model_id} error: {e}")
                time.sleep(5)
    raise RuntimeError(f"All AI video models failed for prompt: {prompt}")

scene_dur = dur / len(SCENES)
clip_paths = []

for idx, (_, vis_prompt) in enumerate(SCENES):
    raw_clip = f"clips/raw_{idx+1}.mp4"
    norm_clip = f"clips/norm_{idx+1}.mp4"
    generate_video_clip(vis_prompt, raw_clip)

    # Loop/stretch & scale each real AI clip to 1080x1920 30fps matching exact narration duration
    subprocess.run([
        "ffmpeg", "-y", "-stream_loop", "-1", "-i", raw_clip,
        "-t", f"{scene_dur:.2f}",
        "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30,format=yuv420p",
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

# 5. Concatenate AI Moving Clips + Voiceover + Tamil Subtitles
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

print("100% AI Cinematic Video Generated Successfully!")
