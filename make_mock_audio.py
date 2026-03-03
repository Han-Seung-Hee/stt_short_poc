import asyncio
import os
import glob
import re
from pathlib import Path
from pydub import AudioSegment
import edge_tts

# Settings
VOICE_A = "ko-KR-SunHiNeural" # 상담사 (여성)
VOICE_B = "ko-KR-InJoonNeural" # 고객 (남성 - 기본)
VOICE_B_FEMALE = "ko-KR-SunHiNeural" # 고객 (여성) - 기본 제공 여성이 SunHi 하나뿐이라 피치 조절 활용

async def generate_speech(text, voice, output_path, pitch="+0Hz", rate="+0%"):
    communicate = edge_tts.Communicate(text, voice, pitch=pitch, rate=rate)
    await communicate.save(output_path)

async def process_script(script_path):
    stem = Path(script_path).stem.replace("_script", "")
    output_wav = f"test_ready/{stem}_mock_stereo.wav"
    output_txt = f"test_ready/{stem}_mock_ground_truth.txt"
    
    # 시간(분) 목표치 추출
    match = re.search(r"sc_(\d+)min", stem)
    target_ms = int(match.group(1)) * 60 * 1000 if match else 300000
    
    with open(script_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    text_full = " ".join(lines)
    if "남편" in text_full or "시어머니" in text_full:
        customer_voice = VOICE_B_FEMALE
    else:
        customer_voice = VOICE_B
        
    final_audio = AudioSegment.silent(duration=0).set_channels(2)
    ground_truth = []
    
    tmp_dir = Path("/tmp/mock_tts")
    tmp_dir.mkdir(exist_ok=True)
    
    print(f"Generating synthetic stereo call for {stem} (Target: {target_ms/1000}s)...")
    
    for i, line in enumerate(lines):
        if len(final_audio) >= target_ms:
            # 목표 시간에 도달하면 즉시 중단 (outro 생략 방지 차원에서 미리 충분한 길이의 텍스트가 있어야 함)
            # 하지만 시간이 넘었으면 그냥 깔끔하게 끊거나 마지막 인사 하나만 수동으로 붙여줍니다.
            print(f"[{stem}] Reached target duration {len(final_audio)/1000}s. Wrapping up.")
            
            # 마지막 작별인사 억지로 추가 (기승전결 마무리)
            closing_text = "[상담사] 네 고객님, 오늘 긴 시간 말씀 나눠주셔서 감사합니다. 평안한 하루 되십시오."
            tmp_end = tmp_dir / f"{stem}_END.mp3"
            try:
                if not os.path.exists(str(tmp_end)):
                    await generate_speech(closing_text.replace("[상담사] ", ""), VOICE_A, str(tmp_end))
                end_seg = AudioSegment.from_file(str(tmp_end)).set_channels(1).pan(-1.0)
                final_audio += end_seg
                ground_truth.append(closing_text)
            except:
                pass
            break

        pitch = "+0Hz"
        rate = "+0%"
        if line.startswith("[상담사] "):
            speaker = "상담사"
            text = line.replace("[상담사] ", "").strip()
            voice = VOICE_A
            pan_val = -1.0 # Left
        elif line.startswith("[고객] "):
            speaker = "고객"
            text = line.replace("[고객] ", "").strip()
            voice = customer_voice
            if voice == VOICE_B_FEMALE:
                pitch = "-15Hz"
                rate = "+5%"
            pan_val = 1.0 # Right
        else:
            continue
            
        if not text:
            continue
            
        tmp_file = tmp_dir / f"{stem}_{i}.mp3"
        try:
            if not os.path.exists(str(tmp_file)):
                await generate_speech(text, voice, str(tmp_file), pitch=pitch, rate=rate)
        except Exception as e:
            await asyncio.sleep(0.5) # rate limit 회피
            continue
            
        if not os.path.exists(str(tmp_file)):
            continue
            
        segment = AudioSegment.from_file(str(tmp_file))
        panned_segment = segment.set_channels(1).pan(pan_val)
        
        final_audio += panned_segment
        final_audio += AudioSegment.silent(duration=600).set_channels(2) 
        
        ground_truth.append(f"[{speaker}] {text}")
        
    final_audio.export(output_wav, format="wav")
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(ground_truth))
        
    print(f"Saved: {output_wav} ({len(final_audio)/1000} seconds)")

async def main():
    scripts = sorted(glob.glob("test_ready/script/custom/sc_*_script.txt"))
    for script in scripts:
        await process_script(script)

if __name__ == "__main__":
    asyncio.run(main())
