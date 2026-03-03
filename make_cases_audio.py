import asyncio
import os
import glob
from pathlib import Path
from pydub import AudioSegment
import edge_tts

# Settings
VOICE_FEMALE_1 = "ko-KR-SunHiNeural"
VOICE_MALE_1 = "ko-KR-InJoonNeural"
VOICE_MALE_2 = "ko-KR-HyunsuMultilingualNeural"

async def generate_speech(text, voice, output_path, pitch="+0Hz", rate="+0%"):
    communicate = edge_tts.Communicate(text, voice, pitch=pitch, rate=rate)
    await communicate.save(output_path)

async def process_case(script_path, agent_voice, customer_voice, customer_pitch="+0Hz", customer_rate="+0%"):
    stem = Path(script_path).stem
    output_wav = f"test_ready/final_{stem}_mock_stereo.wav"
    output_txt = f"test_ready/final_{stem}_mock_ground_truth.txt"
    
    with open(script_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    final_audio = AudioSegment.silent(duration=0).set_channels(2)
    ground_truth = []
    
    tmp_dir = Path("/tmp/mock_tts_cases")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating synthetic stereo call for {stem}...")
    
    for i, line in enumerate(lines):
        pitch = "+0Hz"
        rate = "+0%"
        
        if line.startswith("[상담사]"):
            speaker = "상담사"
            text = line.replace("[상담사]", "").strip()
            voice = agent_voice
            pan_val = -1.0 # Left
        elif line.startswith("[고객]"):
            speaker = "고객"
            text = line.replace("[고객]", "").strip()
            voice = customer_voice
            pitch = customer_pitch
            rate = customer_rate
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
    # 1. 여 / 남 (SunHi / InJoon)
    await process_case("test_ready/script/custom/sc_case_1.txt", 
                       agent_voice=VOICE_FEMALE_1, 
                       customer_voice=VOICE_MALE_1)
                       
    # 2. 남 / 남 (Hyunsu / InJoon)
    await process_case("test_ready/script/custom/sc_case_2.txt", 
                       agent_voice=VOICE_MALE_2, 
                       customer_voice=VOICE_MALE_1)
                       
    # 3. 여 / 여 (SunHi / SunHi with pitch -15Hz)
    await process_case("test_ready/script/custom/sc_case_3.txt", 
                       agent_voice=VOICE_FEMALE_1, 
                       customer_voice=VOICE_FEMALE_1,
                       customer_pitch="-15Hz",
                       customer_rate="+5%")
                       
    # 4. 남 / 여 (Hyunsu / SunHi)
    await process_case("test_ready/script/custom/sc_case_4.txt", 
                       agent_voice=VOICE_MALE_2, 
                       customer_voice=VOICE_FEMALE_1)

    # 5. 남 / 남 (Hyunsu / InJoon - 격양된 톤)
    await process_case("test_ready/script/custom/sc_case_5.txt", 
                       agent_voice=VOICE_MALE_2, 
                       customer_voice=VOICE_MALE_1,
                       customer_rate="+10%")

if __name__ == "__main__":
    asyncio.run(main())
