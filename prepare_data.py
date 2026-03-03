"""AI Hub 비식별화 통화 테이터 병합 파이썬 스크립트.

AI Hub의 콜센터/정신건강 상담 데이터는 문장 단위로 짧게 쪼개져 있습니다.
본 스크립트는 쪼개진 여러 개의 `.wav` 파일과 `.json` 라벨링 파일들을 하나로 이어 붙여서,
STT PoC 테스트에 바로 사용할 수 있는 1개의 통합 오디오 파일과 평가 비교용 정답지(대본) 문서로 생성해 줍니다.

[사용 방법]
1. 가상환경 활성화가 되어 있어야 합니다 (`source .venv/bin/activate`)
2. 터미널에서 아래 명령어를 실행합니다. 변환하고 싶은 대화 폴더의 '고유 ID'를 인자로 전달합니다.

   기본 명령어 예시:
   $ python prepare_data.py MEN0005946

[선택 옵션 (Optional)]
--base : AI Hub 원본 데이터('wav', 'labeling' 폴더가 있는) 최상위 경로 (기본값: 'base_data')
--out  : 추출 및 병합된 파일이 저장될 출력 폴더 경로 (기본값: 'test_ready')

   옵션 적용 명령어 예시:
   $ python prepare_data.py MEN0005946 --base ./my_data_folder --out ./result_folder

[출력 결과물]
실행이 완료되면 `--out` 으로 지정한 폴더(`test_ready/`) 경로에 2개의 파일이 생성됩니다.
1. [ID]_merged.wav : index.html 등 STT 서비스에 업로드하여 테스트할 병합된 1개의 통화 오디오 파일
2. [ID]_ground_truth.txt : 화자(상담사/고객) 구분과 함께 시간 순서로 정리된 실제 대화 내역 (성능 비교/평가용 문서)
"""

import argparse
import json
import logging
from pathlib import Path

from pydub import AudioSegment

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def merge_aihub_data(base_dir: str, conversation_id: str, output_dir: str):
    base_path = Path(base_dir)
    wav_base = base_path / "wav"
    label_base = base_path / "labeling"
    
    # 해당 ID를 가진 디렉토리 찾기 (하위 폴더 깊이에 상관없이 탐색)
    wav_dirs = list(wav_base.rglob(conversation_id))
    label_dirs = list(label_base.rglob(conversation_id))
    
    if not wav_dirs:
        logging.error(f"WAV 디렉토리를 찾을 수 없습니다: {wav_base} 하위에 {conversation_id} 폴더가 없습니다.")
        return
    if not label_dirs:
        logging.warning(f"라벨링 디렉토리를 찾을 수 없습니다: {label_base} 하위에 {conversation_id} 폴더가 없습니다. 오디오 병합(스테레오)만 진행합니다.")
        label_dir = None
    else:
        label_dir = label_dirs[0]
        
    wav_dir = wav_dirs[0]
    
    # 파일 목록 수집 및 정렬 (파일 이름의 순번이 정렬 기준이 됩니다)
    wav_files = sorted(list(wav_dir.glob("*.wav")))
    json_files = sorted(list(label_dir.glob("*.json"))) if label_dir else []
    
    if not wav_files:
        logging.error(f"[{wav_dir}] 경로에 WAV 파일이 존재하지 않습니다.")
        return
        
    logging.info(f"==== 파싱 시작: {conversation_id} ====")
    logging.info(f"WAV 파일 {len(wav_files)}개, JSON 파일 {len(json_files)}개 로드 완료")
    
    json_dict = {f.stem: f for f in json_files}
    
    # A(상담사)와 B(고객) 파일 개수를 세어 비율(비례) 정렬을 위한 기준값 마련
    counts = {"A": 0, "B": 0}
    for w in wav_files:
        if "A" in w.stem: counts["A"] += 1
        elif "B" in w.stem: counts["B"] += 1
        
    def sort_key(w_path):
        stem = w_path.stem
        seq = int(stem[-3:]) if stem[-3:].isdigit() else 999
        
        # 전체 길이 대비 현재 문장의 비율 위치(0.0 ~ 1.0)를 계산하여
        # 대화가 한쪽으로 몰리지 않고 끝까지 자연스럽게 핑퐁(교차)되도록 비율 기반 정렬
        is_a = "A" in stem
        speaker = "A" if is_a else "B"
        max_count = counts[speaker] if counts[speaker] > 0 else 1
        
        ratio = seq / max_count
        speaker_order = 1 if is_a else 0 # 같은 비율 시 B(고객)가 우선
        return (ratio, speaker_order, seq)
        
    wav_files.sort(key=sort_key)
    
    merged_audio = AudioSegment.empty()
    ground_truth_lines = []
    
    for w_file in wav_files:
        stem = w_file.stem
        
        audio_segment = AudioSegment.from_wav(str(w_file))
        
        # AI Hub 파일명 형식
        if "A" in stem:
            panned_segment = AudioSegment.silent(duration=len(audio_segment)).overlay(audio_segment, position=0).pan(-1.0)
        elif "B" in stem:
            panned_segment = AudioSegment.silent(duration=len(audio_segment)).overlay(audio_segment, position=0).pan(1.0)
        else:
            panned_segment = audio_segment.pan(0.0)
            
        # 순차적으로 단순 이어붙이기 (앞뒤 약간의 공백/재생시간 보존)
        merged_audio += panned_segment
        
        # 라벨링 대본 추가
        if label_dir:
            j_file = json_dict.get(stem)
            if j_file and j_file.exists():
                with open(j_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    info = data.get("info", [{}])[0].get("metadata", {})
                    speaker_type = info.get("speaker_type", "알수없음")
                    
                    texts = data.get("inputText", [])
                    text_content = " ".join([t.get("orgtext", "") for t in texts])
                    
                    ground_truth_lines.append(f"[{speaker_type}] {text_content}")
            else:
                ground_truth_lines.append(f"[알수없음] ({stem} 라벨링 매칭 실패)")


            
    # 결과물 저장용 폴더 생성
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    out_wav = out_path / f"{conversation_id}_merged.wav"
    out_txt = out_path / f"{conversation_id}_ground_truth.txt"
    
    # 오디오 내보내기 (오래 걸릴 수 있음)
    logging.info(f"오디오 파일 병합 및 저장 중... (총 {len(merged_audio)/1000:.1f}초 분량)")
    merged_audio.export(out_wav, format="wav")
    
    # 텍스트 내보내기 (라벨링 데이터가 있을 때만)
    if label_dir and ground_truth_lines:
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(ground_truth_lines))
        
    logging.info(f"==== 처리 완료! ====")
    logging.info(f" 🎧 업로드용 테스트 오디오: {out_wav}")
    if label_dir:
        logging.info(f" 📄 원본 비교용 대본(정답지): {out_txt}")
    else:
        logging.info(" 📄 원본 비교용 대본(정답지): 라벨링 데이터가 없어 파일이 생성되지 않았습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Hub 분할 데이터 병합기 (PoC 테스트용)")
    parser.add_argument("call_id", help="합치고 싶은 대화 ID (예: MEN0005946)")
    parser.add_argument("--base", default="base_data", help="base_data 최상위 디렉토리 위치")
    parser.add_argument("--out", default="test_ready", help="완성된 파일이 저장될 위치")
    args = parser.parse_args()
    
    merge_aihub_data(args.base, args.call_id, args.out)
