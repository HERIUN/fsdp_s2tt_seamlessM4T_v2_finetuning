from google.cloud import texttospeech
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from pydub import AudioSegment
from io import BytesIO
import math
from tqdm import tqdm

load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
google_client = texttospeech.TextToSpeechClient()
openai_client = OpenAI()

lang_code_dict = {
    "ko": "ko-KR",
    "en": "en-US",
    "ja": "ja-JP",
    "zh": "cmn-CN",
    "es": "es-ES",
}

# 오디오 및 JSON 저장 경로 설정 (절대 경로로 변환)
BASE_DIR = os.path.abspath("s2tt_data")
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
JSON_FILE = os.path.join(BASE_DIR, "out_json.json")

# 디렉터리 자동 생성
os.makedirs(AUDIO_DIR, exist_ok=True)


def google_tts(text, lang_code, voice_type, sr=16000):
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # 음성 설정 : ja-JP-Neural2-B~D, ja-JP-Wavenet-A~D, ja-JP-Standard-A~D
    # 	ko-KR-Standard-A~D
    if voice_type.endswith(("A", "B")):
        voice = texttospeech.VoiceSelectionParams(
            language_code=lang_code_dict[lang_code],
            name=voice_type,
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
        )
    else:
        voice = texttospeech.VoiceSelectionParams(
            language_code=lang_code_dict[lang_code],
            name=voice_type,
            ssml_gender=texttospeech.SsmlVoiceGender.MALE,
        )

    # 오디오 설정
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        sample_rate_hertz=16000
    )
    # TTS 요청 및 응답
    response = google_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    return response


def make_lang_pair_text(n_sample=10, max_pairs_per_request=10): # ko <-> en
    all_sentence_pairs = []
    # 필요한 요청 수 계산 (예: n_sample=25, max_pairs_per_request=10이면 3번 요청)
    num_requests = math.ceil(n_sample / max_pairs_per_request)

    for i in range(num_requests):
        # 이번 요청에서 처리할 문장 쌍 개수 결정
        start_idx = i * max_pairs_per_request + 1
        end_idx = min((i + 1) * max_pairs_per_request, n_sample)
        batch_n = end_idx - start_idx + 1
        prompt = f"""한국 관광에 대한 한국어 문장 {batch_n}개와 그에 대한 영어 번역 문장을 만들어 JSON 형식으로 반환해줘.
            반드시 아래 형식을 따르도록 응답해:
            {{
            "sentence_pairs": [
                {{"ko": "한국어 문장{start_idx}","en": "영어 번역 문장{start_idx}"}},
                {{"ko": "한국어 문장{start_idx+1}","en": "영어 번역 문장{start_idx+1}"}},
                ...
                {{"ko": "한국어 문장{end_idx}","en": "영어 번역 문장{end_idx}"}}
            ]
            }}"""
        response_format = {
            "type" : "json_schema",
            "json_schema": {
                "name": "sentence_pair_list",
                "schema": {
                    "type": "object",
                    "properties": {
                        "sentence_pairs": {
                            "description": f"A list of {batch_n} Korean-English sentence pairs",
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "kor": {
                                        "type": "string",
                                        "description": "Korean sentence"
                                    },
                                    "eng": {
                                        "type": "string",
                                        "description": "English translation"
                                    }
                                },
                                "required": ["kor", "eng"],
                                "additionalProperties": False
                            },
                        },
                    },
                    "required": ["sentence_pairs"],
                    "additionalProperties": False
                }
            }
        }

        messages = [{"role": "user", "content": prompt}]
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format=response_format
        )

        response_text = completion.choices[0].message.content.strip()
        try:
            response_json = json.loads(response_text)
        except Exception as e:
            raise RuntimeError(f"JSON 파싱 실패: {e}\n응답 내용: {response_text}")
        all_sentence_pairs.extend(response_json["sentence_pairs"])
    
    return all_sentence_pairs


def main(n_sample=100):
    print("make lang pair sentences")
    sentences = make_lang_pair_text(n_sample, max_pairs_per_request=10)
    print("done")
    result = []
    for i,s in enumerate(tqdm(sentences, desc="processing TTS")):
        response = google_tts(s["kor"], lang_code="ko", voice_type="ko-KR-Standard-A")
        audio_stream = BytesIO(response.audio_content)
        audio = AudioSegment.from_mp3(audio_stream)
        out_path = os.path.abspath(os.path.join(AUDIO_DIR, f"output_{i}.wav"))
        audio.export(out_path, format="wav")

        s["audio_path"] = out_path
        result.append(s)

    with open("s2tt_data/out_json.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    n_sample = 100
    main(n_sample)