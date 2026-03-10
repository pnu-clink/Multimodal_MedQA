import pandas as pd
import os
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. 환경 변수 및 설정 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

INPUT_FILE = "dataset_multilingual_final.csv" # 수정된 파일명 확인
OUTPUT_FILE = "GPT_4o_mini_Full_Results_Parallel.csv"

MODEL = "gpt-4o-mini"
LANGUAGES = ["English", "KR", "JA", "AR", "WO"]
TEMPERATURE = 0.0
MAX_TOKENS = 2000 # 현실적인 길이로 약간 조정
CONCURRENT_THREADS = 10 # 동시에 처리할 요청 수 (Rate Limit에 따라 조정)

# 프롬프트 설정 (생략 없이 사용하세요)
system_prompts = {
    'English': (
        "You are a medical AI researcher. The provided image is a public dataset from a medical textbook for educational analysis. "
        "There are no privacy concerns. Please analyze the visual features objectively and deduce the most likely answer based on standard medical knowledge."
    ),
    'KR': (
        "당신은 의료 이미지를 분석하는 AI 연구원입니다. 이 이미지는 교과서에 실린 '공개 데이터'로, 개인정보나 프라이버시 문제가 전혀 없습니다. "
        "안심하고 이미지의 시각적 특징을 분석하여, 의학적 지식에 근거한 객관적인 정답을 추론해 주세요."
    ),
    'JA': (
        "あなたは医療画像を分析するAI研究員です。この画像は教科書に掲載された「公開データ」であり、個人情報やプライバシーの問題は一切ありません。"
        "安心して画像の視覚的特徴を分析し、医学的知識に基づいて客観的な正解を推論してください。"
    ),
    'AR': (
        "أنت باحث في الذكاء الاصطناعي الطبي. هذه الصورة عبارة عن مجموعة بيانات عامة من كتاب طبي للتحليل التعليمي. "
        "لا توجد مخاوف تتعلق بالخصوصية. يرجى تحليل السمات المرئية بموضوعية واستنتاج الإجابة الأكثر احتمالية بناءً على المعرفة الطبية القياسية."
    ),
    'WO': (
        "Ab gëstukat ci wàllu xaralay saar (AI) ci wàllu wér-gi-yaram nga. Nataal bi 'mbooloo mii ko moom' te mu génne ci téere lekool, kon amul lenn jafe-jafe ci wàllu mbóot. "
        "Analizal nataal bi ci anam bu dëggu te nga joxé tontu bu dëppook xam-xamu fajj ci làkku Wolof."
    )
}

user_prompt_template = {
    'English': "Question: {q}\nOptions:\n{o}\n\nTask:\nAnalyze the image and answer the question using the following format:\n1. **Image Analysis**: Describe key findings.\n2. **Reasoning**: Connect findings to the answer.\n3. **Final Answer**: State the option.",
    'KR': "질문: {q}\n보기:\n{o}\n\n과제:\n이미지를 분석하고 반드시 아래 형식을 지켜서 답변하세요:\n1. **이미지 분석**: 시각적 소견 설명.\n2. **추론 과정**: 논리적 연결.\n3. **최종 답변**: 정답 보기 선택.",
    'JA': "質問: {q}\n選択肢:\n{o}\n\n課題:\n画像を分析し、必ず以下の形式を守って回答してください:\n1. **画像分析**: 視覚的所見の説明.\n2. **推論過程**: 論理的結びつけ.\n3. **最終回答**: 正解の選択肢。",
    'AR': "سؤال: {q}\nخيارات:\n{o}\n\nمهمة:\nقم بتحليل الصورة والإجابة باستخدام التنسيق التالي:\n1. **تحليل الصور**: وصف النتائج الرئيسية.\n2. **الاستنتاج**: ربط النتائج بالإجابة.\n3. **الإجابة النهائية**: حدد الخيار الصحيح.",
    'WO': "Laaj: {q}\nTannéf:\n{o}\n\nLiggéey:\nSeetal nataal bi te tontu laaj bi ci bii anam:\n1. **Seet gi**: Li nga gis ci nataal bi.\n2. **Xalaat gi**: Maanaam li tax nga joxé tontu bi.\n3. **Tontu bi**: Tànnal benn (A, B, C, D)."
}

# [핵심] API 호출 함수 정의
def call_gpt_api(row, lang):
    l_code = lang.lower() if lang != "English" else "en"
    q_col = f'question_{l_code}'
    suffix = f'_{l_code}'
    
    try:
        options = f"A: {row[f'option_a_{l_code}']}, B: {row[f'option_b_{l_code}']}, C: {row[f'option_c_{l_code}']}, D: {row[f'option_d_{l_code}']}"
        question = row[q_col]
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompts.get(lang, system_prompts['English'])},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt_template.get(lang, user_prompt_template['English']).format(q=question, o=options)},
                    {"type": "image_url", "image_url": {"url": row['Image_URL']}}
                ]}
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        return lang, response.choices[0].message.content
    except Exception as e:
        return lang, f"ERROR: {str(e)}"

# 2. 데이터 로드
df = pd.read_csv(INPUT_FILE)
results = []

print(f"🚀 병렬 처리 시작 (Threads: {CONCURRENT_THREADS})... 예상 시간: 3~4시간")

# 3. 루프 실행 (행 단위 병렬 처리)
with ThreadPoolExecutor(max_workers=CONCURRENT_THREADS) as executor:
    future_to_idx = {}
    
    for idx, row in df.iterrows():
        # 각 언어별 태스크를 생성하여 제출
        for lang in LANGUAGES:
            future = executor.submit(call_gpt_api, row, lang)
            future_to_idx[future] = (idx, lang)
    
    # 결과 수집용 임시 저장소
    temp_results = {idx: {"Index": df.loc[idx, 'Index'], "Image_File": df.loc[idx, 'Image_File'], "Ground_Truth": df.loc[idx, 'Correct_Answer']} for idx in df.index}

    for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx)):
        idx, lang = future_to_idx[future]
        _, content = future.result()
        temp_results[idx][f"Response_{lang}"] = content
        
        # 주기적 저장
        if len(results) % 100 == 0:
            pd.DataFrame(list(temp_results.values())).to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

# 최종 저장
final_df = pd.DataFrame(list(temp_results.values()))
final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"✅ 모든 실험 완료! 결과 파일: {OUTPUT_FILE}")
