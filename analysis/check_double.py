import pandas as pd
import os
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import time

# 1. 환경 변수 및 설정 로드
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_FILE = "GPT_Scored_Results_Final.csv"
OUTPUT_FILE = "GPT_Scored_Results_Refined.csv"

# 모델 설정
REFINE_MODEL = "gpt-4o-mini"
LANGUAGES = ["English", "KR", "JA", "AR", "WO"]

def refine_extraction_with_llm(raw_text):
    """GPT를 이용해 텍스트 내에서 최종 정답 알파벳만 추출"""
    if pd.isna(raw_text) or "ERROR" in str(raw_text):
        return None
    
    try:
        response = client.chat.completions.create(
            model=REFINE_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts the final answer from a medical analysis report. Extract only the single letter (A, B, C, or D) that represents the final chosen option. If no answer is found, respond with 'None'."},
                {"role": "user", "content": f"Extract the final option (A, B, C, or D) from this response:\n\n{raw_text}"}
            ],
            max_tokens=5,
            temperature=0.0
        )
        result = response.choices[0].message.content.strip().upper()
        # 결과가 A, B, C, D 중 하나인지 확인
        if result in ['A', 'B', 'C', 'D']:
            return result
        return None
    except:
        return None

# 2. 데이터 로드
if not os.path.exists(INPUT_FILE):
    print(f"❌ 파일을 찾을 수 없습니다: {INPUT_FILE}")
else:
    df = pd.read_csv(INPUT_FILE)
    truth_col = 'Correct_Answer' if 'Correct_Answer' in df.columns else 'Ground_Truth'

    print("🔍 정규표현식 추출 실패 건들에 대해 GPT 이중 검수 시작...")

    for lang in LANGUAGES:
        ans_col = f"Extracted_{lang}"
        resp_col = f"Response_{lang}"
        score_col = f"Score_{lang}"
        
        # 추출 실패(None)인 행들만 필터링
        fail_mask = df[ans_col].isna() & df[resp_col].notna() & (~df[resp_col].str.contains("ERROR", na=False))
        fail_indices = df[fail_mask].index
        
        if len(fail_indices) > 0:
            print(f"🔄 {lang}: {len(fail_indices)}건의 미검출 답변 재검토 중...")
            for idx in tqdm(fail_indices):
                refined_ans = refine_extraction_with_llm(df.loc[idx, resp_col])
                if refined_ans:
                    df.loc[idx, ans_col] = refined_ans
                    # 점수 재계산
                    df.loc[idx, score_col] = 1 if refined_ans == df.loc[idx, truth_col] else 0
                
                # API 부하 방지를 위한 미세 대기
                time.sleep(0.1)

    # 3. 최종 정확도 재산출
    print("\n📊 --- 이중 검수 후 최종 정확도 결과 ---")
    summary = []
    for lang in LANGUAGES:
        acc = df[f"Score_{lang}"].mean() * 100
        print(f"✅ {lang}: {acc:.2f}% (보정 완료)")
        summary.append({"Language": lang, "Refined_Accuracy": acc})

    # 4. 결과 저장
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv("Refined_Accuracy_Summary.csv", index=False)
    print(f"\n✨ 모든 보정 작업 완료! 결과 파일: {OUTPUT_FILE}")
