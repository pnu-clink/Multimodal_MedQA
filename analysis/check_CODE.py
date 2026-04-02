import pandas as pd
import re
import os

# 1. 파일 설정
INPUT_FILE = "GPT_4o_mini_Full_Results_Parallel.csv"  # 실험 결과 원본 파일
OUTPUT_SCORED_FILE = "GPT_Scored_Results_Final.csv"

if not os.path.exists(INPUT_FILE):
    print(f"❌ 파일을 찾을 수 없습니다: {INPUT_FILE}")
else:
    df = pd.read_csv(INPUT_FILE)

    # 2. 정답 추출 함수
    def extract_answer(text):
        if pd.isna(text) or "ERROR" in str(text):
            return None
        
        text_str = str(text)
        # 다양한 언어별 키워드 대응
        pattern = r"(?:Final Answer|최종 답변|最終回答|الإجابة النهائية|Tontu bi)[:\s\*]*([A-D])"
        match = re.search(pattern, text_str, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        match = re.search(r"(?:answer is|정답은|正解は)[:\s]*([A-D])", text_str, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        match = re.search(r"\b([A-D])\b", text_str.strip()[-15:])
        if match:
            return match.group(1).upper()
            
        return None

    # 3. 채점 및 실패 분석 진행
    languages = ["English", "KR", "JA", "AR", "WO"]
    truth_col = 'Ground_Truth' if 'Ground_Truth' in df.columns else 'Correct_Answer'
    
    summary = []
    
    for lang in languages:
        resp_col = f"Response_{lang}"
        ans_col = f"Extracted_{lang}"
        score_col = f"Score_{lang}"
        fail_col = f"Is_Failure_{lang}" # 실패 여부 체크 컬럼
        
        if resp_col in df.columns:
            # A. 실패 여부 확인 ("ERROR" 문자열 포함 여부)
            df[fail_col] = df[resp_col].apply(lambda x: 1 if "ERROR" in str(x) else 0)
            
            # B. 정답 추출
            df[ans_col] = df[resp_col].apply(extract_answer)
            
            # C. 채점 (실패하지 않은 경우에만 비교)
            df[score_col] = (df[ans_col] == df[truth_col]).astype(int)
            
            # 통계 계산
            total_count = len(df)
            fail_count = df[fail_col].sum()
            success_count = total_count - fail_count
            correct_count = df[score_col].sum()
            
            # 정확도는 전체 데이터 대비가 아닌 '성공한 호출' 대비로 볼 수도 있으나, 
            # 연구 목적에 따라 전체 대비(Total Accuracy)로 산출합니다.
            accuracy = (correct_count / total_count) * 100
            
            summary.append({
                "Language": lang,
                "Total": total_count,
                "Success": success_count,
                "Failures": fail_count,
                "Accuracy (%)": round(accuracy, 2)
            })
        else:
            print(f"⚠️ {resp_col} 컬럼이 없습니다.")

    # 4. 분석 결과 출력
    print("\n📊 --- 언어별 종합 분석 결과 ---")
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

    # 5. 결과 저장
    df.to_csv(OUTPUT_SCORED_FILE, index=False, encoding='utf-8-sig')
    summary_df.to_csv("Detailed_Accuracy_Summary.csv", index=False)
    print(f"\n✅ 분석 완료! 상세 리포트: Detailed_Accuracy_Summary.csv")
