import pandas as pd
import os

# 1. 파일 로드 (이중 검수가 완료된 파일)
INPUT_FILE = "GPT_Scored_Results_Refined.csv"

if not os.path.exists(INPUT_FILE):
    print(f"❌ 파일을 찾을 수 없습니다: {INPUT_FILE}. 이중 검수 코드를 먼저 실행해주세요.")
else:
    df = pd.read_csv(INPUT_FILE)
    languages = ["English", "KR", "JA", "AR", "WO"]
    truth_col = 'Correct_Answer' if 'Correct_Answer' in df.columns else 'Ground_Truth'

    report_data = []

    print("📊 정확도(Accuracy) 반영 최종 분석 중...")

    for lang in languages:
        resp_col = f"Response_{lang}"
        ext_col = f"Extracted_{lang}"
        score_col = f"Score_{lang}"
        
        total = len(df)
        
        # 1. API 실패 횟수 (기회조차 없었던 경우)
        api_failures = df[resp_col].apply(lambda x: 1 if pd.isna(x) or "ERROR" in str(x) else 0).sum()
        
        # 2. 유효 응답 수 (전체 - API 실패)
        valid_responses = total - api_failures
        
        # 3. 추출 성공 횟수 (A, B, C, D 중 하나라도 뽑힌 경우)
        extracted_success = df[ext_col].notna().sum()
        
        # 4. 추출 실패 횟수 (답변은 했으나 선택지를 안 고른 경우)
        extraction_fails = valid_responses - extracted_success
        
        # 5. 정답 횟수
        correct = df[score_col].sum()
        
        # [계산] 전체 정확도 vs 실질 정확도
        total_accuracy = (correct / total) * 100
        adjusted_accuracy = (correct / valid_responses) * 100 if valid_responses > 0 else 0
        
        report_data.append({
            "Language": lang,
            "Total": total,
            "API Failures": api_failures,
            "Valid Resp.": valid_responses,
            "Correct": correct,
            "Total Acc (%)": round(total_accuracy, 2),
            "Adjusted Acc (%)": round(adjusted_accuracy, 2)
        })

    # 5. 결과 요약 출력
    report_df = pd.DataFrame(report_data)
    print("\n" + "="*85)
    print("🏆 GPT-4o-mini 다국어 의료 추론 실질 정확도 성적표")
    print("="*85)
    # 가독성을 위해 출력 컬럼 조정
    print(report_df.to_string(index=False))
    print("="*85)
    print("💡 Adjusted Acc (%): API 에러를 제외하고 모델이 '실제로 답변한' 문제 중 정답률")

    # 6. 결과 저장
    report_df.to_csv("Final_Adjusted_Accuracy_Report.csv", index=False, encoding='utf-8-sig')
    print(f"\n✅ 분석 완료! 'Final_Adjusted_Accuracy_Report.csv'를 확인하세요.")
