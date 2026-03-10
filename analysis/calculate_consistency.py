import pandas as pd
import os

# 1. 파일 설정
INPUT_FILE = "GPT_Scored_Results_Refined.csv"
OUTPUT_CONSISTENCY_FILE = "Final_Consistency_Report.csv"

if not os.path.exists(INPUT_FILE):
    print(f"❌ 파일을 찾을 수 없습니다: {INPUT_FILE}. 채점/보정 단계를 먼저 완료하세요.")
else:
    df = pd.read_csv(INPUT_FILE)
    languages = ["English", "KR", "JA", "AR", "WO"]
    
    # 2. 중복 쌍(Pair) 정의 (사용자 제공 범위 기반)
    # Group 1: 0~88 <-> 89~177 (오프셋 89)
    # Group 2: 178~363 <-> 364~549 (오프셋 186)
    # Group 3: 550~717 <-> 718~885 (오프셋 168)
    pairs = []
    for i in range(0, 89): pairs.append((i, i + 89))
    for i in range(178, 364): pairs.append((i, i + 186))
    for i in range(550, 718): pairs.append((i, i + 168))

    # 3. 일관성 체크 진행
    consistency_results = []
    print(f"🔍 총 {len(pairs)}개의 중복 질문 쌍에 대해 일관성 분석을 시작합니다...")

    for p1, p2 in pairs:
        try:
            row1 = df.iloc[p1]
            row2 = df.iloc[p2]
            
            entry = {
                "Pair_Range": f"{p1} & {p2}",
                "Image_File": row1['Image_File']
            }
            
            for lang in languages:
                ans_col = f"Extracted_{lang}"
                ans1 = row1[ans_col]
                ans2 = row2[ans_col]
                
                # 두 답변이 존재하고 완전히 일치하는지 확인 (A == A)
                # 추출 실패(None)인 경우는 일관성이 없는 것으로 간주 (0)
                is_consistent = (ans1 == ans2) if pd.notna(ans1) and pd.notna(ans2) else False
                entry[f"{lang}_Consistent"] = int(is_consistent)
            
            consistency_results.append(entry)
        except IndexError:
            continue

    # 4. 결과 집계 및 출력
    consistency_df = pd.DataFrame(consistency_results)
    
    summary = []
    print("\n" + "="*50)
    print("🔄 언어별 추론 일관성(Consistency) 분석 결과")
    print("="*50)
    
    for lang in languages:
        col = f"{lang}_Consistent"
        # 일관성 확률 = (일치하는 쌍의 수 / 전체 쌍의 수) * 100
        consistency_rate = consistency_df[col].mean() * 100
        print(f"🔹 {lang.upper():7s}: {consistency_rate:6.2f}% 일치")
        summary.append({"Language": lang, "Consistency_Rate": consistency_rate})

    # 5. 결과 저장
    consistency_df.to_csv(OUTPUT_CONSISTENCY_FILE, index=False, encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv("Consistency_Summary_Report.csv", index=False)
    print("="*50)
    print(f"✅ 분석 완료! 상세 리포트: {OUTPUT_CONSISTENCY_FILE}")
