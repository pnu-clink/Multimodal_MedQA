import pandas as pd
import os

# 1. 파일 설정 (1단계의 결과물을 Input으로 사용)
INPUT_FILE = "GPT_Scored_Results_Final.csv"
OUTPUT_CONSISTENCY_FILE = "GPT_Consistency_Analysis.csv"

if not os.path.exists(INPUT_FILE):
    print(f"❌ 채점된 파일을 찾을 수 없습니다. 1단계를 먼저 실행하세요.")
else:
    df = pd.read_csv(INPUT_FILE)
    
    # 2. 알려주신 중복 행 범위 매핑
    pairs = []
    # 그룹 1: 0~88 <-> 89~177 (오프셋 89)
    for i in range(0, 89): pairs.append((i, i + 89))
    # 그룹 2: 178~363 <-> 364~549 (오프셋 186)
    for i in range(178, 364): pairs.append((i, i + 186))
    # 그룹 3: 550~717 <-> 718~885 (오프셋 168)
    for i in range(550, 718): pairs.append((i, i + 168))

    # 3. 일관성 체크 진행
    languages = ["English", "KR", "JA", "AR", "WO"]
    consistency_results = []

    print(f"🔍 총 {len(pairs)}개의 중복 질문 쌍 분석 중...")

    for p1, p2 in pairs:
        try:
            row1 = df.iloc[p1]
            row2 = df.iloc[p2]
            
            entry = {
                "Pair_Range": f"{p1} & {p2}",
                "Image": row1['Image_File']
            }
            
            for lang in languages:
                ans_col = f"Extracted_{lang}"
                ans1 = row1[ans_col]
                ans2 = row2[ans_col]
                
                # 추출된 정답 글자(A, B, C, D)가 서로 같은지 비교
                is_consistent = (ans1 == ans2) if pd.notna(ans1) and pd.notna(ans2) else False
                entry[f"{lang}_Consistent"] = int(is_consistent)
            
            consistency_results.append(entry)
        except IndexError:
            continue

    # 4. 일관성 통계 요약
    consistency_df = pd.DataFrame(consistency_results)
    
    print("\n📈 --- 언어별 추론 일관성(Consistency) 결과 ---")
    summary = []
    for lang in languages:
        col = f"{lang}_Consistent"
        rate = consistency_df[col].mean() * 100
        print(f"🔄 {lang}: {rate:.2f}% 일치")
        summary.append({"Language": lang, "Consistency_Rate": rate})

    # 5. 결과 저장
    consistency_df.to_csv(OUTPUT_CONSISTENCY_FILE, index=False, encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv("Consistency_Summary.csv", index=False)
    print(f"\n✅ 분석 완료! 상세 리포트: {OUTPUT_CONSISTENCY_FILE}")
