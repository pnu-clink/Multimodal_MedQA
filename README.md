# Performance and Cross-lingual Consistency Analysis of GPT-4o-mini in Multilingual Medical VQA

이 저장소는 의료 시각 질문 답변(VQA) 환경에서 **GPT-4o-mini** 모델의 다국어 성능과 언어 간 판단 일관성(Cross-lingual Consistency)을 분석한 연구 코드를 담고 있습니다.

## 📌 연구 배경
동일한 의료 영상(Radiology, ECG, Pathology 등)을 보더라도 질문의 언어에 따라 모델의 진단 결과가 달라진다면, 실제 임상 환경에서 모델의 신뢰성에 심각한 문제를 야기할 수 있습니다. 본 연구는 5개 언어(영어, 한국어, 일본어, 아랍어, 월로프어)를 통해 모델의 언어 간 일관성을 정량적으로 평가합니다.

## 📊 데이터셋: M3-MedQA
본 연구에서 구축한 데이터셋은 HuggingFace에서 확인할 수 있습니다.
- **Dataset Link:** [huggingface.co/datasets/parksh03/Multimodal_MedQA](https://huggingface.co/datasets/parksh03/Multimodal_MedQA)
- **규모:** 1,000개의 의료 사례 (이미지 + 5개 국어 질문/선택지)
- **포맷:** Apache Parquet (Image Embedded)

## 📂 프로젝트 구조
- `evaluation/`: GPT-4o-mini API를 연동하여 5개 언어별 정답률을 측정하는 스크립트 
- `analysis/`: 언어 간 답변 일치율(Consistency Score) 분석 및 시각화 도구 

## 🛠️ 시작하기
```bash
# 저장소 복제
git clone [https://github.com/parksh03/Multimodal_MedQA.git](https://github.com/parksh03/Multimodal_MedQA.git)

# 필수 라이브러리 설치
pip install -r requirements.txt
```
