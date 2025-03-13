import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Q1: AI 세금의 효과 분석", layout="wide")

report_md = """
# Q1: AI 세금이 기업 투자 및 경제 성장에 미치는 영향

본 연구는 AI 세금(AI-specific tax)이 직접적으로 존재하지 않으므로,  
대리 변수(Proxy Variable)를 활용하여 AI 세금이 기업 투자와 경제 성장에 미치는 영향을 분석하였다.

## 1. 분석 개요
- **AI 세금(Proxy)**: WIPO Tax, Corporate Tax  
- **기업 투자(Proxy)**: Patent Publications, GERD (Gross Expenditure on R&D)  
- **경제 성장(Proxy)**: GDP, GDP_per_capita_PPP  

## 2. 분석 목표
- **기업 투자 분석**: WIPO Tax와 Corporate Tax가 기업 투자(특히, Patent Publications, GERD)에 미치는 영향 분석  
- **경제 성장 분석**: AI 관련 세금(WIPO Tax)이 GDP 및 GDP_per_capita_PPP에 미치는 영향 확인

## 3. 분석 방법 및 모델
- 고정 효과(FE) vs. 랜덤 효과(RE) 패널 모델 비교  
- Hausman Test를 통한 모델 적합성 평가  
- GDP 모델에 1기 시차(GDP Lag) 추가 및 Arellano-Bond GMM 적용  
- 시각화: 좌측은 Fixed Effects 모델 잔차 플롯, 우측은 Random Effects 모델 잔차 플롯

## 4. 결과 요약
- **Hausman Test 결과**: 두 모델 모두 P-값이 높아 귀무가설 기각 불가 → 랜덤효과 모델이 적합  
- **상관 행렬 분석**: GDP_lag와 WIPO Tax 간 음의 상관관계, WIPO Tax와 Corporate Tax 간 양의 상관관계  
- **모델 결과**: 동적 패널 모델의 R-squared 값이 낮으나, Corporate Tax의 유의미한 P-값(0.0362)을 확인

"""

st.markdown(report_md)

st.subheader("상관 행렬 결과")

corr_data = {
    "Category": ["GDP_lag", "WIPO Tax", "Corporate Tax"],
    "GDP_lag": [1.0000, -0.5499, 0.1009],
    "WIPO Tax": [-0.5499, 1.0000, 0.3690],
    "Corporate Tax": [0.1009, 0.3690, 1.0000]
}
corr_df = pd.DataFrame(corr_data).set_index("Category")
st.table(corr_df)
st.markdown("""
**상관 행렬 해석:**  
- GDP_lag와 WIPO Tax 간에는 -0.55의 음의 상관관계가 존재하여, 두 변수는 반대 방향으로 움직인다.  
- WIPO Tax와 Corporate Tax는 0.37의 양의 상관관계를 보여 함께 증가하는 경향이 있다.
""")

# 3. GDP 결과 (모델 결과)
st.subheader("GDP 모델 결과")

gdp_model_data = {
    "파라미터": ["const", "WIPO Tax", "Corporate Tax"],
    "표준 오차": ["1.72e+11", "3.046e+10", "-3.041e+10"],
    "t-통계량": ["1.092e+12", "3.929e+10", "1.416e+10"],
    "P-값": ["0.1575", "0.7753", "-2.1480"],
    "하한 CI": ["-2.018e+12", "-4.831e+10", "-5.879e+10"],
    "상한 CI": ["2.362e+12", "1.092e+11", "-2.026e+09"]
}
gdp_model_df = pd.DataFrame(gdp_model_data)
st.table(gdp_model_df)
st.markdown("""
**GDP 모델 해석:**  
- 본 모델에서는 Corporate Tax의 P-값이 유의미(0.0362)로 나타나, 해당 변수가 경제 성장에 중요한 역할을 하는 것으로 분석
""")

st.subheader("결론 및 정책적 시사점")
st.markdown("""
- **모델 요약:**  
            Hausman 테스트 결과, 고정효과 모델과 랜덤효과 모델 중 랜덤효과 모델이 적합한 것으로 나타났으며,  
            동적 패널 모델을 통해 AI 세금이 기업 투자 및 경제 성장에 미치는 영향을 정량적으로 평가함

- **정책 시사점:**  
            AI 세금이 존재하지 않는 현 상황에서, Proxy 변수를 활용한 분석 결과  
            각 국가별로 AI 세금 도입이 경제 성장과 기업 투자에 미치는 영향이 상이하게 나타남을 확인  
            이에 따라 AI 세금 도입 시, 국가별 맞춤형 정책 설계가 필요

- **향후 연구:**  
            AI 세금이 기업 혁신 투자 및 노동시장에 미치는 영향을 포함한 다각적인 분석이 필요
""")

st.subheader("분석 시각 자료")
st.image("./jupyternotebooks/R_Hausman_Test.png", caption="Figure: Hausman Test 결과")