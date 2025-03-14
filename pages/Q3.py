import os
import sys
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# 루트 디렉토리 및 src 폴더 추가
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.append(SRC_DIR)

from Q3_logic import (
    load_data,
    synthetic_control,
    macroeconomic_simulation
)

# 페이지 설정
st.set_page_config(page_title="Q3: 거시경제 효과 분석", layout="wide")

report_md = """
# **Q3: AI 세율을 정책적으로 적용했을 때의 거시경제적 효과는 무엇인가?**

## **현재 AI 세율이 도입되지 않은 상황**
- 현재 AI 세율은 실제로 도입되지 않아, 도입된 국가와 도입되지 않은 국가를 직접 비교하는 DID 분석 등 전통적 접근이 불가능
- 따라서 보유한 데이터를 활용하여, AI 세율 도입 시 경제 성장과 기업 투자에 미치는 영향을 **시뮬레이션**하는 방법을 사용

---

## **1️⃣ 분석 방법**
1. **Counterfactual Analysis (반사실 분석)**
   - 각 국가별 최적 AI 세율 Proxy 및 Laffer Curve를 활용하여 GDP 변화를 추정하고,
   - 실제 2022년 GDP를 기준으로 예상 변화율(%)과 변화량(달러)을 계산
2. **Synthetic Control Method (합성 통제법)**
   - Ridge 회귀 등을 사용하여 AI 세율이 적용된 “가상의 국가”를 생성하고,
   - 실제 GDP와 Synthetic GDP의 연도별 차이를 계산하여, 표(연도별 차이와 맨 아래 평균, 최소, 최대 차이 포함)와 그래프로 제공
3. **거시경제 시뮬레이션 (Macroeconomic Simulation)**
   - VAR 모델을 사용해 2013년부터 2022년까지의 실제 GDP와, 2023년부터 2027년까지의 예측 GDP를 산출
   - 2022년 실제 GDP와 2023년 예측 GDP를 붉은 점선으로 연결하고, 전체 그래프에 최대/최소 값을 표시
   - 실제 2022년 GDP를 기준으로 변화량 및 변화율을 계산

---

## **2️⃣ 분석 결과 및 시각화**
- **Counterfactual Analysis**: 선택한 국가의 AI 세율 도입 시 예상 GDP 변화율 및 변화량 표시
- **Synthetic Control Method**: 실제 GDP와 Synthetic GDP의 차이를 연도별로 나타내는 표와 그래프로 시각화
- **Macroeconomic Simulation**: 실제 GDP 추세(2013~2022년)와 예측 GDP 추세(2023~2027년)가 연결하여 시각화

---

## **3️⃣ 결론 및 정책적 시사점**
1. 국가별 AI 세율 도입 효과가 상이하므로 맞춤형 정책이 필요
2. AI 세율은 단일 고정값이 아니라 경제 및 기술 발전에 따라 동적으로 조정되어야 함
3. AI 투자와 연구개발 촉진을 위한 보완 정책도 함께 마련되어야 함
"""
st.markdown(report_md)

# 데이터 로드
df, country_dfs = load_data()

# 사용자 입력: 분석할 국가 선택
selected_country = st.selectbox("분석할 국가를 선택하세요", list(country_dfs.keys()))
country_df = country_dfs[selected_country]


# ----------------------------
# Synthetic Control Method 실행 및 시각화
st.markdown("### Synthetic Control 결과")
synthetic_table, synthetic_fig = synthetic_control(selected_country, country_df)
# 인덱스 제거: reset_index() 후 drop
st.dataframe(synthetic_table.reset_index(drop=True))
st.plotly_chart(synthetic_fig, use_container_width=True)

# ----------------------------
# Macroeconomic Simulation 실행 및 시각화
st.markdown("### Macroeconomic Simulation 결과")
forecast_fig, forecast_gdp_diff, forecast_gdp_change = macroeconomic_simulation(selected_country, country_df)
st.plotly_chart(forecast_fig, use_container_width=True)
# 실제 2022년 GDP 표시
if np.issubdtype(country_df["Year"].dtype, np.datetime64):
    actual_2022 = country_df[pd.to_datetime(country_df["Year"]).dt.year == 2022]["GDP"].values
else:
    actual_2022 = country_df[country_df["Year"] == 2022]["GDP"].values
if len(actual_2022) > 0:
    actual_2022_gdp = actual_2022[0]
    st.markdown(f"**실제 2022년 GDP:** {actual_2022_gdp:,.2f} 달러")

st.markdown(f"**5년 후 예측 GDP 변화량 (2022년 기준):** {forecast_gdp_diff:,.2f} 달러")
st.markdown(f"**Counterfactual Analysis 결과:** {forecast_gdp_change:.2f}%")

