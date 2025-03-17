# System_2.py
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 루트 디렉토리 및 src 폴더 추가
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.append(SRC_DIR)

from System_2_logic import (
    load_data, 
    get_optimal_ai_tax, 
    get_optimal_ai_tax_by_target, 
    macroeconomic_simulation, 
    compute_gain
)

st.set_page_config(page_title="AI 세율 거시경제 시뮬레이션", layout="wide")
st.title("AI 세율 거시경제 시뮬레이션 시스템")

# 데이터 로드 및 국가 선택
df, country_dfs = load_data()
selected_country = st.selectbox("분석할 국가 선택", list(country_dfs.keys()))
country_df = country_dfs[selected_country]
all_vars = [
    "Business sophistication", "Corporate Tax", "Creative outputs", "GDP",
    "GDP_per_capita_PPP", "GERD", "GNI_per_capita", "General Revenue",
    "Global Innovation Index", "Human capital and research", "Infrastructure",
    "Institutions", "Internet Usage", "Knowledge and technology outputs",
    "Market sophistication", "Patent Publications", "Unemployment Rate", "WIPO Tax"
]

st.markdown("### [1단계] 기본 기능 실행 (GDP 기준)")
optimal_tax, params = get_optimal_ai_tax(country_df)
if optimal_tax is None:
    st.error("기존 최적 AI 세율 도출에 실패했습니다.")
    st.stop()
optimal_tax_percent = optimal_tax * 100
st.subheader(f"{selected_country} - 기존 최적 AI 세율: {optimal_tax_percent:.2f}%")
forecast_optimal_basic = macroeconomic_simulation(selected_country, country_df, optimal_tax, optimal_tax, params)
if forecast_optimal_basic is not None:
    forecast_optimal_basic = forecast_optimal_basic.reset_index(drop=True)
    st.dataframe(forecast_optimal_basic)

st.markdown("### [2단계] 사용자 선택 변수 기준 최적 AI 세율 도출")
user_target_var = st.selectbox("최적 세율 판단 변수 선택 (GDP 제외)", options=[v for v in all_vars if v != "GDP"])
st.write("사용자 선택 변수:", user_target_var)

optimal_tax_user, user_params = get_optimal_ai_tax_by_target(country_df, user_target_var)
if optimal_tax_user is None:
    st.error("사용자 선택 변수 기반 최적 AI 세율 도출에 실패했습니다.")
    st.stop()
optimal_tax_user_percent = optimal_tax_user * 100
st.subheader(f"사용자 선택 변수({user_target_var}) 기준 최적 AI 세율: {optimal_tax_user_percent:.2f}%")
forecast_optimal_custom = macroeconomic_simulation(selected_country, country_df, optimal_tax_user, optimal_tax_user, user_params, target_var=user_target_var)
if forecast_optimal_custom is None:
    st.error("VARMAX 예측(최적 사용자 세율) 실패")
    st.stop()
forecast_optimal_custom = forecast_optimal_custom.reset_index(drop=True)
st.dataframe(forecast_optimal_custom)

st.markdown("### [3단계] 사용자 세율 조정 및 예측 비교")
user_tax_percent = st.slider("사용자 AI 세율 선택 (%)", min_value=0.0, max_value=30.0, value=optimal_tax_user_percent, step=0.1, key="ai_tax_slider")
user_tax_frac = user_tax_percent / 100
forecast_user_custom = macroeconomic_simulation(selected_country, country_df, user_tax_frac, optimal_tax_user, user_params, target_var=user_target_var)
if forecast_user_custom is None:
    st.error("사용자 세율 예측 실패")
    st.stop()
forecast_user_custom = forecast_user_custom.reset_index(drop=True)

# [3단계] 사용자 선택 변수의 5년 예측 선그래프 시각화 (사용자 선택 변수)
pred_years = list(range(int(country_df["Year"].max())+1, int(country_df["Year"].max())+6))
fig = go.Figure()
# 실제 데이터 (사용자 선택 변수)
fig.add_trace(go.Scatter(
    x=country_df["Year"], y=country_df[user_target_var],
    mode="lines+markers", name=f"실제 {user_target_var}"
))
last_year = int(country_df["Year"].max())
last_actual_val = country_df[country_df["Year"]==last_year][user_target_var].values[0]
forecast_first_val = forecast_user_custom[user_target_var].iloc[0] if user_target_var in forecast_user_custom.columns else None
fig.add_trace(go.Scatter(
    x=[last_year, pred_years[0]],
    y=[last_actual_val, forecast_first_val],
    mode="lines", name="연결선", line=dict(dash="solid")
))
fig.add_trace(go.Scatter(
    x=pred_years, y=forecast_user_custom[user_target_var] if user_target_var in forecast_user_custom.columns else [None]*5,
    mode="lines+markers", name=f"예측 {user_target_var} (사용자 세율 {user_tax_percent:.2f}%)", line=dict(dash="dot")
))
fig.add_trace(go.Scatter(
    x=pred_years, y=forecast_optimal_custom[user_target_var] if user_target_var in forecast_optimal_custom.columns else [None]*5,
    mode="lines+markers", name=f"예측 {user_target_var} (최적 세율 {optimal_tax_user_percent:.2f}%)", line=dict(dash="dot")
))
st.plotly_chart(fig, use_container_width=True)

# 전체 미래 예측값 비교 표: 사용자 선택 변수 예측 및 변화량 비교
baseline_val = country_df[user_target_var].iloc[-1]
# 사용자 세율 예측 DataFrame
user_forecast = forecast_user_custom[["Year", user_target_var]].copy()
user_forecast["Change (사용자 세율)"] = user_forecast[user_target_var] - baseline_val
# 최적 세율 예측 DataFrame
optimal_forecast = forecast_optimal_custom[["Year", user_target_var]].copy()
optimal_forecast["Change (최적 세율)"] = optimal_forecast[user_target_var] - baseline_val
# 두 DataFrame 병합 (Year 기준, 접미사 적용)
forecast_compare = pd.merge(user_forecast, optimal_forecast, on="Year", suffixes=(" (사용자 세율)", " (최적 세율)"))
forecast_compare["차이 (변화량)"] = forecast_compare["Change (최적 세율)"] - forecast_compare["Change (사용자 세율)"]
# 열 순서 재정렬
cols_order = ["Year", 
              f"{user_target_var} (사용자 세율)", f"{user_target_var} (최적 세율)",
              "Change (사용자 세율)", "Change (최적 세율)", "차이 (변화량)"]
forecast_compare = forecast_compare[cols_order].round(2)

# 스타일 적용: 양수(이득)는 빨강, 음수(손해)는 파랑
def color_gain(val):
    try:
        v = float(val)
    except:
        return ""
    if v > 0:
        return "color: red"
    elif v < 0:
        return "color: blue"
    else:
        return ""
styled_table = forecast_compare.style.applymap(color_gain, subset=["Change (사용자 세율)", "Change (최적 세율)", "차이 (변화량)"])

st.markdown("### 전체 미래 예측값 비교 (사용자 선택 변수)")
st.dataframe(styled_table, use_container_width=True)
