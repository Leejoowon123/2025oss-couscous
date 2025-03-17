import os
import sys
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# 루트 디렉토리 및 src 폴더 추가
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.append(SRC_DIR)

from System_logic import load_data, get_optimal_ai_tax, macroeconomic_simulation, compute_changes

st.set_page_config(page_title="AI 세율 정책 시뮬레이션", layout="wide")
st.title("AI 세율 정책 시뮬레이션 시스템")

# 데이터 로드
df, country_dfs = load_data()

# 사용자 입력: 국가 선택
selected_country = st.selectbox("분석할 국가를 선택하세요", list(country_dfs.keys()))
country_df = country_dfs[selected_country]

# 최적 AI 세율 산출 (Q2 방식 적용)
optimal_tax, params = get_optimal_ai_tax(country_df)
if optimal_tax is None:
    st.error("최적 AI 세율을 산출할 수 없습니다.")
    st.stop()
optimal_tax_percent = optimal_tax * 100
st.subheader(f"{selected_country} - 최적 AI 세율: {optimal_tax_percent:.2f}%")

# 사용자 입력: AI 세율 조정 (슬라이더)
ai_tax_percent = st.slider("AI 세율 입력 (%)", min_value=0.0, max_value=30.0, value=optimal_tax_percent, step=0.1, key="ai_tax_slider")
ai_tax = ai_tax_percent / 100  # 소수

# Macroeconomic Simulation 실행: 사용자와 최적 세율 각각 적용
st.markdown("### Macroeconomic Simulation 결과")
forecast_user = macroeconomic_simulation(selected_country, country_df, ai_tax, optimal_tax, params)
forecast_optimal = macroeconomic_simulation(selected_country, country_df, optimal_tax, optimal_tax, params)

# GDP 시각화: 실제 GDP, 사용자 AI 세율 예측, 최적 AI 세율 예측 및 2022-2023 연결 선 (실제 연결선은 파란 실선)
if forecast_user is not None:
    pred_years = list(range(country_df["Year"].max()+1, country_df["Year"].max()+6))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=country_df["Year"], y=country_df["GDP"],
        mode="lines+markers", name="실제 GDP (2013-2022)",
        line=dict(color="blue")
    ))
    # 2022년 실제와 2023년 예측값 연결 (파란 실선)
    last_year = country_df["Year"].max()
    last_actual_gdp = country_df[country_df["Year"]==last_year]["GDP"].values[0]
    forecast_2023_user = forecast_user["GDP"].iloc[0]
    fig.add_trace(go.Scatter(
        x=[last_year, pred_years[0]],
        y=[last_actual_gdp, forecast_2023_user],
        mode="lines", name="연결선", line=dict(color="blue")
    ))
    fig.add_trace(go.Scatter(
        x=pred_years, y=forecast_user["GDP"],
        mode="lines+markers", name=f"예측 GDP (AI 세율 {ai_tax_percent:.2f}%)",
        line=dict(color="red", dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=pred_years, y=forecast_optimal["GDP"],
        mode="lines+markers", name=f"예측 GDP (최적 AI 세율 {optimal_tax_percent:.2f}%)",
        line=dict(color="green", dash="dot")
    ))
    st.plotly_chart(fig, use_container_width=True, key="gdp_chart")
    
    # 미래 예측값 표: 사용자 vs 최적, 열 순서: GDP(사용자), GDP(최적), GERD(사용자), GERD(최적), Unemployment Rate(사용자), Unemployment Rate(최적)
    forecast_user_df = forecast_user.set_index("Year")
    forecast_optimal_df = forecast_optimal.set_index("Year")
    combined_df = pd.DataFrame({
        "GDP (사용자)": forecast_user_df["GDP"],
        "GDP (최적)": forecast_optimal_df["GDP"],
        "GERD (사용자)": forecast_user_df["GERD"] if "GERD" in forecast_user_df.columns else np.nan,
        "GERD (최적)": forecast_optimal_df["GERD"] if "GERD" in forecast_optimal_df.columns else np.nan,
        "Unemployment Rate (사용자)": forecast_user_df["Unemployment Rate"] if "Unemployment Rate" in forecast_user_df.columns else np.nan,
        "Unemployment Rate (최적)": forecast_optimal_df["Unemployment Rate"] if "Unemployment Rate" in forecast_optimal_df.columns else np.nan
    })
    
    # 전년 대비 변화량 표: 실제 2022년 값을 첫 행으로 추가 후 diff() 적용
    actual_values = {}
    for var in combined_df.columns:
        # column name은 e.g., "GDP (사용자)"
        base_var = var.split(" ")[0]
        if base_var in country_df.columns:
            actual_values[var] = country_df[base_var].iloc[-1]
        else:
            actual_values[var] = np.nan
    actual_series = pd.Series(actual_values)
    combined_with_actual = pd.concat([actual_series.to_frame().T, combined_df])
    changes = combined_with_actual.diff().iloc[1:]  # 첫 행 변화: forecast2023 - actual
    # 스타일: 일반 변수 - 양수: 빨강, 음수: 파랑; Unemployment Rate 반대로
    def style_change(val, col_name):
        try:
            val = float(val)
        except:
            return ""
        if "Unemployment Rate" in col_name:
            return "color: red" if val < 0 else ("color: blue" if val > 0 else "")
        else:
            return "color: red" if val > 0 else ("color: blue" if val < 0 else "")
    styled_changes = changes.style.apply(lambda col: [style_change(v, col.name) for v in col], axis=0)
    
    # 재구성: 최종 표는 Year 제외하고, 행 순서: 사용자 예측, 최적 예측, 변화량 (전년 대비)
    final_table = pd.concat([
        combined_df.iloc[0:1].rename(index={combined_df.index[0]:"2023 예측값"}),
        combined_df.iloc[1:].rename(index=lambda x: f"예측값 {x}")
    ])
    st.subheader("미래 5년 예측값 (사용자 vs 최적)")
    st.dataframe(final_table)
    
    st.subheader("전년 대비 변화량")
    st.dataframe(styled_changes)

    # 전체 미래 예측값 테이블 (사용자 결과만)
    st.subheader("미래 5년 예측값 (전체, 사용자)")
    forecast_user_display = forecast_user_df.copy().round(2)
    st.table(forecast_user_display)