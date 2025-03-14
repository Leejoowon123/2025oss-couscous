import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.append(SRC_DIR)

from System_logic import load_data, get_optimal_ai_tax, macroeconomic_simulation, predict_variable_change_rates

st.set_page_config(page_title="AI 세율 정책 시뮬레이션", layout="wide")

df, country_dfs = load_data()

st.title("AI 세율 정책 시뮬레이션 시스템")

selected_country = st.selectbox("분석할 국가를 선택하세요", list(country_dfs.keys()))
country_df = country_dfs[selected_country]

optimal_tax, _, best_proxy, mod_country_df = get_optimal_ai_tax(country_df)
if optimal_tax is None:
    st.error("최적 AI 세율을 산출할 수 없습니다.")
    st.stop()
optimal_tax_percent = optimal_tax * 100
st.subheader(f"{selected_country} - 최적 AI 세율: {optimal_tax_percent:.2f}%")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=mod_country_df["Year"], y=mod_country_df["GDP"],
    mode="lines+markers", name="실제 GDP (2013-2022)",
    line=dict(color="blue")
))

var_list = [
    "GDP_per_capita_PPP",
    "GNI_per_capita", "General Revenue", "Global Innovation Index",
    "Human capital and research", "Unemployment Rate"
]
latest_year = mod_country_df["Year"].max()
actual_vars = mod_country_df[mod_country_df["Year"] == latest_year]
actual_vars = actual_vars[[col for col in var_list if col in actual_vars.columns]]

ai_tax = st.number_input("AI 세율 입력 (%)", min_value=0.0, max_value=30.0, value=0.0, step=0.01, format="%.2f")

if ai_tax > 0:
    forecast_df_user = macroeconomic_simulation(selected_country, mod_country_df, ai_tax)
    if forecast_df_user is not None:
        pred_years = list(range(latest_year + 1, latest_year + 6))
        fig.add_trace(go.Scatter(
            x=pred_years, y=forecast_df_user["GDP"],
            mode="lines+markers", name=f"예측 GDP (AI 세율 {ai_tax:.2f}%)",
            line=dict(color="red", dash="dot")
        ))
        actual_2022 = mod_country_df[mod_country_df["Year"] == latest_year]["GDP"].values[0]
        forecast_2023_user = forecast_df_user["GDP"].iloc[0]
        fig.add_trace(go.Scatter(
            x=[latest_year, pred_years[0]],
            y=[actual_2022, forecast_2023_user],
            mode="lines", line=dict(color="red", dash="dot"),
            showlegend=False
        ))
    
    if abs(ai_tax - optimal_tax_percent) > 0.01 and forecast_df_user is not None:
        forecast_df_optimal = macroeconomic_simulation(selected_country, mod_country_df, optimal_tax_percent)
        if forecast_df_optimal is not None:
            fig.add_trace(go.Scatter(
                x=pred_years, y=forecast_df_optimal["GDP"],
                mode="lines+markers", name=f"예측 GDP (최적 AI 세율 {optimal_tax_percent:.2f}%)",
                line=dict(color="green", dash="dot")
            ))
            forecast_2023_opt = forecast_df_optimal["GDP"].iloc[0]
            fig.add_trace(go.Scatter(
                x=[latest_year, pred_years[0]],
                y=[actual_2022, forecast_2023_opt],
                mode="lines", line=dict(color="green", dash="dot"),
                showlegend=False
            ))
    
    change_rates_df = predict_variable_change_rates(mod_country_df, ai_tax, optimal_tax_percent)
    st.subheader("AI 세율 적용에 따른 변수 예측 변화율")
    
    def color_change(val):
        try:
            val = float(val)
            if val > 0:
                return "color: blue"
            elif val < 0:
                return "color: red"
            else:
                return ""
        except:
            return ""
    
    styled_df = change_rates_df.style.format("{:.2f}", 
        subset=["실제 값", "예측값", "예측 변화율", "최적 AI 세율일 경우 예측값", "최적 AI 세율일 경우 예측 변화율", "최적 AI 세율일 때와의 변화량 차이"]
    ).applymap(color_change, subset=["예측 변화율", "최적 AI 세율일 경우 예측 변화율", "최적 AI 세율일 때와의 변화량 차이"])
    
    st.dataframe(styled_df)
    
st.subheader("GDP 추세 시각화")
st.plotly_chart(fig, use_container_width=True)