import streamlit as st
import pandas as pd
from common.data_utils import load_data, split_by_country
from src.System_logic import get_optimal_ai_tax, macroeconomic_simulation

def run_System():
    st.title("통합 AI 세율 정책 시뮬레이션 시스템")
    df = load_data()
    country_dfs = split_by_country(df)
    selected_country = st.selectbox("분석할 국가 선택", list(country_dfs.keys()))
    country_df = country_dfs[selected_country]
    
    # 최적 AI 세율 도출 (System_logic.py 내 함수 활용)
    optimal_tax, params = get_optimal_ai_tax(country_df)
    if optimal_tax is None:
        st.error("최적 AI 세율 도출에 실패했습니다.")
        return
    st.subheader(f"{selected_country} 최적 AI 세율: {optimal_tax*100:.2f}%")
    
    # 사용자 입력을 통한 AI 세율 조정
    user_tax_percent = st.slider("사용자 AI 세율 입력 (%)", min_value=0.0, max_value=30.0, value=optimal_tax*100, step=0.1)
    user_tax = user_tax_percent / 100
    
    # 사용자 세율과 최적 세율에 따른 거시경제 시뮬레이션 결과
    forecast_user = macroeconomic_simulation(selected_country, country_df, user_tax, optimal_tax, params)
    forecast_optimal = macroeconomic_simulation(selected_country, country_df, optimal_tax, optimal_tax, params)
    
    st.markdown("### 예측 결과 비교")
    st.write("사용자 입력 세율에 따른 예측:")
    st.dataframe(forecast_user)
    st.write("최적 세율에 따른 예측:")
    st.dataframe(forecast_optimal)
    
    st.markdown("**추가: 예측 결과의 신뢰구간 및 민감도 분석 코드 추가 예정**.")

if __name__ == "__main__":
    run_System()
