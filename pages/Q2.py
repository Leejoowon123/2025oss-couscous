# pages/Q2.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import json
import ast
from common.data_utils import load_data
from src.Q2_logic import (
    analyze_single_variables,
    filter_variables_by_policy_and_r2,
    build_combinations_of_variables,
    analyze_all
)

def app_Q2():
    st.title("Q2: AI 세율의 최적 수준은 어떻게 설정할 것인가?")
    st.markdown("""
    ## 결론 및 연구 요약
    - 본 연구에서는 단일 변수 및 복합 변수 접근을 통해, 각 변수(또는 변수 조합)와 GDP 간의 비선형 관계(Laffer-Curve)를 피팅하여  
    세율(0.0 ~ 5.0%) 범위 내에서 GDP를 최대화하는 최적 세율을 도출했습니다.  
    - 결정계수(R²)가 높을수록 해당 변수(또는 변수 조합)가 GDP를 잘 설명하므로, R² 기준 (예: 0.8 이상)으로 신뢰할 만한 세율 대리 변수를 선별할 수 있습니다.  
    
    ## 연구 접근 방법
    1. **단일 변수 접근**:  
        - Country, year, GDP를 제외한 모든 변수에 대해 Laffer-Curve을 피팅하고,  
        - Differential Evolution, Dual Annealing, SHGO 등의 최적화 알고리즘으로 최적 세율 도출 (세율 범위: 0.0 ~ 5.0)
    2. **정책적 타당성과 결정계수(R²)**를 기준으로 세율 대리 변수를 선별하여,  
        - 실제 조세 정책에 활용할 후보를 도출합니다.
    3. **복합 변수 접근**:  
        - 2~3개 변수 조합을 통해 복합 변수를 구성한 후,  
        - Laffer-Curve 피팅 및 최적 세율 도출 결과 중 결정계수(R²) 0.8 이상인 결과만 선택하여 비교합니다.
    """)
    
    df = load_data()
    
    st.markdown("### 1) 단일 변수 접근 결과")
    single_filename = os.path.join("data", "single_results.csv")
    if os.path.exists(single_filename):
        try:
            single_df = pd.read_csv(single_filename)
            st.dataframe(single_df)
        except Exception as e:
            st.warning(f"단일 변수 결과 CSV 로드 실패: {e}")
            single_results = analyze_single_variables(df)
            pd.DataFrame(single_results).to_csv(single_filename, index=False, quoting=1)
            st.dataframe(pd.DataFrame(single_results))
    else:
        single_results = analyze_single_variables(df)
        os.makedirs("data", exist_ok=True)
        pd.DataFrame(single_results).to_csv(single_filename, index=False, quoting=1)
        st.dataframe(pd.DataFrame(single_results))
    
    st.markdown("### 2) 정책적 타당성과 결정계수(R²) 기준 변수 선별")
    candidate_vars = [c for c in df.columns if c not in ["Country", "year", "Year", "GDP"]]
    selected = filter_variables_by_policy_and_r2(df, candidate_vars, r2_threshold=0.3)
    if not selected:
        st.warning("정책적으로 유의미한 변수가 없습니다.")
    else:
        sel_df = pd.DataFrame([{"Variable": var, "R2": r2, "Params": params} for (var, r2, params) in selected])
        st.dataframe(sel_df)
    
    st.markdown("### 3) 복합 변수 접근 결과 (결정계수 R² >= 0.8)")
    combo_filename = os.path.join("data", "combo_results.csv")
    if os.path.exists(combo_filename):
        try:
            combo_df = pd.read_csv(combo_filename)
            st.dataframe(combo_df)
        except Exception as e:
            st.warning(f"복합 변수 결과 CSV 로드 실패: {e}")
            combo_results = build_combinations_of_variables(df, candidate_vars, max_comb=3, r2_filter=0.8)
            pd.DataFrame(combo_results).to_csv(combo_filename, index=False, quoting=1)
            st.dataframe(pd.DataFrame(combo_results))
    else:
        combo_results = build_combinations_of_variables(df, candidate_vars, max_comb=3, r2_filter=0.8)
        os.makedirs("data", exist_ok=True)
        pd.DataFrame(combo_results).to_csv(combo_filename, index=False, quoting=1)
        st.dataframe(pd.DataFrame(combo_results))
    
    st.markdown("""
    ## 결론 및 정책적 시사점
    - 단일 변수 접근 및 복합 변수 접근을 통해 도출된 최적 AI 세율은, 경제 지표(GDP)를 최대화하는 세율로서, 정책적으로 세율 대리 변수로 사용할 수 있는 후보를 선정하는 기초 자료를 제공합니다.
    - 특히, 결정계수(R²)는 해당 모형이 GDP와의 관계를 얼마나 잘 설명하는지를 나타내므로, R²가 0.8 이상인 경우 신뢰도가 높아 정책 적용에 적합한 것으로 판단됩니다.
    - 정부는 이를 바탕으로 AI 세율을 설정할 때, 단일 변수 또는 여러 경제 지표를 결합한 복합 변수를 고려하여, 경제 성장 및 기업 투자 촉진을 위한 최적 세율 범위를 도출할 수 있습니다.
    """)
    
def run_Q2():
    app_Q2()

if __name__ == "__main__":
    run_Q2()
