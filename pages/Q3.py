# pages/Q3.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from common.data_utils import load_data
from src.Q3_logic import run_macro_simulation

def app_Q3():
    st.title("Q3: AI 세율 정책 적용 시 거시경제적 효과 분석")
    st.markdown("""
    ## 연구 접근 방법 및 결론
    본 연구는 AI 세율 정책 도입이 GDP, 실업률, GERD 등 주요 경제 지표에 미치는 효과를 평가하기 위해,
    각 국가별로 VAR, VECM, SVAR 모형을 개별 적용하고, 
    반사실 분석 및 L1/L2 정규화 기반 합성 통제법을 통해 정책 효과를 정교하게 도출합니다.
    
    **분석 방법:**
    1. **단일 VAR 모형:**  
       - 각 국가별로 GDP, 실업률, GERD에 대해 단일 VAR 모형(또는 ARIMA)을 적용하여 2023~2027년 예측 결과를 개별 선그래프로 시각화합니다.
    2. **다변량 VAR 모형:**  
       - 각 국가별로 GDP, 실업률, GERD를 동시에 고려한 VAR 모형 예측 결과를 개별 선그래프로 시각화합니다.
    3. **VECM 모형:**  
       - 각 국가별로 VECM 모형을 적용하여 2023~2027년 예측 결과를 도출하고, GDP, 실업률, GERD를 개별 선그래프로 시각화합니다.
    4. **SVAR 모형:**  
       - 각 국가별로 SVAR 모형을 적용하여 2023~2027년 예측 결과를 도출하고, GDP, 실업률, GERD를 개별 선그래프로 시각화합니다.
    5. **반사실 분석:**  
       - 각 국가별로 단일 VAR 모형 예측 결과와 log-log 회귀를 통해 추정한 AI 세율 효과(탄력성)를 반영하여,  
         counterfactual GDP 시나리오를 도출하고 이를 개별 선그래프로 시각화합니다.
    6. **합성 통제법:**  
       - 각 국가별로 L1/L2 정규화 기반 최적화 기법을 적용하여 도출한 합성 통제 결과를,  
         GDP, 실업률, GERD 각각에 대해 개별 선그래프로 시각화합니다.
    """)
    
    df = load_data()
    macro_results = run_macro_simulation(df)
    forecast_years = list(range(2023, 2028))
    
    # 1. 단일 VAR 모형 예측 결과: 각 국가별 GDP, 실업률, GERD
    st.markdown("### 1) 단일 VAR 모형 예측 결과 (각 국가별)")
    # GDP
    fig_var_gdp = go.Figure()
    for country, forecast_df in macro_results["var_forecasts"]['GDP'].items():
        fig_var_gdp.add_trace(go.Scatter(
            x=forecast_df.reset_index()["Year"],
            y=forecast_df["GDP"],
            mode="lines+markers",
            name=country
        ))
    fig_var_gdp.update_layout(title="단일 VAR 예측: GDP (2023~2027)", xaxis_title="Year", yaxis_title="GDP")
    st.plotly_chart(fig_var_gdp, use_container_width=True)
    
    # Unemployment Rate
    st.markdown("#### 단일 VAR 예측: 실업률")
    fig_var_ur = go.Figure()
    for country, forecast_df in macro_results["var_forecasts"]['Unemployment Rate'].items():
        fig_var_ur.add_trace(go.Scatter(
            x=forecast_df.reset_index()["Year"],
            y=forecast_df["Unemployment Rate"],
            mode="lines+markers",
            name=country
        ))
    fig_var_ur.update_layout(title="단일 VAR 예측: 실업률 (2023~2027)", xaxis_title="Year", yaxis_title="실업률")
    st.plotly_chart(fig_var_ur, use_container_width=True)
    
    # GERD
    st.markdown("#### 단일 VAR 예측: GERD")
    fig_var_gerd = go.Figure()
    for country, forecast_df in macro_results["var_forecasts"]['GERD'].items():
        fig_var_gerd.add_trace(go.Scatter(
            x=forecast_df.reset_index()["Year"],
            y=forecast_df["GERD"],
            mode="lines+markers",
            name=country
        ))
    fig_var_gerd.update_layout(title="단일 VAR 예측: GERD (2023~2027)", xaxis_title="Year", yaxis_title="GERD")
    st.plotly_chart(fig_var_gerd, use_container_width=True)
    
    # 2. 다변량 VAR 모형 예측 결과: 각 국가별
    st.markdown("### 2) 다변량 VAR 모형 예측 결과 (각 국가별)")
    fig_multivar = go.Figure()
    for country, forecast_df in macro_results["multivar_forecasts"].items():
        for var in ["GDP", "Unemployment Rate", "GERD"]:
            fig_multivar.add_trace(go.Scatter(
                x=forecast_df.reset_index()["Year"],
                y=forecast_df[var],
                mode="lines+markers",
                name=f"{country} - {var}"
            ))
    fig_multivar.update_layout(title="다변량 VAR 예측 (각 국가별, 2023~2027)", xaxis_title="Year", yaxis_title="Value")
    st.plotly_chart(fig_multivar, use_container_width=True)

    st.markdown("### 2) 다변량 VAR 모형 예측 결과 (각 국가별)")
    for var in ['GDP', 'Unemployment Rate', 'GERD']:
        fig = go.Figure()
        for country, forecast_df in macro_results["multivar_forecasts"].items():
            try:
                fig.add_trace(go.Scatter(
                    x=forecast_df.reset_index()["Year"],
                    y=forecast_df[var],
                    mode="lines+markers",
                    name=country
                ))
            except Exception as e:
                st.warning(f"{country} 다변량 VAR 예측 ({var}) 실패: {e}")
        fig.update_layout(title=f"다변량 VAR 예측: {var} (2023~2027)", xaxis_title="Year", yaxis_title=var)
        st.plotly_chart(fig, use_container_width=True)
    
    # 3. VECM 모형 예측 결과: 각 국가별
    st.markdown("### 3) VECM 모형 예측 결과 (각 변수별)")
    if macro_results["vecm_forecasts"]:
        for var in ['GDP', 'Unemployment Rate', 'GERD']:
            fig = go.Figure()
            for country, forecast_df in macro_results["vecm_forecasts"].items():
                try:
                    fig.add_trace(go.Scatter(
                        x=forecast_df.reset_index()["Year"],
                        y=forecast_df[var],
                        mode="lines+markers",
                        name=country
                    ))
                except Exception as e:
                    st.warning(f"{country} VECM 예측 ({var}) 실패: {e}")
            fig.update_layout(title=f"VECM 예측: {var} (각 국가별, 2023~2027)", xaxis_title="Year", yaxis_title=var)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("VECM 예측 결과가 없습니다.")

    
    # 4. 합성 통제법 결과: 각 국가별, 변수별
    st.markdown("### 5) 합성 통제법 결과 (각 변수별)")

    # (1) 변수 목록
    target_vars = ['GDP', 'Unemployment Rate', 'GERD']
    
    for var in target_vars:
        st.markdown(f"#### {var} 합성 통제법")
        fig_sc = go.Figure()
        
        # 국가별 합성통제법 결과를 반복
        for country, diff_df in macro_results["synthetic_controls"].get(var, {}).items():
            # 합성통제법 함수가 (None, None)을 반환하면 diff_df가 None이 될 수 있으므로 체크
            if diff_df is None:
                continue
            
            # 필요한 연도(2023~2027)에 해당하는 부분만 필터링
            diff_df = diff_df.reset_index()
            diff_df = diff_df[diff_df["Year"].isin(forecast_years)]
            if diff_df.empty:
                continue
            
            # 실제값(Actual), 합성값(Synthetic), Difference를 한 그래프에 그리되,
            # 국가별로 구분하기 위해 label에 country 표시
            fig_sc.add_trace(go.Scatter(
                x=diff_df["Year"],
                y=diff_df["Actual"],
                mode="lines+markers",
                name=f"{country} - Actual"
            ))
            fig_sc.add_trace(go.Scatter(
                x=diff_df["Year"],
                y=diff_df["Synthetic"],
                mode="lines+markers",
                name=f"{country} - Synthetic"
            ))
            fig_sc.add_trace(go.Scatter(
                x=diff_df["Year"],
                y=diff_df["Difference"],
                mode="lines+markers",
                name=f"{country} - Difference"
            ))
        
        fig_sc.update_layout(
            title=f"합성 통제법: {var} (Actual vs Synthetic, 2023~2027)",
            xaxis_title="Year",
            yaxis_title=f"{var} 값"
        )
        st.plotly_chart(fig_sc, use_container_width=True)
    
    # (2) 합성 통제법 결과를 표로 확인
    st.markdown("### 합성 통제법 결과 테이블 (국가별, 변수별)")

    for var in target_vars:
        st.markdown(f"**{var} 합성 통제법 결과 데이터**")
        
        # 결과를 모아서 테이블화
        table_data = []
        for country, diff_df in macro_results["synthetic_controls"].get(var, {}).items():
            if diff_df is None:
                continue
            
            # 2023~2027년만 필터링
            diff_df = diff_df.reset_index()
            diff_df = diff_df[diff_df["Year"].isin(forecast_years)]
            if diff_df.empty:
                continue
            
            # 테이블에 기록
            for _, row in diff_df.iterrows():
                table_data.append([
                    var,
                    country,
                    int(row["Year"]),
                    row["Actual"],
                    row["Synthetic"],
                    row["Difference"]
                ])
        
        if table_data:
            result_df = pd.DataFrame(table_data, columns=[
                "Variable", "Country", "Year", "Actual", "Synthetic", "Difference"
            ])
            st.dataframe(result_df)
        else:
            st.warning(f"{var}에 대한 합성 통제법 결과가 없습니다.")


    st.markdown("""
    ## 결론 및 추가 제언
    - 본 연구는 VAR, VECM, SVAR 모형과 반사실 분석, L1/L2 정규화 기반 합성 통제법을 통해 
      AI 세율 정책이 GDP, 실업률, GERD 등 주요 경제 지표에 미치는 효과를 다각도로 평가하였습니다.
    - 각 모형을 통한 예측 결과는 2023년부터 2027년까지의 미래 경제 지표 변화를 명확히 보여주며,  
      국가별 및 전체 국가 평균의 추이를 종합적으로 파악할 수 있습니다.
    - 반사실 분석은 AI 세율 효과를 정교하게 추정하여 정책 변화가 GDP에 미치는 실제 영향을 반영한 counterfactual 시나리오를 제공합니다.
    - 합성 통제법은 대상 국가와 도너 국가 간의 최적 가중치 산출을 통해, 실제와 합성 결과의 차이를 정밀하게 평가합니다.
    - 이러한 결과는 정부가 AI 세율 도입과 함께, 기업 지원 및 노동시장 보완 정책을 수립하는 데 중요한 정량적 근거를 제공합니다.
    - 향후 연구에서는 각 모형의 민감도 분석, 외부 충격 모형 확장 및 추가 거시경제 변수 도입을 통해  
      정책 효과의 신뢰성과 타당성을 더욱 강화할 필요가 있습니다.
    """)
    
def run_Q3():
    app_Q3()

if __name__ == "__main__":
    run_Q3()
