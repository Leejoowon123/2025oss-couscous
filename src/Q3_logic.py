# src/Q3_logic.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.vector_ar.svar_model import SVAR
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize

########################################
# 1. 패널 데이터 준비 및 전처리
########################################

def prepare_panel_data(df, variables):
    """
    입력 데이터(df)에서 Country, Year 및 분석 대상 변수들을 사용하여
    전체 국가 평균 패널과 각 국가별 패널 데이터를 생성합니다.
    """
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    for var in variables:
        df[var] = pd.to_numeric(df[var], errors='coerce')
    df = df.dropna(subset=["Country", "Year"] + variables)
    df = df.sort_values(by=["Country", "Year"])
    overall_panel = df.groupby("Year")[variables].mean().reset_index().set_index("Year")
    country_panels = {country: data.set_index("Year")[variables]
                    for country, data in df.groupby("Country")}
    return overall_panel, country_panels

########################################
# 2. VAR 모형 및 예측 (단일 및 다변량) – 각 국가별 적용
########################################

def run_var_model(panel_df, target_vars, maxlags=2, forecast_steps=5):
    """
    VAR 모형을 사용하여 미래 예측을 수행합니다.
    target_vars가 한 개이면 ARIMA 모형을 사용하고, 그렇지 않으면 VAR 모형을 사용합니다.
    """
    model_data = panel_df[target_vars]
    if len(target_vars) == 1:
        var_name = target_vars[0]
        model = sm.tsa.ARIMA(model_data[var_name], order=(maxlags,0,0))
        results = model.fit()
        forecast = results.forecast(steps=forecast_steps)
        forecast_df = pd.DataFrame({var_name: forecast})
        forecast_years = list(range(int(model_data.index.max())+1, int(model_data.index.max())+forecast_steps+1))
        forecast_df["Year"] = forecast_years
        forecast_df = forecast_df.set_index("Year")
        return results, forecast_df
    else:
        model = VAR(model_data)
        results = model.fit(maxlags=maxlags)
        last_values = model_data.values[-results.k_ar:]
        forecast = results.forecast(last_values, steps=forecast_steps)
        forecast_years = list(range(int(model_data.index.max())+1, int(model_data.index.max())+forecast_steps+1))
        forecast_df = pd.DataFrame(forecast, columns=model_data.columns)
        forecast_df["Year"] = forecast_years
        forecast_df = forecast_df.set_index("Year")
        return results, forecast_df

def get_country_forecasts_VAR(country_panels, target_var, maxlags=2, forecast_steps=5):
    """
    각 국가별로 단일 VAR(또는 ARIMA) 모형을 적용하여 target_var 예측 결과를 반환합니다.
    """
    forecasts = {}
    for country, panel in country_panels.items():
        try:
            _, forecast_df = run_var_model(panel, [target_var], maxlags, forecast_steps)
            forecasts[country] = forecast_df
        except Exception as e:
            st.warning(f"{country} VAR 예측 실패 ({target_var}): {e}")
    return forecasts

def get_country_forecasts_multivariate_VAR(country_panels, target_vars, maxlags=2, forecast_steps=5):
    """
    각 국가별로 다변량 VAR 모형을 적용하여 target_vars 예측 결과를 반환합니다.
    """
    forecasts = {}
    for country, panel in country_panels.items():
        try:
            _, forecast_df = run_var_model(panel, target_vars, maxlags, forecast_steps)
            forecasts[country] = forecast_df
        except Exception as e:
            st.warning(f"{country} 다변량 VAR 예측 실패: {e}")
    return forecasts

########################################
# 3. VECM 및 반사실 분석 – 각 국가별 개별 적용
########################################

def run_vecm_model_country(panel_df, target_vars, lag_order=1, forecast_steps=5, coint_rank=1):
    """
    각 국가별로 VECM 모형을 적용하여 예측합니다.
    """
    model_data = panel_df[target_vars]
    try:
        vecm_model = VECM(model_data, k_ar_diff=lag_order, coint_rank=coint_rank)
        vecm_res = vecm_model.fit()
        forecast = vecm_res.predict(steps=forecast_steps)
        forecast_years = list(range(int(model_data.index.max())+1, int(model_data.index.max())+forecast_steps+1))
        forecast_df = pd.DataFrame(forecast, columns=target_vars)
        forecast_df["Year"] = forecast_years
        forecast_df = forecast_df.set_index("Year")
        return vecm_res, forecast_df
    except Exception as e:
        st.warning(f"VECM 국가별 예측 실패: {e}")
        return None, None

def run_counterfactual_analysis_country(country_panel, tax_var="WIPO Tax", gdp_var="GDP", forecast_steps=5, maxlags=2):
    """
    각 국가별로 단일 VAR 모형을 이용해 GDP 예측을 도출한 후,
    해당 국가의 최근 5년 데이터를 이용한 log-log 회귀로 AI 세율 효과(탄력성)를 추정하고,
    정책 변화가 GDP에 미치는 효과를 반영하여 counterfactual GDP를 산출합니다.
    """
    try:
        _, forecast_df = run_var_model(country_panel, target_vars=[gdp_var], maxlags=maxlags, forecast_steps=forecast_steps)
        panel_sorted = country_panel.sort_index().copy()
        if len(panel_sorted) < 5:
            return None
        pre_data = panel_sorted.iloc[-5:].copy()
        pre_data["log_GDP"] = np.log(pre_data[gdp_var])
        pre_data["log_Tax"] = np.log(pre_data[tax_var])
        X = sm.add_constant(pre_data[["log_Tax"]])
        y = pre_data["log_GDP"]
        reg_model = sm.OLS(y, X).fit()
        elasticity = reg_model.params["log_Tax"]
        baseline_tax = pre_data[tax_var].mean()
        policy_tax = pre_data[tax_var].quantile(0.9)
        delta_log = elasticity * (np.log(policy_tax) - np.log(baseline_tax))
        cf_df = forecast_df.copy()
        cf_df["log_GDP_forecast"] = np.log(cf_df[gdp_var])
        cf_df["log_GDP_cf"] = cf_df["log_GDP_forecast"] + delta_log
        cf_df[gdp_var] = np.exp(cf_df["log_GDP_cf"])
        return {"actual": forecast_df, "counterfactual": cf_df, "elasticity": elasticity,
                "baseline_tax": baseline_tax, "policy_tax": policy_tax, "delta_log": delta_log}
    except Exception as e:
        st.warning(f"반사실 국가별 분석 실패: {e}")
        return None

########################################
# 4. 거시경제 시뮬레이션 통합 실행 함수 – 각 국가별 모형 적용
########################################

def run_macro_simulation(df):
    """
    VAR, VECM, SVAR, 반사실 분석, 합성 통제법을 적용하여
    전체 국가 평균 및 각 국가별 예측 결과와 분석 지표를 산출합니다.
    사용 변수: GDP, Unemployment Rate, GERD, WIPO Tax
    """
    variables_extended = ['GDP', 'Unemployment Rate', 'GERD', 'WIPO Tax']
    overall_panel, country_panels = prepare_panel_data(df, variables_extended)
    
    # 단일 VAR 모형: 각 변수별 예측 (각 국가별)
    var_forecasts = {}
    for var in ['GDP', 'Unemployment Rate', 'GERD']:
        var_forecasts[var] = get_country_forecasts_VAR(country_panels, var, maxlags=2, forecast_steps=5)
    
    # 다변량 VAR 모형: 각 국가별 예측
    multivar_forecasts = {}
    for country, panel in country_panels.items():
        try:
            _, forecast_df = run_var_model(panel, ['GDP', 'Unemployment Rate', 'GERD'], maxlags=2, forecast_steps=5)
            multivar_forecasts[country] = forecast_df
        except Exception as e:
            st.warning(f"{country} 다변량 VAR 예측 실패: {e}")
    
    # 각 국가별 VECM 모형 예측
    vecm_forecasts = {}
    for country, panel in country_panels.items():
        try:
            _, forecast_df = run_vecm_model_country(panel, ['GDP', 'Unemployment Rate', 'GERD'],
                                                    lag_order=1, forecast_steps=5, coint_rank=1)
            vecm_forecasts[country] = forecast_df
        except Exception as e:
            st.warning(f"{country} VECM 예측 실패: {e}")

    # 각 국가별 반사실 분석
    counterfactual = {}
    for country, panel in country_panels.items():
        cf = run_counterfactual_analysis_country(panel, tax_var="WIPO Tax", gdp_var="GDP",
                                                forecast_steps=5, maxlags=2)
        if cf is not None:
            counterfactual[country] = cf
    
    return {
        "var_forecasts": var_forecasts,
        "multivar_forecasts": multivar_forecasts,
        "vecm_forecasts": vecm_forecasts,
        "counterfactual": counterfactual,
        "overall_panel": overall_panel
    }
