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
# 3. VECM 및 SVAR 모형 – 각 국가별 개별 적용
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
        st.warning(f"VECM 모형 실패: {e}")
        return None, None



########################################
# 4. 반사실 분석 – 국가별로 정교하게 실시 (새로운 로직 추가)
########################################

# Q3_logic.py 내부 예시

def run_synthetic_control(
    df, target_country, variable, pre_period, post_period,
    lambda1=0.05, lambda2=0.05
):
    """
    대상 국가의 pre-intervention 데이터와 다른 국가들의 동일 기간 데이터를 사용하여,
    L1 및 L2 정규화 기반 최적화 기법으로 도너 가중치를 산출하고 합성 통제군을 도출합니다.
    """
    pre_start, pre_end = pre_period
    post_start, post_end = post_period

    # Pre-intervention 데이터 준비
    pre_data = df[(df["Year"] >= pre_start) & (df["Year"] <= pre_end)]
    target_pre = pre_data[pre_data["Country"] == target_country].set_index("Year")[variable]

    # 도너 국가 데이터 준비
    donors = pre_data[pre_data["Country"] != target_country]
    donor_matrix = donors.pivot(index="Year", columns="Country", values=variable)

    # 1️⃣ 도너 국가 결측치 처리
    if donor_matrix.isna().all().all():
        st.warning(f"합성 통제법 경고: {target_country}의 도너 국가 데이터가 거의 없습니다.")
        # (변경됨) 기존: return None, None --> 주석 처리
        # return None, None

        # 예시) 결측일 경우 0으로 대체하여 강제로 진행
        donor_matrix = donor_matrix.fillna(0)

    # 2️⃣ 연도 일치
    common_years = target_pre.index.intersection(donor_matrix.index)
    target_pre = target_pre.loc[common_years]
    donor_matrix = donor_matrix.loc[common_years]

    # 결측치 보완
    donor_matrix = donor_matrix.fillna(donor_matrix.mean(axis=0)).interpolate().fillna(0)

    def objective(weights):
        weights = np.array(weights)
        synthetic = donor_matrix.dot(weights)
        error = np.sum((target_pre - synthetic) ** 2)
        reg = lambda1 * np.sum(np.abs(weights)) + lambda2 * np.sum(weights**2)
        return error + reg

    num_donors = donor_matrix.shape[1]
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1)] * num_donors
    initial_weights = np.ones(num_donors) / num_donors

    # 3️⃣ 최적화 알고리즘
    opt = minimize(objective, initial_weights, method='trust-constr', bounds=bounds, constraints=cons)

    # (변경됨) 기존: if np.isclose(opt.x.sum(), 0): return None, None --> 주석 처리
    if np.isclose(opt.x.sum(), 0):
        st.warning(f"합성 통제법 경고: {target_country} 가중치 합이 0에 수렴합니다. (계속 진행)")
        # return None, None

    optimal_weights = opt.x
    synthetic_pre = donor_matrix.dot(optimal_weights)

    # 4️⃣ Post-intervention 데이터 처리
    post_data = df[(df["Year"] >= post_start) & (df["Year"] <= post_end)]
    donor_post = post_data[post_data["Country"] != target_country]
    target_post = post_data[post_data["Country"] == target_country].set_index("Year")[variable]
    donor_post_matrix = donor_post.pivot(index="Year", columns="Country", values=variable)

    # (변경됨) 결측치가 모두일 경우 처리
    if donor_post_matrix.isna().all().all():
        st.warning(f"합성 통제법 경고: {target_country}의 post 데이터가 부족합니다. (0 대체 후 진행)")
        # return None, None  # 기존 코드 주석 처리
        donor_post_matrix = donor_post_matrix.fillna(0)

    donor_post_matrix = donor_post_matrix.interpolate(method='linear', limit_direction='both').fillna(0)
    synthetic_post = donor_post_matrix.dot(optimal_weights)

    # 5️⃣ 최종 diff_df 구성
    diff_df = pd.DataFrame({
        "Year": target_post.index,
        "Actual": target_post.values,
        "Synthetic": synthetic_post.values,
        "Difference": target_post.values - synthetic_post.values
    }).set_index("Year")

    weights_dict = dict(zip(donor_matrix.columns, optimal_weights))
    return diff_df, weights_dict


def get_country_synthetic_controls(df, target_variable, pre_period, post_period):
    """
    각 국가별로 합성 통제법을 적용하여 target_variable에 대한 합성 통제 결과를 반환합니다.
    """
    synthetic_dict = {}
    countries = df["Country"].unique()
    for country in countries:
        try:
            diff_df, _ = run_synthetic_control(df, country, target_variable, pre_period, post_period,
                                                lambda1=0.1, lambda2=0.1)
            synthetic_dict[country] = diff_df
        except Exception as e:
            st.warning(f"{country} 합성 통제법 ({target_variable}) 실패: {e}")
    return synthetic_dict

########################################
# 6. 거시경제 시뮬레이션 통합 실행 함수 – 각 국가별 모형 적용
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

    # 합성 통제법: 각 국가별 (GDP, Unemployment Rate, GERD)
    years = pd.to_numeric(df["Year"], errors='coerce')
    pre_period = (int(years.min()), int(years.median()))
    post_period = (int(years.median())+1, int(years.max()))
    synthetic_controls = {
        "GDP": get_country_synthetic_controls(df, "GDP", pre_period, post_period),
        "Unemployment Rate": get_country_synthetic_controls(df, "Unemployment Rate", pre_period, post_period),
        "GERD": get_country_synthetic_controls(df, "GERD", pre_period, post_period)
    }
    
    return {
        "var_forecasts": var_forecasts,
        "multivar_forecasts": multivar_forecasts,
        "vecm_forecasts": vecm_forecasts,
        "counterfactual": counterfactual,
        "synthetic_controls": synthetic_controls,
        "overall_panel": overall_panel
    }

# 추가: 각 국가별 VECM 및 SVAR 모형 함수
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