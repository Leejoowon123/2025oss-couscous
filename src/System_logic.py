import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit, differential_evolution
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import VAR
import streamlit as st

# 랜덤 시드 고정
np.random.seed(42)

def load_data():
    conn = st.connection("ossdb", type="sql", autocommit=True)
    sql = """
        SELECT *
        FROM master_data_by_category_clear
        WHERE 1=1;
    """
    df = conn.query(sql, ttl=3600)
    country_dfs = {
        "China": df[df["Country"] == "China"].copy(),
        "France": df[df["Country"] == "France"].copy(),
        "USA": df[df["Country"] == "United States of America"].copy(),
        "Germany": df[df["Country"] == "Germany"].copy(),
        "Japan": df[df["Country"] == "Japan"].copy(),
        "Korea": df[df["Country"] == "Korea"].copy(),
        "UK": df[df["Country"] == "United Kingdom"].copy(),
    }
    return df, country_dfs

def laffer_curve(x, a, b, c):
    return a * x**2 + b * x + c

def get_high_corr_vars(country_df, threshold=0.45):
    country_numeric_df = country_df.select_dtypes(include=[np.number]).drop(["GDP", "Year"], axis=1, errors="ignore")
    correlations = country_numeric_df.corrwith(country_df["GDP"]).dropna().abs()
    return correlations[correlations > threshold].index.tolist()

def find_best_proxy(country_df, high_corr_vars):
    if not high_corr_vars:
        return []
    proxy_candidates = []
    scaler = MinMaxScaler(feature_range=(0, 0.5))
    for r in range(2, 4):
        for combo in combinations(high_corr_vars, r):
            valid_combo = [col for col in combo if col in country_df.columns]
            if len(valid_combo) == r:
                proxy_name = f"AI_Tax_Proxy_{'_'.join(valid_combo)}"
                df_scaled = scaler.fit_transform(country_df[valid_combo])
                country_df[proxy_name] = df_scaled.mean(axis=1)
                proxy_candidates.append(proxy_name)
    return proxy_candidates

def find_best_model(country_df, proxy_candidates):
    if not proxy_candidates:
        return None, None
    best_degree = 1
    best_bic = np.inf
    best_proxy = None
    for proxy in proxy_candidates:
        X = country_df[[proxy]].values
        y = country_df["GDP"].values
        for d in range(2, 7):
            poly = PolynomialFeatures(degree=d)
            X_poly = poly.fit_transform(X)
            model = LinearRegression()
            model.fit(X_poly, y)
            y_pred = model.predict(X_poly)
            bic = len(y) * np.log(mean_squared_error(y, y_pred)) + d * np.log(len(y))
            if bic < best_bic:
                best_bic = bic
                best_degree = d
                best_proxy = proxy
    return best_degree, best_proxy

def get_optimal_ai_tax(country_df):
    """
    필요한 컬럼만 남긴 복사본(df_mod)에 Proxy 컬럼을 추가한 후,
    최적 AI 세율(예: 0.15 → 15%)과 함께 사용된 Proxy 변수(best_proxy)와 수정된 데이터프레임(df_mod)을 반환합니다.
    """
    high_corr_vars = get_high_corr_vars(country_df)
    df_mod = country_df[["Country", "Year", "GDP"] + high_corr_vars].copy()
    proxy_candidates = find_best_proxy(df_mod, high_corr_vars)
    best_degree, best_proxy = find_best_model(df_mod, proxy_candidates)
    if not best_proxy:
        return None, None, None, country_df
    X_proxy = df_mod[[best_proxy]].values.flatten()
    y_gdp = df_mod["GDP"].values
    if best_degree > 1:
        try:
            params, _ = curve_fit(laffer_curve, X_proxy, y_gdp, p0=[-0.5, 0.1, y_gdp.mean()])
        except RuntimeError:
            return None, None, None, country_df
    else:
        params = [0, 0, y_gdp.mean()]
    def optimize_tax(tax_rate):
        return -laffer_curve(tax_rate, *params)
    bounds = [(0, 0.4)]
    opt_result = differential_evolution(optimize_tax, bounds, seed=42)
    optimal_tax = opt_result.x[0]
    return optimal_tax, params, best_proxy, df_mod

# --- Macroeconomic Simulation (VAR 기반) ---
def macroeconomic_simulation(country, country_df, applied_tax):
    """
    VAR 모델을 사용하여 2013~2022년 데이터에서
    ["GDP", "GERD", "Patent Publications"] 중 실제 존재하는 컬럼(available_cols)으로 모델을 적합하고,
    5년 후 예측 결과를 DataFrame으로 반환합니다.
    예측값은 applied_tax에 따라 tax_effect_factor를 반영해 조정됩니다.
    """
    required_cols = ["GDP", "GERD", "Patent Publications"]
    available_cols = [col for col in required_cols if col in country_df.columns]
    if len(available_cols) < 2:
        return None
    model_data = country_df[available_cols].dropna()
    if model_data.shape[0] < 5:
        return None
    model = VAR(model_data)
    results = model.fit(maxlags=2)
    forecast = results.forecast(model_data.values[-2:], steps=5)
    forecast_df = pd.DataFrame(forecast, columns=available_cols)
    tax_effect_factor = 0.001
    adjustment = 1 - applied_tax * tax_effect_factor
    forecast_df_adjusted = forecast_df * adjustment
    return forecast_df_adjusted

def predict_variable_change_rates(country_df, applied_tax, optimal_tax_percent):
    """
    각 변수에 대해 예측값 및 예측 변화율을 산출합니다.
    - ["GDP", "GERD", "Patent Publications"]는 VAR forecast 결과(세율 효과 반영)를 사용하여 예측합니다.
    - 나머지 변수는 ["GDP", "GERD", "Patent Publications"]와의 다변량 관계를 분석하기 위해
    2013~2022년 데이터를 바탕으로 다중 선형 회귀 모델을 학습한 후, 
    예측 시에는 VAR forecast 결과(주요 변수 예측치)에 AI 세율 효과를 반영한 조정치를 곱해 예측합니다.
    최종 결과 DataFrame은 아래의 컬럼들을 포함하며, 모든 수치는 소수점 2째자리까지 반올림됩니다.
    "변수", "실제 값", "예측값", "예측 변화율", "최적 AI 세율일 경우 예측값", 
    "최적 AI 세율일 경우 예측 변화율", "최적 AI 세율일 때와의 변화량 차이"
    """
    results = []
    tax_effect_factor = 0.001
    available_main_vars = [col for col in ["GDP", "GERD", "Patent Publications"] if col in country_df.columns]
    var_list_all = ["GDP", "GERD", "Patent Publications", "GDP_per_capita_PPP",
                    "GNI_per_capita", "General Revenue", "Global Innovation Index",
                    "Human capital and research", "Unemployment Rate"]
    country = country_df["Country"].iloc[0]
    forecast_df_VAR = macroeconomic_simulation(country, country_df, applied_tax)
    if forecast_df_VAR is not None:
        forecast_main = forecast_df_VAR.iloc[-1]
    else:
        forecast_main = None
    latest_row = country_df[country_df["Year"] == country_df["Year"].max()].iloc[0]
    adjustment_factor = 1 - 0.01 * (applied_tax - optimal_tax_percent)
    for var in var_list_all:
        if var not in country_df.columns:
            continue
        actual_value = latest_row[var]
        if var in available_main_vars:
            if forecast_main is not None:
                predicted_value = forecast_main[var]
            else:
                predicted_value = actual_value
            predicted_change_rate = ((predicted_value - actual_value) / actual_value * 100) if actual_value != 0 else 0.0
            predicted_value_opt = predicted_value
            predicted_change_rate_opt = predicted_change_rate
        else:
            data = country_df[country_df["Year"] >= 2013][available_main_vars + [var]].dropna()
            if data.shape[0] < 5:
                predicted_value = actual_value
                predicted_change_rate = 0.0
                predicted_value_opt = actual_value
                predicted_change_rate_opt = 0.0
            else:
                model = LinearRegression()
                X = data[available_main_vars].values
                y = data[var].values
                model.fit(X, y)
                if forecast_main is not None:
                    current_predictors = forecast_main[available_main_vars].values.reshape(1, -1) * adjustment_factor
                    optimal_predictors = forecast_main[available_main_vars].values.reshape(1, -1)
                    predicted_value = model.predict(current_predictors)[0]
                    predicted_value_opt = model.predict(optimal_predictors)[0]
                else:
                    predicted_value = actual_value
                    predicted_value_opt = actual_value
                predicted_change_rate = ((predicted_value - actual_value) / actual_value * 100) if actual_value != 0 else 0.0
                predicted_change_rate_opt = ((predicted_value_opt - actual_value) / actual_value * 100) if actual_value != 0 else 0.0
        diff_change = round(predicted_change_rate - predicted_change_rate_opt, 2)
        results.append({
            "변수": var,
            "실제 값": round(actual_value, 2),
            "예측값": round(predicted_value, 2),
            "예측 변화율": round(predicted_change_rate, 2),
            "최적 AI 세율일 경우 예측값": round(predicted_value_opt, 2),
            "최적 AI 세율일 경우 예측 변화율": round(predicted_change_rate_opt, 2),
            "최적 AI 세율일 때와의 변화량 차이": diff_change
        })
    return pd.DataFrame(results)