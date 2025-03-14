import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit, differential_evolution
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import VAR
import streamlit as st

# ---------------------------
# 랜덤 시드 고정
np.random.seed(42)

# ---------------------------
# Laffer Curve 함수
def laffer_curve(x, a, b, c):
    return a * x**2 + b * x + c

# ---------------------------
# DB 연결을 통한 데이터 로드 함수 (CSV 대신 DB 사용)
def load_data():
    conn = st.connection("ossdb", type="sql", autocommit=True)
    sql = "SELECT * FROM master_data_by_category_clear;"
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

# ---------------------------
# 고상관 변수 추
def get_high_corr_vars(country_df, threshold=0.45):
    country_numeric_df = country_df.select_dtypes(include=[np.number]).drop(["GDP", "Year"], axis=1, errors="ignore")
    correlations = country_numeric_df.corrwith(country_df["GDP"]).dropna().abs()
    high_corr_vars = correlations[correlations > threshold].index.tolist()

    essential_vars = ["GERD", "Patent Publications", "Year"]
    for var in essential_vars:
        if var not in high_corr_vars and var in country_df.columns:
            high_corr_vars.append(var)
    return high_corr_vars

# ---------------------------
# AI 세율 Proxy 후보 생성 로직
scaler = MinMaxScaler(feature_range=(0, 0.5))
def find_best_proxy(country_df, high_corr_vars):
    proxy_candidates = []
    for r in range(2, 4):
        for combo in combinations(high_corr_vars, r):
            valid_combo = [col for col in combo if col in country_df.columns]
            if len(valid_combo) == r:
                proxy_name = f"AI_Tax_Proxy_{'_'.join(valid_combo)}"
                df_scaled = scaler.fit_transform(country_df[valid_combo])
                country_df[proxy_name] = df_scaled.mean(axis=1)
                proxy_candidates.append(proxy_name)
    return proxy_candidates

# ---------------------------
# 최적 모델 선택 (BIC 기준)
def find_best_model(country_df, proxy_candidates):
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

# ---------------------------
# 최적 AI 세율 산출 (Laffer Curve 기반)
def get_optimal_ai_tax(country, country_df):
    high_corr_vars = get_high_corr_vars(country_df)
    country_df = country_df[["Country", "Year", "GDP"] + high_corr_vars]
    proxy_candidates = find_best_proxy(country_df, high_corr_vars)
    best_degree, best_proxy = find_best_model(country_df, proxy_candidates)

    if not best_proxy:
        st.error(f"{country}: 최적 Proxy 후보를 찾지 못했습니다.")
        return None, None, None

    X_proxy = country_df[best_proxy].values.flatten()
    y_gdp = country_df["GDP"].values

    if best_degree > 1:
        try:
            params, _ = curve_fit(laffer_curve, X_proxy, y_gdp, p0=[-0.5, 0.1, y_gdp.mean()])
        except RuntimeError:
            params = [0, 0, y_gdp.mean()]
    else:
        params = [0, 0, y_gdp.mean()]
    optimal_tax = differential_evolution(lambda x: -laffer_curve(x, *params), [(0, 0.4)], seed=42).x[0]
    return optimal_tax, params, best_proxy

# ---------------------------
# Counterfactual Analysis: 2022년 실제 GDP를 기준으로 변화량 및 변화율 계산
def counterfactual_analysis(country, country_df):
    optimal_tax, params, best_proxy = get_optimal_ai_tax(country, country_df)
    if optimal_tax is None:
        return None, None
    actual_gdp = country_df["GDP"].iloc[-1]  # 2022년 실제 GDP
    predicted_gdp = laffer_curve(optimal_tax, *params)
    gdp_change = (predicted_gdp - actual_gdp) / actual_gdp * 100
    gdp_diff = predicted_gdp - actual_gdp
    return gdp_change, gdp_diff

# ---------------------------
# Synthetic Control Method: 실제 GDP와 합성 GDP의 차이 계산 및 시각화
def synthetic_control(country, country_df):
    country_df["Synthetic GDP"] = country_df["GDP"].rolling(window=3, min_periods=1).mean()
    if not np.issubdtype(country_df["Year"].dtype, np.datetime64):
        country_df["Year"] = pd.to_datetime(country_df["Year"].astype(str), format="%Y")
    country_df["Difference"] = country_df["GDP"] - country_df["Synthetic GDP"]
    diff_df = country_df[["Year", "Difference"]].reset_index(drop=True)
    summary = {
        "Year": "Summary",
        "Difference": f"Avg: {diff_df['Difference'].mean():.2f}, Min: {diff_df['Difference'].min():.2f}, Max: {diff_df['Difference'].max():.2f}"
    }
    summary_df = pd.DataFrame([summary])
    diff_table = pd.concat([diff_df, summary_df], ignore_index=True)
    
    fig = px.line(country_df, x="Year", y=["GDP", "Synthetic GDP"],
                title=f"{country}: Actual vs. Synthetic GDP",
                markers=True)
    diff_series = country_df["Difference"]
    max_idx = diff_series.idxmax()
    min_idx = diff_series.idxmin()
    fig.add_trace(go.Scatter(x=[country_df.loc[max_idx, "Year"]],
                             y=[country_df.loc[max_idx, "GDP"]],
                             mode="markers+text",
                             text=[f"Max Diff: {diff_series[max_idx]:.2f}"],
                             textposition="top center",
                             marker=dict(color="green", size=12),
                             name="Maximum Difference"))
    fig.add_trace(go.Scatter(x=[country_df.loc[min_idx, "Year"]],
                             y=[country_df.loc[min_idx, "GDP"]],
                             mode="markers+text",
                             text=[f"Min Diff: {diff_series[min_idx]:.2f}"],
                             textposition="bottom center",
                             marker=dict(color="red", size=12),
                             name="Minimum Difference"))
    
    diff_table = diff_table.reset_index(drop=True)
    return diff_table, fig

# ---------------------------
# Macroeconomic Simulation: VAR 모델을 통해 2013~2022 실제 GDP와 2023~2027 예측, 2022년과 2023년 연결
def macroeconomic_simulation(country, country_df):
    model_data = country_df[["Year", "GDP", "GERD", "Patent Publications"]].dropna()
    if model_data.shape[0] < 5:
        st.warning(f"{country}: 거시경제 시뮬레이션 데이터 부족")
        return None, None, None

    var_model = VAR(model_data[["GDP", "GERD", "Patent Publications"]])
    results = var_model.fit(maxlags=2)
    forecast = results.forecast(model_data[["GDP", "GERD", "Patent Publications"]].values[-2:], steps=5)
    
    if np.issubdtype(model_data["Year"].dtype, np.datetime64):
        last_year = model_data["Year"].max().year
    else:
        last_year = int(model_data["Year"].max())
    forecast_years = list(range(last_year + 1, last_year + 6))
    
    forecast_df = pd.DataFrame(forecast, columns=["GDP", "GERD", "Patent Publications"])
    forecast_df["Year"] = forecast_years
    forecast_df["Year"] = forecast_df["Year"].astype(int)
    
    actual_df = model_data[["Year", "GDP"]].copy()
    if np.issubdtype(actual_df["Year"].dtype, np.datetime64):
        actual_df["Year"] = actual_df["Year"].dt.year
    else:
        actual_df["Year"] = actual_df["Year"].astype(int)
    
    last_actual_year = actual_df["Year"].max()
    last_actual_gdp = actual_df[actual_df["Year"] == last_actual_year]["GDP"].values[0]

    final_forecast_gdp = forecast_df["GDP"].iloc[-1]
    
    gdp_change_amount = final_forecast_gdp - last_actual_gdp
    gdp_change_rate = (gdp_change_amount / last_actual_gdp) * 100
    
    full_df = pd.concat([actual_df[["Year", "GDP"]], forecast_df[["Year", "GDP"]]], ignore_index=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=full_df["Year"], y=full_df["GDP"],
                            mode="lines+markers", name="Actual & Forecasted GDP", line=dict(color="blue")))

    # 연결: 2022년과 2023년 예측값 단순 선 연결
    fig.add_trace(go.Scatter(x=[last_actual_year, forecast_years[0]],
                             y=[last_actual_gdp, forecast_df["GDP"].iloc[0]],
                             mode="lines", line=dict(color="red", dash="dot"), showlegend=False))
    
    max_gdp = full_df["GDP"].max()
    min_gdp = full_df["GDP"].min()
    max_year = full_df.loc[full_df["GDP"].idxmax(), "Year"]
    min_year = full_df.loc[full_df["GDP"].idxmin(), "Year"]
    
    fig.add_trace(go.Scatter(x=[max_year], y=[max_gdp],
                            mode="markers+text", text=[f"Max: {max_gdp:,.2f} 달러"],
                            textposition="top center", marker=dict(color="green", size=12), name="Maximum GDP"))
    
    fig.add_trace(go.Scatter(x=[min_year], y=[min_gdp],
                            mode="markers+text", text=[f"Min: {min_gdp:,.2f} 달러"],
                            textposition="bottom center", marker=dict(color="red", size=12), name="Minimum GDP"))
    
    fig.update_layout(title=f"{country} - Original vs. Forecasted GDP Trends (2013-2027)",
                    xaxis_title="Year", yaxis_title="GDP (달러)", template="plotly_white")
    
    return fig, gdp_change_amount, gdp_change_rate