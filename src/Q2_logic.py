import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit, differential_evolution
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import streamlit as st

# 랜덤 시드
np.random.seed(42)

# Laffer Curve 함수 정의
def laffer_curve(x, a, b, c):
    return a * x**2 + b * x + c

# 데이터 로드 함수
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

# GDP와 높은 상관관계를 가진 변수 선택
def get_high_corr_vars(country_df, threshold=0.45):

    country_numeric_df = country_df.select_dtypes(include=[np.number]).drop(["GDP", "Year"], axis=1, errors="ignore")
    correlations = country_numeric_df.corrwith(country_df["GDP"]).dropna().abs()
    
    return correlations[correlations > threshold].index.tolist()

# AI 세율 Proxy 후보 생성
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

# 최적 모델 선택
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

# 최적 AI 세율 및 Laffer Curve 파라미터 계산 (국가별 & 전체 데이터)
def get_optimal_ai_tax(country_df):

    high_corr_vars = get_high_corr_vars(country_df)

    country_df = country_df[["Country", "Year", "GDP"] + high_corr_vars]
    proxy_candidates = find_best_proxy(country_df, high_corr_vars)
    best_degree, best_proxy = find_best_model(country_df, proxy_candidates)

    if not best_proxy:
        return None, None

    X_proxy = country_df[[best_proxy]].values.flatten()
    y_gdp = country_df["GDP"].values

    if best_degree > 1:
        try:
            params, _ = curve_fit(laffer_curve, X_proxy, y_gdp, p0=[-0.5, 0.1, y_gdp.mean()])
        except RuntimeError:
            return None, None
    else:
        params = [0, 0, y_gdp.mean()]

    def optimize_tax(tax_rate):
        return -laffer_curve(tax_rate, *params)

    bounds = [(0, 0.4)]
    opt_result = differential_evolution(optimize_tax, bounds, seed=42)
    optimal_tax = opt_result.x[0]

    return optimal_tax, params

# 최적 AI 세율 그래프
def plot_laffer_curve(country, optimal_tax, laffer_params):
    x_vals = np.linspace(0, 0.5, 100)
    y_vals = laffer_curve(x_vals, *laffer_params)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name="Laffer Curve", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=[optimal_tax], y=[laffer_curve(optimal_tax, *laffer_params)], 
                            mode='markers', name="Optimal AI Tax", marker=dict(color="red", size=10)))

    fig.update_layout(title=f"{country} - Optimal AI Tax Rate",
                    xaxis_title="AI Tax Rate",
                    yaxis_title="GDP",
                    template="plotly_white")

    return fig