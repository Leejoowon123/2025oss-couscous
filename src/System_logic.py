import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit, differential_evolution, OptimizeWarning
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.api import VAR
import warnings
import streamlit as st

# 경고 무시
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)

np.random.seed(42)

# ---------------------------------
# 데이터 로드 함수
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

# ---------------------------------
# Laffer Curve 함수
def laffer_curve(x, a, b, c):
    return a * x**2 + b * x + c

# ---------------------------------
# GDP와 높은 상관관계를 가진 변수 선택
def get_high_corr_vars(country_df, threshold=0.45):
    country_numeric_df = country_df.select_dtypes(include=[np.number]).drop(["GDP", "Year"], axis=1, errors="ignore")
    correlations = country_numeric_df.corrwith(country_df["GDP"]).dropna().abs()
    return correlations[correlations > threshold].index.tolist()

# ---------------------------------
# AI 세율 Proxy 후보 생성 (DataFrame 분절 문제 해결)
def find_best_proxy(country_df, high_corr_vars):
    if not high_corr_vars:
        return [], country_df
    proxy_candidates = []
    new_cols = {}
    scaler = MinMaxScaler(feature_range=(0, 0.5))
    for r in range(2, 4):
        for combo in combinations(high_corr_vars, r):
            valid_combo = [col for col in combo if col in country_df.columns]
            if len(valid_combo) == r:
                proxy_name = f"AI_Tax_Proxy_{'_'.join(valid_combo)}"
                new_cols[proxy_name] = pd.Series(
                    scaler.fit_transform(country_df[valid_combo]).mean(axis=1),
                    index=country_df.index
                )
                proxy_candidates.append(proxy_name)
    if new_cols:
        country_df = pd.concat([country_df, pd.DataFrame(new_cols)], axis=1)
    return proxy_candidates, country_df

# ---------------------------------
# 최적 모델 선택 (BIC 기준)
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
            bic = len(y) * np.log(np.maximum(1e-8, np.mean((y - y_pred)**2))) + d * np.log(len(y))
            if bic < best_bic:
                best_bic = bic
                best_degree = d
                best_proxy = proxy
    return best_degree, best_proxy

# ---------------------------------
# 최적 AI 세율 및 Laffer Curve 파라미터 산출 (Q2 방식)
def get_optimal_ai_tax(country_df):
    high_corr_vars = get_high_corr_vars(country_df)
    df_mod = country_df[["Country", "Year", "GDP"] + high_corr_vars].copy()
    proxy_candidates, df_mod = find_best_proxy(df_mod, high_corr_vars)
    best_degree, best_proxy = find_best_model(df_mod, proxy_candidates)
    if not best_proxy:
        st.error("최적 Proxy 후보를 찾지 못했습니다.")
        return None, None
    X_proxy = df_mod[[best_proxy]].values.flatten()
    y_gdp = df_mod["GDP"].values
    X_proxy = np.nan_to_num(X_proxy)
    y_gdp = np.nan_to_num(y_gdp)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            params, _ = curve_fit(laffer_curve, X_proxy, y_gdp, p0=[-0.5, 0.1, y_gdp.mean()], maxfev=10000)
            if np.any(np.isnan(params)):
                raise RuntimeError("NaN in parameters")
    except Exception:
        params = [-0.5, 0.1, y_gdp.mean()]
    def optimize_tax(tax_rate):
        return -laffer_curve(tax_rate, *params)
    bounds = [(0, 0.4)]
    opt_result = differential_evolution(optimize_tax, bounds, seed=42)
    optimal_tax = opt_result.x[0]
    return optimal_tax, params

# ---------------------------------
# 미래 5년 예측 변화 계산: VARMAX 모델을 사용 (exogenous: AI_Tax)
def macroeconomic_simulation(country, country_df, applied_tax, optimal_tax, params):
    # 사용할 변수: GDP, GERD, Unemployment Rate
    required_cols = ["GDP", "GERD", "Unemployment Rate"]
    available_cols = [col for col in required_cols if col in country_df.columns]
    if len(available_cols) < 2:
        st.warning(f"{country}: VAR 모델에 필요한 데이터 부족")
        return None
    model_data = country_df[available_cols].dropna().copy()
    if model_data.shape[0] < 5:
        st.warning(f"{country}: VAR 모델에 필요한 데이터 부족")
        return None

    # exogenous 변수 생성: 역사적 데이터에서는 최적 세율을 사용하여 AI_Tax 열 생성
    n = model_data.shape[0]
    exog_hist = np.full((n, 1), optimal_tax)
    
    # VARMAX 모델 학습 (exogenous 포함)
    from statsmodels.tsa.statespace.varmax import VARMAX
    try:
        model = VARMAX(model_data, exog=exog_hist, order=(2,0),
                       enforce_stationarity=False,
                       enforce_invertibility=False)
        results = model.fit(disp=False, method='nm', maxiter=1000)
    except Exception as e:
        st.error("VARMAX 모델 학습 오류: " + str(e))
        return None

    steps = 5
    # 예측 시 exogenous 변수로 사용자가 입력한 AI 세율 적용
    exog_forecast = np.full((steps, 1), applied_tax)
    try:
        forecast_obj = results.get_forecast(steps=steps, exog=exog_forecast)
        forecast_df = forecast_obj.predicted_mean
    except Exception as e:
        st.error("VARMAX 예측 오류: " + str(e))
        return None
    forecast_df["Year"] = list(range(country_df["Year"].max()+1, country_df["Year"].max()+6))
    return forecast_df

# ---------------------------------
# 변화량 계산 함수: 예측값 표에서 첫 행은 2023년 = (forecast 2023 - 2022 실제), 이후는 연속 차분
def compute_changes(forecast_df, actual_values):
    # forecast_df: DataFrame of forecast values (indexed by forecast year)
    # actual_values: Series of 2022 실제 값 for each variable (index = 변수 이름)
    changes = forecast_df.diff().fillna(0)
    # 첫 행: forecast_df.iloc[0] - actual_values
    first_change = forecast_df.iloc[0] - actual_values
    changes.iloc[0] = first_change
    return changes
