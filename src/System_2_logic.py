# System_2_logic.py
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.varmax import VARMAX
import warnings
from scipy.optimize import curve_fit, differential_evolution, OptimizeWarning
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)
np.random.seed(42)

# 1. 데이터 로드 함수
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

# 2. Laffer Curve 함수
def laffer_curve(x, a, b, c):
    return a * x**2 + b * x + c

# 3. GDP 기준 상관관계가 높은 변수 후보 선택
def get_high_corr_vars(country_df, threshold=0.45):
    country_numeric_df = country_df.select_dtypes(include=[np.number]).drop(["GDP", "Year"], axis=1, errors="ignore")
    correlations = country_numeric_df.corrwith(country_df["GDP"]).dropna().abs()
    return correlations[correlations > threshold].index.tolist()

# 3-1. target_var 기준 상관관계가 높은 변수 후보 선택
def get_high_corr_vars_target(country_df, target_var, threshold=0.45):
    numeric_cols = country_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in [target_var, "Year"]]
    if not numeric_cols:
        return []
    proxies = []
    for col in numeric_cols:
        corr_val = np.abs(country_df[col].corr(country_df[target_var]))
        if corr_val >= threshold:
            proxies.append(col)
    return proxies

# 4. 최적 Proxy 후보 생성 (사용자 선택 변수 포함 옵션 추가)
def find_best_proxy(country_df, high_corr_vars, user_target=None):
    proxy_candidates = []
    new_cols = {}
    scaler = MinMaxScaler(feature_range=(0, 0.5))
    if user_target is not None:
        candidate_vars = list(set(high_corr_vars) | {user_target})
        # 사용자 선택 변수가 반드시 포함되도록 크기 1~3 조합 생성
        for r in range(1, 4):
            for combo in combinations(candidate_vars, r):
                if user_target not in combo:
                    continue
                valid_combo = [col for col in combo if col in country_df.columns]
                if len(valid_combo) == r:
                    proxy_name = f"AI_Tax_Proxy_{'_'.join(valid_combo)}"
                    new_cols[proxy_name] = pd.Series(
                        scaler.fit_transform(country_df[valid_combo]).mean(axis=1),
                        index=country_df.index
                    )
                    proxy_candidates.append(proxy_name)
    else:
        candidate_vars = high_corr_vars
        for r in range(2, 4):
            for combo in combinations(candidate_vars, r):
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

# 5. 최적 모델 선택 (BIC 기준) - GDP 기준
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

# 5-1. target_var 기준 최적 모델 선택 (BIC 기준)
def find_best_model_target(country_df, proxy_candidates, target_var):
    if not proxy_candidates:
        return None, None
    best_degree = 1
    best_bic = np.inf
    best_proxy = None
    for proxy in proxy_candidates:
        X = country_df[[proxy]].values
        y = country_df[target_var].values
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

# 6. GDP 기준 최적 AI 세율 도출
def get_optimal_ai_tax(country_df):
    high_corr_vars = get_high_corr_vars(country_df)
    df_mod = country_df[["Country", "Year", "GDP"] + high_corr_vars].copy()
    proxy_candidates, df_mod = find_best_proxy(df_mod, high_corr_vars)
    best_degree, best_proxy = find_best_model(df_mod, proxy_candidates)
    if not best_proxy:
        st.error("최적 Proxy 후보를 찾지 못했습니다.")
        return None, None
    X_proxy = df_mod[[best_proxy]].values.flatten()
    y_target = df_mod["GDP"].values
    X_proxy = np.nan_to_num(X_proxy)
    y_target = np.nan_to_num(y_target)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            params, _ = curve_fit(laffer_curve, X_proxy, y_target, p0=[-0.5, 0.1, y_target.mean()], maxfev=10000)
            if np.any(np.isnan(params)):
                raise RuntimeError("NaN in parameters")
    except Exception:
        params = [-0.5, 0.1, y_target.mean()]
    def optimize_tax(tax_rate):
        return -laffer_curve(tax_rate, *params)
    bounds = [(0, 0.4)]
    opt_result = differential_evolution(optimize_tax, bounds, seed=42)
    optimal_tax = opt_result.x[0]
    return optimal_tax, params

# 7. 사용자 선택(비GDP) 변수 기준 최적 AI 세율 도출 (1단계와 동일 방식 적용)
def get_optimal_ai_tax_by_target(country_df, target_var):
    high_corr_vars = get_high_corr_vars_target(country_df, target_var, threshold=0.45)
    df_mod = country_df[["Country", "Year", target_var] + high_corr_vars].copy()
    # 사용자 선택 변수가 반드시 포함되도록 find_best_proxy 호출
    proxy_candidates, df_mod = find_best_proxy(df_mod, high_corr_vars, user_target=target_var)
    best_degree, best_proxy = find_best_model_target(df_mod, proxy_candidates, target_var)
    if not best_proxy:
        st.error("최적 Proxy 후보를 찾지 못했습니다.")
        return None, None
    X_proxy = df_mod[[best_proxy]].values.flatten()
    y_target = df_mod[target_var].values
    X_proxy = np.nan_to_num(X_proxy)
    y_target = np.nan_to_num(y_target)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            params, _ = curve_fit(laffer_curve, X_proxy, y_target, p0=[-0.5, 0.1, y_target.mean()], maxfev=10000)
            if np.any(np.isnan(params)):
                raise RuntimeError("NaN in parameters")
    except Exception:
        params = [-0.5, 0.1, y_target.mean()]
    def optimize_tax(tax_rate):
        return -laffer_curve(tax_rate, *params)
    bounds = [(0, 0.4)]
    opt_result = differential_evolution(optimize_tax, bounds, seed=42)
    optimal_tax = opt_result.x[0]
    return optimal_tax, params

# 8. VARMAX 모델을 사용한 미래 5년 예측 (exogenous: AI_Tax)
def macroeconomic_simulation(country, country_df, applied_tax, optimal_tax, params, target_var=None):
    # 사용할 변수: 기본적으로 GDP, GERD, Unemployment Rate
    required_cols = ["GDP", "GERD", "Unemployment Rate"]
    if target_var is not None and target_var in country_df.columns and target_var not in required_cols:
        required_cols = [target_var] + required_cols
    available_cols = [col for col in required_cols if col in country_df.columns]
    if len(available_cols) < 2:
        st.warning(f"{country}: VAR 모델에 필요한 데이터 부족")
        return None
    model_data = country_df[available_cols].dropna().copy()
    if model_data.shape[0] < 5:
        st.warning(f"{country}: VAR 모델에 필요한 데이터 부족")
        return None
    n = model_data.shape[0]
    exog_hist = np.full((n, 1), optimal_tax)
    
    results = None
    orders_to_try = [(2,0), (1,0)]
    for order in orders_to_try:
        try:
            model = VARMAX(model_data, exog=exog_hist, order=order,
                           enforce_stationarity=False,
                           enforce_invertibility=False)
            try:
                results = model.fit(disp=False, method='lbfgs', maxiter=500)
            except Exception as e_lbfgs:
                try:
                    results = model.fit(disp=False, method='nm', maxiter=500)
                except Exception as e_nm:
                    results = model.fit(disp=False, method='powell', maxiter=1000)
            if results is not None:
                break
        except Exception as e:
            if "Schur decomposition" in str(e):
                continue
            else:
                st.error("VARMAX 모델 학습 오류: " + str(e))
                return None
    if results is None:
        st.error("VARMAX 모델 학습 오류: 모든 시도가 실패했습니다.")
        return None

    steps = 5
    exog_forecast = np.full((steps, 1), applied_tax)
    try:
        forecast_obj = results.get_forecast(steps=steps, exog=exog_forecast)
        forecast_df = forecast_obj.predicted_mean
    except Exception as e:
        st.error("VARMAX 예측 오류: " + str(e))
        return None
    if "Year" in model_data.columns:
        last_year = int(model_data["Year"].max())
    elif "Year" in country_df.columns:
        last_year = int(country_df["Year"].max())
    else:
        last_year = 0
    forecast_df["Year"] = [last_year + i for i in range(1, steps+1)]
    cols = forecast_df.columns.tolist()
    if "Year" in cols:
        cols.remove("Year")
        forecast_df = forecast_df[["Year"] + cols]
    return forecast_df

# 9. 변화량 계산 함수
def compute_gain(baseline, forecast_val, var):
    baseline_val = pd.to_numeric(baseline, errors='coerce')
    forecast_val = pd.to_numeric(forecast_val, errors='coerce')
    if pd.isna(baseline_val) or pd.isna(forecast_val):
        return np.nan
    if var == "Unemployment Rate":
        return baseline_val - forecast_val
    else:
        return forecast_val - baseline_val
