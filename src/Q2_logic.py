# src/Q2_logic.py
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from itertools import combinations
from scipy.optimize import curve_fit, differential_evolution, dual_annealing, shgo
import plotly.graph_objects as go
from sklearn.metrics import r2_score
import statsmodels.api as sm
from linearmodels.iv import IVGMM

########################################
# 1. Laffer Curve (Quadratic) Definition
########################################

def laffer_curve(x, a, b, c):
    """
    라퍼 곡선 함수 (Quadratic)
    GDP = a * x^2 + b * x + c
    """
    return a * x**2 + b * x + c

########################################
# 2. Single Variable Approach with File Caching
########################################

def fit_laffer_curve_single(df, var, gdp_col="GDP"):
    """
    단일 변수 var와 GDP 간의 라퍼 곡선 피팅 후,
    결정계수(R²), 추정 파라미터(popt), 사용 데이터, 그리고 x 값을 반환합니다.
    """
    data = df.dropna(subset=[var, gdp_col]).copy()
    if len(data) < 5:
        return None, None, None, None
    x = data[var].values
    y = data[gdp_col].values

    p0 = [-0.1, 1.0, np.mean(y)]
    try:
        popt, _ = curve_fit(laffer_curve, x, y, p0=p0)
    except Exception as e:
        st.warning(f"[{var}] 라퍼 곡선 피팅 실패: {e}")
        return None, None, None, data
    y_pred = laffer_curve(x, *popt)
    r2 = r2_score(y, y_pred)
    return popt, r2, data, x

def optimize_tax_multiple_algorithms(popt, x_bound=(0.0, 5.0)):
    """
    여러 최적화 알고리즘을 사용하여, 추정된 라퍼 곡선에서 GDP를 최대화하는 세율을 찾습니다.
    반환: { "algorithm_name": (optimal_tax, max_predicted_GDP), ... }
    """
    def objective(t):
        return -laffer_curve(t, *popt)

    results = {}
    try:
        res_de = differential_evolution(objective, bounds=[x_bound])
        tax_de = res_de.x[0]
        max_gdp_de = laffer_curve(tax_de, *popt)
        results["DifferentialEvolution"] = (tax_de, max_gdp_de)
    except Exception as e:
        results["DifferentialEvolution"] = (None, None)
        st.warning(f"differential_evolution 최적화 실패: {e}")
    try:
        res_da = dual_annealing(objective, bounds=[x_bound])
        tax_da = res_da.x[0]
        max_gdp_da = laffer_curve(tax_da, *popt)
        results["DualAnnealing"] = (tax_da, max_gdp_da)
    except Exception as e:
        results["DualAnnealing"] = (None, None)
        st.warning(f"dual_annealing 최적화 실패: {e}")
    try:
        res_shgo = shgo(objective, bounds=[x_bound])
        tax_shgo = res_shgo.x[0]
        max_gdp_shgo = laffer_curve(tax_shgo, *popt)
        results["SHGO"] = (tax_shgo, max_gdp_shgo)
    except Exception as e:
        results["SHGO"] = (None, None)
        st.warning(f"shgo 최적화 실패: {e}")
    return results

########################################
# 3. Combination Variable Approach with File Caching
########################################

def combine_variables(df, vars_to_combine):
    """
    지정된 변수 목록(vars_to_combine)을 2~3개 조합하여,
    단순 평균으로 복합 변수를 생성합니다.
    """
    cols = list(vars_to_combine)
    subset = df.dropna(subset=cols)
    if len(subset) < 5:
        return None
    subset = subset.copy()
    subset["combined_var"] = subset[cols].mean(axis=1)
    return subset

def build_combinations_of_variables(df, candidate_vars, max_comb=3, r2_filter=0.8):
    """
    candidate_vars 중 2~3개 조합을 생성하여 복합 변수를 구성하고,
    복합 변수에 대해 라퍼 곡선 피팅 및 최적 세율 도출 결과를 반환합니다.
    결과는 CSV 파일("data/combo_results.csv")에 저장하며, 파일이 있으면 이를 불러옵니다.
    최종 결과 중 결정계수(R²)가 r2_filter 이상인 결과만 반환합니다.
    """
    base_path = os.path.join("data")
    os.makedirs(base_path, exist_ok=True)
    filename = os.path.join(base_path, "combo_results.csv")
    
    if os.path.exists(filename):
        try:
            results_df = pd.read_csv(filename)
            if "OptResults" in results_df.columns:
                results_df["OptResults"] = results_df["OptResults"].apply(lambda s: json.loads(s))
            filtered = results_df[results_df["R2"] >= r2_filter]
            return filtered.to_dict("records")
        except Exception as e:
            st.warning(f"복합 변수 결과 CSV 로드 실패: {e}")
    
    combos = []
    for r in range(2, max_comb+1):
        for combo in combinations(candidate_vars, r):
            combos.append(combo)
    results = []
    for combo in combos:
        merged_df = combine_variables(df, list(combo))
        if merged_df is None:
            continue
        var_name = "+".join(combo)
        popt, r2, data, x = fit_laffer_curve_single(merged_df, "combined_var")
        if popt is None:
            continue
        opt_results = optimize_tax_multiple_algorithms(popt, x_bound=(0.0, 5.0))
        result_row = {
            "Combo": str(combo),
            "NewVarName": var_name,
            "R2": r2,
            "Params": popt.tolist(),
            "OptResults": opt_results
        }
        if r2 >= r2_filter:
            results.append(result_row)
    results_df = pd.DataFrame(results)
    try:
        results_df["OptResults"] = results_df["OptResults"].apply(json.dumps)
        results_df.to_csv(filename, index=False, quoting=1)
        results_df["OptResults"] = results_df["OptResults"].apply(json.loads)
    except Exception as e:
        st.warning(f"CSV 저장 실패: {e}")
    return results_df.to_dict("records")

########################################
# 4. 종합 실행 로직
########################################

def filter_variables_by_policy_and_r2(df, var_list, r2_threshold=0.3):
    """
    정책적 타당성과 결정계수(R²)를 고려하여,
    변수명에 'tax', 'revenue', 'unemployment' 등이 포함되고 R² >= r2_threshold인 변수를 선별합니다.
    """
    selected = []
    for var in var_list:
        if any(kw in var.lower() for kw in ["tax", "revenue", "unemployment"]):
            popt, r2, data, x = fit_laffer_curve_single(df, var)
            if popt is not None and r2 is not None and r2 >= r2_threshold:
                selected.append((var, r2, popt))
    return selected

def analyze_single_variables(df, exclude_cols={"Country", "year", "Year", "GDP"}):
    """
    단일 변수 접근: Country, year, GDP를 제외한 변수에 대해 라퍼 곡선 피팅 및 최적 세율 도출 결과를 반환합니다.
    """
    candidate_vars = [c for c in df.columns if c not in exclude_cols]
    results_list = []
    for var in candidate_vars:
        popt, r2, data, x = fit_laffer_curve_single(df, var)
        if popt is None:
            continue
        opt_results = optimize_tax_multiple_algorithms(popt, x_bound=(0.0, 5.0))
        results_list.append({
            "Variable": var,
            "R2": r2,
            "Params": popt.tolist(),
            "OptResults": opt_results
        })
    return results_list

def analyze_all(df):
    """
    전체 단일 변수 접근과 복합 변수 접근 결과를 모두 반환합니다.
    """
    candidate_vars = [c for c in df.columns if c not in ["Country", "year", "Year", "GDP"]]
    single_results = analyze_single_variables(df)
    combo_results = build_combinations_of_variables(df, candidate_vars, max_comb=3, r2_filter=0.8)
    return single_results, combo_results