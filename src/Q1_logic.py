import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV

from linearmodels.iv import IVGMM

import streamlit as st

# 디버깅용 출력은 제거하고, 핵심 결과만 반환합니다.
def get_balanced_panel(df):
    # 필요한 컬럼들을 numeric으로 변환
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['GDP'] = pd.to_numeric(df['GDP'], errors='coerce')
    df['WIPO Tax'] = pd.to_numeric(df['WIPO Tax'], errors='coerce')
    df['Corporate Tax'] = pd.to_numeric(df['Corporate Tax'], errors='coerce')
    
    # 각 국가별 공통 연도 추출
    counts = df.groupby('Country')['Year'].nunique()
    common_years = None
    for country, group in df.groupby('Country'):
        yrs = set(group['Year'].dropna().unique())
        if common_years is None:
            common_years = yrs
        else:
            common_years = common_years.intersection(yrs)
    if not common_years:
        raise ValueError("공통 연도가 없습니다. 패널 데이터를 확인하세요.")
    common_years = sorted(list(common_years))
    
    balanced = df[df['Year'].isin(common_years)].copy()
    return balanced, counts, common_years

def run_ols_with_fe(df):
    df['GDP'] = pd.to_numeric(df['GDP'], errors='coerce')
    df['WIPO Tax'] = pd.to_numeric(df['WIPO Tax'], errors='coerce')
    df['Corporate Tax'] = pd.to_numeric(df['Corporate Tax'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    df = df.sort_values(by=["Country", "Year"]).copy()
    df['GDP_lag'] = df.groupby('Country')['GDP'].shift(1)
    
    required = ['Country', 'Year', 'GDP', 'WIPO Tax', 'Corporate Tax', 'GDP_lag']
    df_panel = df.dropna(subset=required)
    
    dummies = pd.get_dummies(df_panel['Country'], drop_first=True)
    X = df_panel[['GDP_lag', 'WIPO Tax', 'Corporate Tax']]
    X = pd.concat([X, dummies], axis=1)
    X = sm.add_constant(X)
    y = df_panel['GDP']
    
    X = X.astype(float)
    y = y.astype(float)
    
    model = sm.OLS(y, X).fit(cov_type='HC3')
    return model, df_panel

def run_ridge_lasso(df_panel):
    dummies = pd.get_dummies(df_panel['Country'], drop_first=True)
    X = df_panel[['GDP_lag', 'WIPO Tax', 'Corporate Tax']]
    X = pd.concat([X, dummies], axis=1)
    y = df_panel['GDP']
    
    param_grid = {'alpha': np.logspace(-3, 3, 50)}
    
    ridge = Ridge(random_state=42)
    ridge_cv = GridSearchCV(ridge, param_grid, cv=5)
    ridge_cv.fit(X, y)
    best_ridge = ridge_cv.best_estimator_
    
    lasso = Lasso(random_state=42, max_iter=10000)
    lasso_cv = GridSearchCV(lasso, param_grid, cv=5)
    lasso_cv.fit(X, y)
    best_lasso = lasso_cv.best_estimator_
    
    return best_ridge, best_lasso, X.columns, None

def run_dynamic_panel_gmm(df):
    """
    동태적 패널 IV GMM 모형 수행:
      - dependent: log(GDP)
      - endog: log(GDP)_lag
      - instruments: log(GDP)_lag2, log(GDP)_lag3
      - exog: 상수항 및 외생 변수 (WIPO Tax, Corporate Tax)
    """
    balanced, unbalanced_counts, common_years = get_balanced_panel(df)
    
    # 로그 변수 및 시차 변수 생성
    balanced['log_GDP'] = np.log(balanced['GDP'].replace({0: np.nan}))
    balanced['log_GDP_lag'] = balanced.groupby('Country')['log_GDP'].shift(1)
    balanced['log_GDP_lag2'] = balanced.groupby('Country')['log_GDP'].shift(2)
    balanced['log_GDP_lag3'] = balanced.groupby('Country')['log_GDP'].shift(3)
    
    required = ['log_GDP', 'log_GDP_lag', 'log_GDP_lag2', 'log_GDP_lag3', 'WIPO Tax', 'Corporate Tax']
    balanced = balanced.dropna(subset=required).copy()
    
    # IV 모형에 사용할 변수만 선택하고, MultiIndex 제거
    panel_iv = balanced[['log_GDP', 'log_GDP_lag', 'log_GDP_lag2', 'log_GDP_lag3', 'WIPO Tax', 'Corporate Tax']].copy()
    panel_iv = panel_iv.rename(columns={'WIPO Tax': 'WIPO_Tax', 'Corporate Tax': 'Corporate_Tax'})
    panel_iv = panel_iv.reset_index(drop=True)
    
    for col in panel_iv.columns:
        panel_iv[col] = pd.to_numeric(panel_iv[col], errors='coerce')
    panel_iv = panel_iv.dropna()
    
    # 변수 추출: 모두 numpy 배열
    dependent = panel_iv['log_GDP'].values.astype(float)
    endog = panel_iv['log_GDP_lag'].values.astype(float)
    exog = sm.add_constant(panel_iv[['WIPO_Tax', 'Corporate_Tax']].values.astype(float))
    instruments = panel_iv[['log_GDP_lag2', 'log_GDP_lag3']].values.astype(float)
    
    # iv_summary에 각 변수의 정보 저장 (출력은 하지 않고, 나중에 참고용으로 보관)
    iv_summary = {
        "dependent": {"variable": "log_GDP", "shape": dependent.shape},
        "endog": {"variable": "log_GDP_lag", "shape": endog.shape},
        "exog": {"variable": "WIPO_Tax, Corporate_Tax (with constant)", "shape": exog.shape},
        "instruments": {"variable": "log_GDP_lag2, log_GDP_lag3", "shape": instruments.shape}
    }
    
    try:
        mod = IVGMM(dependent=dependent, endog=endog, instruments=instruments, exog=exog)
        res = mod.fit(cov_type='robust')  # iterations 인자는 fit()에 전달하지 않습니다.
    except Exception as e:
        st.error("IVGMM 모델 피팅 오류: " + str(e))
        raise e
    return res, panel_iv, unbalanced_counts, common_years, iv_summary

def run_investment_model(df):
    df['Patent Publications'] = pd.to_numeric(df['Patent Publications'], errors='coerce')
    df['GDP'] = pd.to_numeric(df['GDP'], errors='coerce')
    df['WIPO Tax'] = pd.to_numeric(df['WIPO Tax'], errors='coerce')
    df['Corporate Tax'] = pd.to_numeric(df['Corporate Tax'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    df = df.sort_values(by=['Country', 'Year']).copy()
    df_panel = df.dropna(subset=['Country', 'Year', 'Patent Publications', 'WIPO Tax', 'Corporate Tax', 'GDP'])
    
    dummies = pd.get_dummies(df_panel['Country'], drop_first=True)
    X = df_panel[['WIPO Tax', 'Corporate Tax', 'GDP']]
    X = pd.concat([X, dummies], axis=1)
    X = sm.add_constant(X)
    y = df_panel['Patent Publications']
    
    X = X.astype(float)
    y = y.astype(float)
    
    model = sm.OLS(y, X).fit(cov_type='HC3')
    return model

def run_growth_model(df):
    df['GDP'] = pd.to_numeric(df['GDP'], errors='coerce')
    df['WIPO Tax'] = pd.to_numeric(df['WIPO Tax'], errors='coerce')
    df['Corporate Tax'] = pd.to_numeric(df['Corporate Tax'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    df = df.sort_values(by=['Country', 'Year']).copy()
    df['log_GDP'] = np.log(df['GDP'].replace({0: np.nan}))
    df['log_GDP_diff'] = df.groupby('Country')['log_GDP'].diff(1)
    
    df_panel = df.dropna(subset=['log_GDP_diff', 'WIPO Tax', 'Corporate Tax'])
    
    dummies = pd.get_dummies(df_panel['Country'], drop_first=True)
    X = df_panel[['WIPO Tax', 'Corporate Tax']]
    X = pd.concat([X, dummies], axis=1)
    X = sm.add_constant(X)
    y = df_panel['log_GDP_diff']
    
    X = X.astype(float)
    y = y.astype(float)
    
    model = sm.OLS(y, X).fit(cov_type='HC3')
    return model

def main_analysis(df):
    # 균형 패널 구성
    balanced, unbalanced_counts, common_years = get_balanced_panel(df)
    
    # OLS with Fixed Effects
    ols_model, df_panel_ols = run_ols_with_fe(balanced)
    
    # Ridge / Lasso 회귀
    best_ridge, best_lasso, X_cols, _ = run_ridge_lasso(df_panel_ols)
    
    # 동태적 패널 IV GMM
    gmm_res, panel_iv, unbal_counts, comm_years, iv_summary = run_dynamic_panel_gmm(balanced)
    
    # 기업 투자 모델
    invest_model = run_investment_model(balanced)
    
    # 경제 성장률 모델
    growth_model_ = run_growth_model(balanced)
    
    results = {
        "ols": ols_model,
        "ridge": best_ridge,
        "lasso": best_lasso,
        "gmm": gmm_res,
        "investment": invest_model,
        "growth": growth_model_,
        "balanced_panel": panel_iv,
        "unbalanced_counts": unbalanced_counts,
        "common_years": common_years,
        "iv_summary": iv_summary
    }
    return results