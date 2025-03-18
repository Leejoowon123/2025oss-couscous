# Laffer Curve, 최적 세율 등 모델 관련 공통 함수

import numpy as np
from scipy.optimize import curve_fit, differential_evolution
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def laffer_curve(x, a, b, c):
    """
    Laffer Curve 함수:
    x (AI 세율), 
    a, b, c 파라미터에 따른 GDP 예측 함수
    """
    return a * x**2 + b * x + c

def get_optimal_ai_tax_from_proxy(df, proxy_col, target_col="GDP"):
    """
    주어진 Proxy 컬럼과 타깃 변수(GDP)를 이용해 Laffer Curve를 피팅하고
    differential_evolution을 통해 최적 AI 세율을 도출하는 함수.
    """
    X = df[[proxy_col]].values.flatten()
    y = df[target_col].values
    try:
        params, _ = curve_fit(laffer_curve, X, y, p0=[-0.5, 0.1, y.mean()])
    except RuntimeError:
        params = [0, 0, y.mean()]
    def optimize_tax(tax_rate):
        return -laffer_curve(tax_rate, *params)
    bounds = [(0, 0.4)]
    optimal_tax = differential_evolution(optimize_tax, bounds, seed=42).x[0]
    return optimal_tax, params
