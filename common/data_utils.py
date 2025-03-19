# 데이터 로드, 분할 등 공통 함수
import streamlit as st
import pandas as pd

def load_data():
    """
    데이터베이스에서 데이터를 로드하는 함수.
    Streamlit의 connection 기능을 활용.
    """
    conn = st.connection("ossdb", type="sql", autocommit=True)
    sql = "SELECT * FROM master_data_by_category_clear;"
    df = conn.query(sql, ttl=3600)
    return df

def split_by_country(df):
    """
    전체 데이터프레임을 국가별 데이터프레임으로 분할하여 반환.
    """
    country_dfs = {
        "China": df[df["Country"] == "China"].copy(),
        "France": df[df["Country"] == "France"].copy(),
        "USA": df[df["Country"] == "United States of America"].copy(),
        "Germany": df[df["Country"] == "Germany"].copy(),
        "Japan": df[df["Country"] == "Japan"].copy(),
        "Korea": df[df["Country"] == "Korea"].copy(),
        "UK": df[df["Country"] == "United Kingdom"].copy(),
    }
    return country_dfs
