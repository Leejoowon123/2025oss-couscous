import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.append(SRC_DIR)

from Q2_logic import load_data, get_optimal_ai_tax, plot_laffer_curve


st.title("Q2: AI 세율의 최적 수준은 어떻게 설정할 것인가")

st.markdown("""
## AI 세율 최적화 분석

본 연구는 AI 세율이 경제에 미치는 영향을 분석하고 최적의 AI 세율을 도출하는 것을 목표로 한다.  
AI 산업의 성장과 관련하여 세율 정책이 기업 투자 및 경제 성장에 미치는 영향을 탐색하였다.

### 1. 데이터 수집 및 전처리
본 연구에서는 주요 경제 및 기술 변수들을 포함하는 데이터셋을 구축하였다.  
주요 변수들은 다음과 같다:
- **GDP**: 국가의 경제 성장 수준을 나타냄
- **AI_Tax_Proxy**: AI 산업 관련 세율을 나타내는 대리 변수
- **기타 경제적 요인**: 인터넷 사용률, 연구개발 투자(GERD), 특허 출원 등

### 2. AI 세율과 경제 성장의 관계 분석
본 연구에서는 AI 세율이 경제 성장에 미치는 영향을 평가하기 위해 다음 방법론을 사용하였다:
1. **상관관계 분석**: AI 세율과 GDP 간의 관계를 평가
2. **Laffer Curve 모델 적용**: AI 세율이 경제 성장에 미치는 영향을 분석하여 최적 세율을 도출
3. **최적화 기법**: Laffer Curve를 기반으로 AI 세율을 최적화

### 3. 최적 AI 세율 도출
- 국가별 최적 AI 세율은 Laffer Curve를 이용하여 도출되었음
- 주요 국가의 최적 AI 세율은 다음과 같음:
    - **China**: 9.6%
    - **France**: 9.6%
    - **USA**: 11.49%
    - **Germany**: 9.6%
    - **Japan**: 9.6%
    - **Korea**: 9.6%
    - **UK**: 9.6%

---

## 4. 결론 및 정책적 시사점
본 연구를 통해 AI 세율이 경제 성장에 미치는 영향을 분석하였다.  
도출된 최적 세율을 기반으로 정책적 시사점을 제안할 수 있다:
- **AI 산업 성장 지원**: AI 관련 세율을 적절히 조정하여 기업 투자 유인을 강화
- **경제 성장과 균형 유지**: 세율이 너무 낮거나 높을 경우 경제 성장에 미치는 부정적 영향을 고려
- **지속적인 모니터링 필요**: 국가별 경제적 상황에 따라 최적 세율이 달라질 수 있으므로, 지속적인 분석 필요

""")

df, country_dfs = load_data()

# 국가 선택
st.header("국가별 최적 AI 세율 분석")
selected_country = st.selectbox("분석할 국가를 선택하세요", list(country_dfs.keys()))

# 선택한 국가의 최적 AI 세율 분석
if selected_country:
    country_df = country_dfs[selected_country]
    optimal_tax, laffer_params = get_optimal_ai_tax(country_df)
    
    if optimal_tax is not None and laffer_params is not None:
        st.write(f"**{selected_country} 최적 AI 세율:** {optimal_tax:.4f}")
        st.plotly_chart(plot_laffer_curve(selected_country, optimal_tax, laffer_params))

    else:
        st.warning(f"{selected_country} 데이터 분석 실패")