import streamlit as st
from common.data_utils import load_data
from src.Q1_logic import main_analysis

def run_Q1():
    st.title("Q1: AI 세금이 기업 투자와 경제 성장에 미치는 영향")
    st.markdown("""
    ## 종합 결론 및 심층 분석
    ### Ridge 및 Lasso 회귀 결과 분석

    - Ridge 회귀에서는 일부 계수가 통계적으로 유의하지 않은 것으로 나타남 
      e.g. 회귀 계수 중 WIPO Tax와 Corporate Tax의 계수는 표준 오차가 크고, t-통계량이 낮아 P-값이 유의수준(보통 0.05)보다 높은 것으로 확인 -> 변수가 GDP에 미치는 직접적인 효과가 명확하지 않음
   
    - Lasso 회귀에서도 비슷하게, 일부 계수의 값이 0에 매우 가깝거나 0으로 수축되는 경향이 있어, 이들 변수의 설명력이 제한적임
    
    ## IVGMM 모형 결과 상세 분석
    
    - IVGMM 모형은 내생성을 보완하기 위해 전 시점의 GDP (log_GDP_lag)를 내생 변수로, 그 이후 두 시점(log_GDP_lag2, log_GDP_lag3)을 도구 변수로 사용하여 모형을 추정
    - 내생 변수(log_GDP_lag)의 계수: 1.0129로 매우 강한 양의 영향을 보이며, 통계적으로 유의함(P-value < 0.001)
    - 외생 변수(WIPO Tax, Corporate Tax): 모형에 포함되어 있으나, 그 효과는 상대적으로 미미하거나 통계적 유의성이 낮음
    - 모형의 설명력: R²가 약 0.9945로 매우 높음 -> 사용한 도구 변수와 내생 변수 추정이 모형의 적합도에 기여했음
    
    ## 종합 분석: AI 세금이 기업 투자와 경제 성장에 미치는 영향
    ### 기업 투자 측면:
    - 기업 투자 모델(종속 변수:특허 출원 수)에서는 AI 세금과 관련된 변수(WIPO Tax, Corporate Tax)가 통계적으로 유의한 영향을 나타냄. 
      -> AI 관련 세금 정책이 기업의 연구개발 및 특허 출원 활동에 영향을 줄 수 있음
    - 경제 성장 측면:
    - 경제 성장률 모형(종속 변수: GDP의 로그 차분)에서 단기 성장률 변동을 분석한 결과, 일부 세금 변수의 영향은 미미하거나 통계적으로 유의하지 않음. 
      -> 단기 경제 성장에 미치는 AI 세금의 효과가 다른 거시경제 변수나 정책 요소에 비해 상대적으로 약할 수 있음
    
    ## 종합적 고찰
    - 종합적으로, 본 연구는 다양한 회귀 및 동태적 패널 모형을 활용하여 AI 세금이 기업 투자 및 경제 성장에 미치는 영향을 다각도로 분석함
    - OLS 모형은 높은 설명력을 보임
    - Ridge와 Lasso 회귀 결과는 일부 세금 관련 계수의 통계적 유의성이 낮아, 단순 선형 모형으로는 AI 세금의 복잡한 효과를 완벽하게 설명하기 어렵다는 점을 시사
    - IVGMM 모형은 내생성 문제를 보완하며, 특히 전 시점의 GDP가 현재 GDP에 미치는 강한 영향력을 통해 AI 세금 정책이 기업 활동 및 장기 경제 성장에 미치는 구조적 효과를 강조
    - 따라서, 현재 분석 결과는 AI 세금이 기업 투자와 경제 성장에 미치는 영향이 단순한 선형 효과보다 복합적이며, 특히 기업의 미래 투자 결정과 장기 경제 성장에 중요한 영향을 미칠 가능성이 있음을 시사함.
    - 다만, 세부 효과의 크기와 방향은 추가적인 모형 개선 및 외부 변수 고려가 필요하므로, 향후 연구에서는 보다 정교한 모형 및 다양한 도구 변수를 활용한 심층 분석이 요구됨
    """)

    df = load_data()
    results = main_analysis(df)
    
    st.markdown("""
    ### 수행된 분석 내용:
    1. OLS (고정효과) 회귀  
    2. Ridge / Lasso 회귀 (교차검증 포함)  
    3. 동태적 패널 모델 (IVGMM을 통한 2단계 추정 및 도구 변수 지정)  
    4. 기업 투자 모델 (Patent Publications)  
    5. 경제 성장률 모델 (log(GDP) 차분)
    """)
    
    st.subheader("1. OLS (고정효과) 결과")
    st.text(results["ols"].summary())
    
    st.subheader("2. Ridge / Lasso 회귀 결과")
    st.write("**Ridge 회귀 최적 계수:**", results["ridge"].coef_)
    st.write("**Lasso 회귀 최적 계수:**", results["lasso"].coef_)
    
    st.subheader("3. 동태적 패널 모델 (IVGMM) 결과")
    st.text(results["gmm"].summary)
    
    st.subheader("4. 기업 투자 모델 (Patent Publications) 결과")
    st.text(results["investment"].summary())
    
    st.subheader("5. 경제 성장률 모델 (log(GDP) 차분) 결과")
    st.text(results["growth"].summary())
    
    iv_sum = results["iv_summary"]
    st.subheader("IVGMM 데이터 요약")
    st.write("Dependent (log_GDP): shape =", iv_sum["dependent"]["shape"])
    st.write("Endogenous (log_GDP_lag): shape =", iv_sum["endog"]["shape"])
    st.write("Exogenous (WIPO_Tax, Corporate_Tax + constant): shape =", iv_sum["exog"]["shape"])
    st.write("Instruments (log_GDP_lag2, log_GDP_lag3): shape =", iv_sum["instruments"]["shape"])
    
    
if __name__ == "__main__":
    run_Q1()
