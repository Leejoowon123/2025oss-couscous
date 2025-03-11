import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def main():
    st.title("Analyze Page")

    conn = st.connection("ossdb", type="sql", autocommit=True)

    sql = """
        select 
        *
        from 
            master_data_by_category
        where 1=1
        ;
    """

    df = conn.query(sql, ttl= 3600)

    # 3. 연도 컬럼(2013 ~ 2022) -> float
    years = [str(year) for year in range(2013, 2023)]
    for y in years:
        df[y] = pd.to_numeric(df[y], errors='coerce').round(4)

    # 4. 카테고리 목록 확인
    if 'category' not in df.columns:
        st.error("Column 'category' not found in data.")
        return
    categories = sorted(df['category'].unique())

    # 기본 선택: 'Corporate Tax'가 있으면 선택, 없으면 첫 번째 항목
    default_cat = "Corporate Tax" if "Corporate Tax" in categories else categories[0]
    cat_choice = st.selectbox("Select Category", categories, index=categories.index(default_cat))

    # 5. 선택된 카테고리 데이터 추출 및 long format으로 변환
    cat_df = df[df['category'] == cat_choice].copy()
    if cat_df.empty:
        st.warning(f"No data available for category '{cat_choice}'.")
        return
    cat_long = cat_df.melt(id_vars=['Country', 'category'],
                           value_vars=years,
                           var_name='Year',
                           value_name='Value')
    cat_long['Year'] = cat_long['Year'].astype(int)

    st.write(f"Data for {cat_choice}:", cat_long)

    # 6. 시계열 추이 그래프 (Plotly Express)
    fig_line = px.line(cat_long, x='Year', y='Value', color='Country',
                       markers=True, title=f"Time Series for {cat_choice}")
    st.plotly_chart(fig_line, use_container_width=True)

    # 7. 국가 간 상관관계 히트맵
    pivot_df = cat_df.set_index('Country')[years]
    pivot_df = pivot_df.dropna(axis=1, how='all') 
    corr_matrix = pivot_df.T.corr().round(4)
    
    # Plotly Heatmap으로 시각화
    fig_heat = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='Viridis',
        zmin=-1, zmax=1,
        colorbar=dict(title="Correlation")
    ))
    fig_heat.update_layout(
        title=f"Correlation Heatmap for {cat_choice}",
        xaxis_title="Country",
        yaxis_title="Country"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

if __name__ == "__main__":
    main()
