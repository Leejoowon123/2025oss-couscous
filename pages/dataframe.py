import streamlit as st

# 커넥션 객체 생성
# secrets.toml에서 1번째 줄인 connection.mydb를 사용해서 객체를 만든다는 내용
conn = st.connection('ossdb', type="sql", autocommit= True)

# sql 쿼리 작성
sql = """
    select 
       *
    from 
        master_data_by_category
    where 1=1
    limit 10;
"""

df = conn.query(sql, ttl= 3600)
st.dataframe(df)