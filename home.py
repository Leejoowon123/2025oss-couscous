import streamlit as st

st.title('Multi Page Link')
st.page_link("./home.py", label="Home", icon="🏠")
st.page_link("./pages/analyze.py", label="analyze", icon="1️⃣")
st.page_link("./pages/dataframe.py", label="Dataframe", icon="2️⃣")

# 아이콘을 숨기기
# st.page_link("./pages/text.py", label="Page 2", icon="2️⃣", disabled=True)