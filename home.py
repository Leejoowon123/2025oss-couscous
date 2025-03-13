import streamlit as st

st.title('Multi Page Link')
st.page_link("./home.py", label="Home", icon="🏠")
st.page_link("./pages/analyze.py", label="analyze")
st.page_link("./pages/data.py", label="Data")
st.page_link("./pages/Q1.py", label="Qusetion 1")
st.page_link("./pages/Q2.py", label="Question 2")
st.page_link("./pages/Q3.py", label="Qusetion 3")

# 아이콘을 숨기기
# st.page_link("./pages/text.py", label="Page 2", icon="2️⃣", disabled=True)