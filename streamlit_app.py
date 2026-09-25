import streamlit as st

st.set_page_config(page_title="HW Manager")

# Multi-page 'HW Manager' application.
# Page files live in the HW/ subdirectory; paths are relative to this file.
hw1 = st.Page("HW/HW1.py", title="HW 1")
hw2 = st.Page("HW/HW2.py", title="HW 2")
hw3 = st.Page("HW/HW3.py", title="HW 3")
hw4 = st.Page("HW/HW4.py", title="HW 4")
hw5 = st.Page("HW/HW5.py", title="HW 5", default=True)

pg = st.navigation([hw1, hw2, hw3, hw4, hw5])
pg.run()