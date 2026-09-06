import streamlit as st

st.set_page_config(page_title="HW Manager")

# Multi-page 'HW Manager' application.
# Page files live in the HW/ subdirectory; paths are relative to this file.
hw1 = st.Page("HW/HW1.py", title="HW 1")
hw2 = st.Page("HW/HW2.py", title="HW 2", default=True)

pg = st.navigation([hw1, hw2])
pg.run()