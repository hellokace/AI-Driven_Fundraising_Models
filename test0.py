import streamlit as st
import pandas as pd
import numpy as np

st.title("This is for Capstone")

st.header("_Streamlit_ is :blue[cool?] :sunglasses:", divider="rainbow")

st.markdown("This is a markdown text")

#x = st.slider("Choose an ***x*** value",1,10)

# to print the value selected in the slider
#st.write("The value of ***x*** is ", x)

# to add the slider and the text in 2 columns

col1, col2 = st.columns(2)

with col1:
    y = st.slider("Choose an ***y*** value",1,10)
with col2:
    st.write("The value of ***y*** is ", y)


st.header("Chart Elements", divider="red")

chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

st.bar_chart(chart_data)