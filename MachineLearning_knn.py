import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
#from sklearn.inspection import DecisionBoundaryDisplay
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import StandardScaler



hide = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        body {overflow: hidden;}
        div.block-container {padding-top:1rem;}
        div.block-container {padding-bottom:1rem;}
        </style>
        """

st.markdown(hide, unsafe_allow_html=True)

#st.header("k-nearest neighbors")

#url = "https://raw.githubusercontent.com/aimeeschwab-mccoy/streamlit_asm/main/forest_fires.csv"

images20 = {1: "ML_knn_images/knn_streamlit_20-1.png",
                2: "ML_knn_images/knn_streamlit_20-2.png",
                3: "ML_knn_images/knn_streamlit_20-3.png",
                4: "ML_knn_images/knn_streamlit_20-4.png",
                5: "ML_knn_images/knn_streamlit_20-5.png",
                6: "ML_knn_images/knn_streamlit_20-6.png",
                7: "ML_knn_images/knn_streamlit_20-7.png",
                8: "ML_knn_images/knn_streamlit_20-8.png",
                9: "ML_knn_images/knn_streamlit_20-9.png",
                10: "ML_knn_images/knn_streamlit_20-10.png",
                11: "ML_knn_images/knn_streamlit_20-11.png",
                12: "ML_knn_images/knn_streamlit_20-12.png",
                13: "ML_knn_images/knn_streamlit_20-13.png",
                14: "ML_knn_images/knn_streamlit_20-14.png",
                15: "ML_knn_images/knn_streamlit_20-15.png",
                16: "ML_knn_images/knn_streamlit_20-16.png",
                17: "ML_knn_images/knn_streamlit_20-17.png",
                18: "ML_knn_images/knn_streamlit_20-18.png",
                19: "ML_knn_images/knn_streamlit_20-19.png",
                20: "ML_knn_images/knn_streamlit_20-20.png"}

images50 = {1: "ML_knn_images/knn_streamlit_50-1.png",
                2: "ML_knn_images/knn_streamlit_50-2.png",
                3: "ML_knn_images/knn_streamlit_50-3.png",
                4: "ML_knn_images/knn_streamlit_50-4.png",
                5: "ML_knn_images/knn_streamlit_50-5.png",
                6: "ML_knn_images/knn_streamlit_50-6.png",
                7: "ML_knn_images/knn_streamlit_50-7.png",
                8: "ML_knn_images/knn_streamlit_50-8.png",
                9: "ML_knn_images/knn_streamlit_50-9.png",
                10: "ML_knn_images/knn_streamlit_50-10.png",
                11: "ML_knn_images/knn_streamlit_50-11.png",
                12: "ML_knn_images/knn_streamlit_50-12.png",
                13: "ML_knn_images/knn_streamlit_50-13.png",
                14: "ML_knn_images/knn_streamlit_50-14.png",
                15: "ML_knn_images/knn_streamlit_50-15.png",
                16: "ML_knn_images/knn_streamlit_50-16.png",
                17: "ML_knn_images/knn_streamlit_50-17.png",
                18: "ML_knn_images/knn_streamlit_50-18.png",
                19: "ML_knn_images/knn_streamlit_50-19.png",
                20: "ML_knn_images/knn_streamlit_50-20.png"}

col1, col2 = st.columns([1,1])

with col1:

    st.write("Decision boundary for $n=20$")

    k1 = st.slider(label="Select a value of k between 1 and 20.", min_value=1, max_value=20, value=5, step=1, key=1)

    st.image(images20[k1])
   
#    check = st.checkbox("Display frequency table")
#
#    if check:
#        summary = bobross.groupby(categorical).size().to_frame()
#        st.dataframe(summary)

with col2:

    st.write("Decision boundary for $n=50$")

    k2 = st.slider(label="Select a value of k between 1 and 20.", min_value=1, max_value=20, value=5, step=1, key=2)

    st.image(images50[k2])