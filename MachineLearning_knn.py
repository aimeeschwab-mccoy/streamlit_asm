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
   
    check1 = st.checkbox("Show sample frequencies?", key=1)
    
    if check1:
        st.write("Sample contains 6 Adelie penguins, 8 Chinstrap penguins, and 6 Gentoo penguins.")

    check1a = st.checkbox("Show plot description?", key=3)

    if check1a:
        st.write("Scatterplot with bill length ranging from 35 to 55 mm on horizontal axis and bill depth ranging from 14 to 21 on the vertical axis. Regions corresponding to each class are shaded. Small values of k tend to result in a jagged decision boundary line. As k increases, the region for Chinstrap increases, eventually taking over the entire plot.")


with col2:

    st.write("Decision boundary for $n=50$")

    k2 = st.slider(label="Select a value of k between 1 and 20.", min_value=1, max_value=20, value=5, step=1, key=2)

    st.image(images50[k2])

    check2 = st.checkbox("Show sample frequencies?", key=2)
    
    if check2:
        st.write("Sample contains 25 Adelie penguins, 10 Chinstrap penguins, and 15 Gentoo penguins.")

    check2a = st.checkbox("Show plot description?", key=4)
    
    if check2a:
        st.write("Scatterplot with bill length ranging from 35 to 55 mm on horizontal axis and bill depth ranging from 14 to 21 on the vertical axis. Regions corresponding to each class are shaded. Small values of k tend to result in a jagged decision boundary line. As k increases, the region for Adelie increases. At k=20 three distinct regions still exist, but Adelie penguins are predicted for over half the plot.")
