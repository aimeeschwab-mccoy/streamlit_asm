import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


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

#st.header("Finding the maximal margin hyperplane")

url = "https://raw.githubusercontent.com/aimeeschwab-mccoy/streamlit_asm/main/forest_fires.csv"

fires = pd.read_csv(url).sample(20, random_state=1)
fires.columns = list(fires.columns)

# Define input and output features
X = fires[['Temp', 'Humidity', 'WindSpeed', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']]
y = fires[['Fire']]

# Relabel some instances for full separation
y.at[113,'Fire']=1
y.at[119,'Fire']=1
y.at[46,'Fire']=0
y.at[67,'Fire']=0

y2 = np.ravel(y)
y2[y2==0] = -1

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled inputs back to a dataframe
X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

col1, col2 = st.columns([2,3])

with col1:

    st.write("$w_1$")

    beta1 = st.slider(label="Choose a value between 0.01 and 1.0.", min_value=0.01, max_value=1.0, value=0.5)

    st.write("$w_2$")

    beta2 = st.slider(label="Choose a value between -1.0 and -0.01.", min_value=-1.0, max_value=0.01, value=-0.5)

    #beta0 = 1 - (beta1**2) - (beta2**2)
    a = -beta1 / beta2
    xx = np.linspace(-3, 3)
    yy = a * xx - (-0.45366098) / beta2

    total = beta1**2 + beta2**2

    M = np.min(y2*(-0.45366098 + beta1*X['Temp'] + beta2*X['Humidity'])/total)

    st.write("Margin = ", round(M, 3), "")

    showmargin = st.checkbox(label="Show maximal margin hyperplane?", value=False)

    if showmargin:

        st.write("The maximal margin hyperplane has $w_0$ = -0.45, $w_1$ = 0.46, and $w_2$ = -0.76.")

    showtext = st.checkbox(label="Show description?", value=False)

    if showtext:

        st.write("The scatterplot shows standardized temperature on the horizontal axis and standardized humidity on the vertical axis. Both features range from -3 to +3. A hyperplane separates two classes. Instances below the hyperplane have high temperatures, low humidity, and are classified as Fire. Instances above the hyperplane have low temperatures, high humidity, and are classified as No fire.")

#    check = st.checkbox("Display frequency table")
#
#    if check:
#        summary = bobross.groupby(categorical).size().to_frame()
#        st.dataframe(summary)

with col2:

    fig, ax = plt.subplots()

    p = sns.scatterplot(data=X, x='Temp', y='Humidity', hue=np.ravel(y), 
        style=np.ravel(y), palette='deep')
    p.set_ylabel("Humidity", fontsize=14)
    p.set_xlabel("Temperature", fontsize=14)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    plt.legend(labels=['Fire', 'No fire'])
    plt.plot(xx, yy, 'red')

    if showmargin:
        a2 = -1.19042809 / -1.99591807
        xx2 = np.linspace(-3, 3)
        yy2 = a2 * xx2 - (-1.18304842) / -1.99591807

        yy3 = yy2 - 0.48
        yy4 = yy2 + 0.48
        plt.plot(xx2, yy2, 'black')
        ax.fill_between(xx2, yy3, yy4, alpha=0.1, color='grey')

    st.pyplot(fig)