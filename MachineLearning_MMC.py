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

st.header("Finding the maximal margin hyperplane")

url = "https://raw.githubusercontent.com/aimeeschwab-mccoy/streamlit_asm/main/forest_fires.csv"

fires = pd.read_csv(url).sample(20, random_state=1)
fires.columns = list(fires.columns)

# Define input and output features
X = fires[['Temp', 'Humidity', 'WindSpeed', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']]
y = fires[['Fire']]

# Relabel some instances for full separation
y.at[113,'Fire']=1
y.at[119,'Fire']=1
y.at[46,'Fire']=-1
y.at[67,'Fire']=-1

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled inputs back to a dataframe
X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

col1, col2 = st.columns([2,3])

with col1:

    beta1 = st.slider(label="Beta1", min_value=0.0, max_value=1.0, value=0.5)
    beta2 = st.slider(label="Beta2", min_value=-1.0, max_value=0.0, value=-0.5)

    #beta0 = 1 - (beta1**2) - (beta2**2)
    a = -beta1 / beta2
    xx = np.linspace(-3, 3)
    yy = a * xx - (-0.45366098) / beta2

    total = (-0.45366098)**2 + beta1**2 + beta2**2

    M = np.min(np.ravel(y)*(-0.45366098 + beta1*X['Temp'] + beta2*X['Humidity'])/total)

    st.write("For beta1=", beta1, " and beta2=", beta2, ", the margin is M=", round(M, 3), ".")

    showmargin = st.checkbox(label="Show maximal margin hyperplane?", value=False)

    if showmargin:

        st.write("The maximal margin hyperplane has beta1 = 0.46 and beta2 = -0.76.")

#    check = st.checkbox("Display frequency table")
#
#    if check:
#        summary = bobross.groupby(categorical).size().to_frame()
#        st.dataframe(summary)

with col2:

    fig, ax = plt.subplots()

    p = sns.scatterplot(data=X, x='Temp', y='Humidity', hue=np.ravel(y), 
        style=np.ravel(y), palette='muted')
    p.set_ylabel("Humidity")
    p.set_xlabel("Temperature")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
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