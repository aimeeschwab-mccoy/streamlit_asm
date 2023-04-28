import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


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

## Load the small cab rides dataset
url = "https://raw.githubusercontent.com/aimeeschwab-mccoy/streamlit_asm/main/cab_rides_small.csv"
data = pd.read_csv(url)

X = data[['Distance']]
y = data[['Price']]

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)


col01, col02 = st.columns([1, 1])

with col01:

    p = st.number_input("Enter a degree for the polynomial regression model.", min_value=1, max_value=5, value=1)

    polyFeatures = PolynomialFeatures(degree=p, include_bias=False)
    Xp = polyFeatures.fit_transform(X)

    polyModel = LinearRegression()
    polyModel.fit(Xp, y)

with col02:

    fig, ax = plt.subplots()

    p = sns.scatterplot(data=data, x='Distance', y='Price')
    p.set_xlabel('Distance', fontsize=14)
    p.set_ylabel('Price', fontsize=14)
    xDelta = np.linspace(0, 5, 100)
    yDelta = polyModel.predict(polyFeatures.fit_transform(xDelta.reshape(-1, 1)))
    plt.plot(xDelta, yDelta, color='black', linewidth=2)

    st.pyplot(fig)


col11, col12 = st.columns([1, 1])

with col11:

    preds = polyModel.predict(Xp)

    errors = y - preds

    data['Prediction'] = np.ravel(preds)
    data['Error'] = data['Price'] - data['Prediction']

    st.dataframe(data)

with col12:

    bias = data[['Error']].mean()
    variance = data[['Error']].var()

    st.write("Bias:", bias[0].round(2))

    st.write("Variance:", variance[0].round(3))

    showplot = st.checkbox(label="Show plot description?", value=False)

    if showplot:

        st.write("A scatter plot with distance on the horizontal axis ranging from 0 to 5 and price on the vertical axis ranging from 5 to 35. The price and distance of 20 cab rides is shown. A positive trend exists in the scatterplot, and a polynomial model with degree $p$ is overlaid on the plot.")

