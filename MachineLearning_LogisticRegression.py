import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression


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

## Load the penguins dataset and drop instances with missing values
penguins = sns.load_dataset("penguins").dropna()


# Create binary features for Adelie, Chinstrap, and Gentoo
penguins['Adelie'] = penguins['species'].replace(to_replace = ['Adelie','Chinstrap', 'Gentoo'],value = [int(1), int(0), int(0)])
penguins['Chinstrap'] = penguins['species'].replace(to_replace = ['Adelie','Chinstrap', 'Gentoo'],value = [int(0), int(1), int(0)])
penguins['Gentoo'] = penguins['species'].replace(to_replace = ['Adelie','Chinstrap', 'Gentoo'],value = [int(0), int(0), int(1)])

col1, col2 = st.columns([2, 3])

with col1:

    inputFeature = st.selectbox("Select an input feature",
        ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"])

    outputFeature = st.selectbox("Select a target species",
        ["Adelie", "Chinstrap", "Gentoo"])

    X = penguins[[inputFeature]]
    y = penguins[[outputFeature]]

    showtext = st.checkbox(label="Show description?", value=False)

    if showtext:

        st.write("Scatterplot with input feature on the horizontal axis and target species on the vertical axis. Predicted probabilities from three logistic regression models are added to the scatterplot.")


with col2: 

    # Initialize a logistic regression model
    lm0 = LogisticRegression(penalty='none')
    lm0.fit(X, np.ravel(y))

    lm1 = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
    lm1.fit(X, np.ravel(y))

    # Initialize a logistic regression model
    lm2 = LogisticRegression(penalty='l2', C=0.01)
    lm2.fit(X, np.ravel(y))


    fig, ax = plt.subplots()

    # Plot logistic regression model
    plt.scatter(X, y, color='black')

    xrange = np.linspace(X.min(), X.max(), 10000)
    yprob0 = lm0.predict_proba(xrange.reshape(-1, 1))[:, 1]
    yprob1 = lm1.predict_proba(xrange.reshape(-1, 1))[:, 1]
    yprob2 = lm2.predict_proba(xrange.reshape(-1, 1))[:, 1]

    plt.plot(xrange, yprob0, color='#1f77b4', linewidth=2, linestyle='solid')
    plt.plot(xrange, yprob1, color='#ff7f0e', linewidth=2, linestyle='dotted')
    plt.plot(xrange, yprob2, color='#3ca02c', linewidth=2, linestyle='dashed')
    plt.xlabel(inputFeature, fontsize=14)
    plt.ylabel(outputFeature, fontsize=14)

    st.pyplot(fig)


col21, col22, col23 = st.columns([1, 1, 1])

with col21: 

    st.markdown("No regularization (<span style='color:#1f77b4'>blue solid line</span>)", unsafe_allow_html=True)

    st.write("Intercept:", str(round(lm0.intercept_[0], 3)))
    st.write("Slope:", str(round(lm0.coef_[0,0], 3)))
    st.write("Score:", str(round(lm0.score(X, y), 4)))

with col22: 

    st.markdown("L1 regularization (<span style='color:#ff7f0e'>orange dotted line</span>)", unsafe_allow_html=True)

    st.write("Intercept:", str(round(lm1.intercept_[0], 3)))
    st.write("Slope:", str(round(lm1.coef_[0,0], 3)))
    st.write("Score:", str(round(lm1.score(X, y), 4)))

with col23: 

    st.markdown("L2 regularization (<span style='color:#3ca02c'>green dashed line</span>)", unsafe_allow_html=True)

    st.write("Intercept:", str(round(lm2.intercept_[0], 3)))
    st.write("Slope:", str(round(lm2.coef_[0,0], 3)))
    st.write("Score:", str(round(lm2.score(X, y), 4)))
