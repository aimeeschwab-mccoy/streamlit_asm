import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from mlxtend.plotting import plot_decision_regions

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

# Create integer-valued species
penguins['species_int'] = penguins['species'].replace(to_replace = ['Adelie','Chinstrap', 'Gentoo'],
                                                      value = [int(0), int(1), int(2)])

col1, col2 = st.columns([2, 3])

with col1:

    a,b = st.slider("Use the slider to adjust the prior probabilties",
        min_value=0.0, max_value=1.0, value=(0.333,0.667), step=0.001)

    st.write("P(Adelie):", round(a, 3))
    st.write("P(Chinstrap):", round(b-a, 3))
    st.write("P(Gentoo):", round(1-b, 3))

    X = penguins[['bill_length_mm']]
    y = penguins[['species_int']]

    showtext = st.checkbox(label="Show description?", value=False)

    if showtext:

        '''
        Top: Predicted probabilities for Adelie (blue solid line), Chinstrap (orange dashed line), 
        and Gentoo (green dotted line) penguins using Gaussian naive Bayes. Changing the prior probabilities 
        adjusts the fitted probability curves. 
        
        Bottom: Decision boundary plot based on bill length (mm). Decision boundary cutoffs correspond to 
        values of bill length where the predicted probabilities intersect. The species with the highest probability
        curve at a given value of bill length is the predicted class. Classes with greater prior probability have larger 
        regions in the decision boundary plot.
        '''


with col2: 

    # Initialize and fit Gaussian naive Bayes with priors
    NBModel = GaussianNB(priors = [a, b-a, 1-b])
    NBModel.fit(X, np.ravel(y))

    fig, ax = plt.subplots()

    # Plot Gaussian naive Bayes model
    xrange = np.linspace(X.min(), X.max(), 10000)
    probAdelie = NBModel.predict_proba(xrange.reshape(-1, 1))[:, 0]
    probChinstrap = NBModel.predict_proba(xrange.reshape(-1, 1))[:, 1]
    probGentoo = NBModel.predict_proba(xrange.reshape(-1, 1))[:, 2]
 
    plt.plot(xrange, probAdelie, color='#1f77b4', linewidth=2, linestyle='-')
    plt.plot(xrange, probChinstrap, color='#ff7f0e', linewidth=2, linestyle='--')
    plt.plot(xrange, probGentoo, color='#3ca02c', linewidth=2, linestyle=':')
    plt.xlabel('Bill length (mm)', fontsize=14)
    plt.ylabel('Probability of each species', fontsize=14)

    st.pyplot(fig)

    fig, ax = plt.subplots()
    contourf_kwargs = {'alpha': 0.2}

    plot_decision_regions(X.to_numpy(), np.ravel(y), clf=NBModel, contourf_kwargs=contourf_kwargs)
    L = plt.legend()
    L.get_texts()[0].set_text('Adelie')
    L.get_texts()[1].set_text('Chinstrap')
    L.get_texts()[2].set_text('Gentoo')

    plt.xlabel('Bill length (mm)', fontsize=14)
    plt.ylabel('\n     ', fontsize=14)

    st.pyplot(fig)