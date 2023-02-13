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

a,b = st.slider("Use the slider to adjust the prior probabilties",
    min_value=0.0, max_value=1.0, value=(0.33,0.67), step=0.01)

col01, col02, col03 = st.columns([1, 1, 1])

with col01:

    st.write("P(Adelie):", round(a, 2))

with col02:

    st.write("P(Chinstrap):", round(b-a, 2))

with col03:

    st.write("P(Gentoo):", round(1-b, 2))

col1, col2 = st.columns([1,1])

with col1:

    X = penguins[['body_mass_g']]
    y = penguins[['species_int']]
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
    plt.xlabel('Body mass (g)', fontsize=14)
    plt.ylabel('Probability of each species', fontsize=14)

    st.pyplot(fig)

    showtext = st.checkbox(label="Show probability curve description?", value=False)

    if showtext:

        '''
        Left: Predicted probabilities for Adelie (blue solid line), Chinstrap (orange dashed line), 
        and Gentoo (green dotted line) penguins using Gaussian naive Bayes. Changing the prior probabilities 
        adjusts the fitted probability curves. 
        '''


with col2: 


    fig, ax = plt.subplots()
    contourf_kwargs = {'alpha': 0.2}

    plot_decision_regions(X.to_numpy(), np.ravel(y), clf=NBModel, contourf_kwargs=contourf_kwargs)
    L = plt.legend()
    L.get_texts()[0].set_text('Adelie')
    L.get_texts()[1].set_text('Chinstrap')
    L.get_texts()[2].set_text('Gentoo')

    plt.xlabel('Body mass (g)', fontsize=14)
    plt.ylabel('\n     ', fontsize=14)

    st.pyplot(fig)

    showtext2 = st.checkbox(label="Show decision boundary description?", value=False)

    if showtext2:

        '''
        Right: Decision boundary plot based on bill length (mm). Decision boundary cutoffs correspond to 
        values of body mass where the predicted probabilities intersect. The species with the highest probability
        curve at a given value of body mass is the predicted class. Classes with greater prior probability have larger 
        regions in the decision boundary plot.
        '''