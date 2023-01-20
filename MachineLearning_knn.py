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

penguins = sns.load_dataset("penguins")
penguins = penguins.replace(to_replace = ['Adelie','Chinstrap', 'Gentoo'],value = [int(0), int(1), int(2)])

penguins20 = penguins.sample(n=20, random_state=2).sort_values(by='species')
penguins50 = penguins.dropna().sample(n=50, random_state=2).sort_values(by='species')

# Define input and output features
X20 = penguins20[['bill_length_mm', 'bill_depth_mm']]
y20 = penguins20[['species']]

# Scale the input features
scaler = StandardScaler()
X_scaled20 = scaler.fit_transform(X20)

# Convert scaled inputs back to a dataframe
X20 = pd.DataFrame(X_scaled20, index=X20.index, columns=X20.columns)

# Define input and output features
X = penguins50[['bill_length_mm', 'bill_depth_mm']]
y = penguins50[['species']]

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled inputs back to a dataframe
X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)


col1, col2 = st.columns([1,1])

with col1:

    st.write("Sample size: $n=20$")
    st.write("Number of neighbors: $k$")

    k = st.slider(label="Select a value between 1 and 20.", min_value=1, max_value=20, value=5, step=1)


    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X20, np.ravel(y20['species']))

    fig, ax = plt.subplots()
    contourf_kwargs = {'alpha': 0.2}

    p = plot_decision_regions(X_scaled20, np.ravel(y20), clf=knn, contourf_kwargs=contourf_kwargs)
    p.set_title('Decision boundary')
    L = plt.legend()
    L.get_texts()[0].set_text('Adelie')
    L.get_texts()[1].set_text('Chinstrap')
    L.get_texts()[2].set_text('Gentoo')

    st.pyplot(fig)

   
#    check = st.checkbox("Display frequency table")
#
#    if check:
#        summary = bobross.groupby(categorical).size().to_frame()
#        st.dataframe(summary)

with col2:

    st.write("Sample size: $n=50$")
    st.write("Number of neighbors: $k$")

    k2 = st.slider(label="Select a value between 1 and 20.", min_value=1, max_value=20, value=5, step=1, key=1)


    knn = KNeighborsClassifier(n_neighbors=k2)
    knn.fit(X, np.ravel(y['species']))

    fig, ax = plt.subplots()
    contourf_kwargs = {'alpha': 0.2}

    p = plot_decision_regions(X_scaled, np.ravel(y), clf=knn, contourf_kwargs=contourf_kwargs)
    p.set_title('Decision boundary')
    L = plt.legend()
    L.get_texts()[0].set_text('Adelie')
    L.get_texts()[1].set_text('Chinstrap')
    L.get_texts()[2].set_text('Gentoo')

    st.pyplot(fig)

