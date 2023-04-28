import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import resample

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

## Load the airline satisfaction dataset
url = "https://raw.githubusercontent.com/aimeeschwab-mccoy/streamlit_asm/main/airline_satisfaction.csv"
data = pd.read_csv(url).dropna()

X = data[['Distance']]

col01, col02, col03 = st.columns([1, 1, 2])

with col01:

    rand = st.number_input("Enter a random state:", min_value=1, max_value=100, value=1)
    n = st.number_input("Enter a sample size:", min_value=10, max_value=100, value=50)

    Xsample = X.sample(random_state=rand, n=n)


with col02: 

    st.write("Descriptive statistics for original sample")

    st.dataframe(Xsample.describe().style.format("{:.1f}"))
    #st.write(Xsample.describe())

with col03:

    fig, ax = plt.subplots()

    p = sns.histplot(data=Xsample, x='Distance', bins=10)
    p.set_xlabel('Distance', fontsize=14)
    p.set_ylabel('Count', fontsize=14)
    p.set_title('Original sample', fontsize=16)
    p.set_xlim(0, 4000)
    plt.axvline(Xsample['Distance'].mean(), color='#ff7f0e', linewidth=3)
    st.pyplot(fig)


col11, col12, col13 = st.columns([1, 1, 2])

with col11:

    nBoot = st.number_input("Enter the number of bootstrap samples:", min_value=10, max_value=100, value=50)

    bootstrapMeans = []
    bootstrapSDs = []

    for i in range(0, nBoot):
        # Create the bootstrap sample
        bootstrapSample = resample(Xsample, replace=True)

        # Calculate the bootstrap sample means
        mean = bootstrapSample.mean()
        bootstrapMeans.append(mean)

        # Calculate the bootstrap sample SDs
        sd = bootstrapSample.std()
        bootstrapSDs.append(sd)

with col12: 

    st.write("Descriptive statistics for bootstrap sample means")

    st.dataframe(pd.DataFrame(bootstrapMeans).describe().style.format("{:.1f}"))
    #st.write(round(pd.DataFrame(bootstrapMeans).describe(), 1))

with col13:

    fig, ax = plt.subplots()

    p = sns.histplot(pd.DataFrame(bootstrapMeans), bins=5)
    p.set_xlabel('Mean distance', fontsize=14)
    p.set_ylabel('Count', fontsize=14)
    p.set_title('Bootstrap samples', fontsize=16)
    plt.legend([],[], frameon=False)
    plt.axvline(Xsample['Distance'].mean(), color='#ff7f0e', linewidth=3)
    p.set_xlim(0, 4000)
    st.pyplot(fig)