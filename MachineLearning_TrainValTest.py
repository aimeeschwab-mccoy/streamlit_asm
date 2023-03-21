import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
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

## Load the wine sample dataset
url = "https://raw.githubusercontent.com/aimeeschwab-mccoy/streamlit_asm/main/wine_sample.csv"
wine = pd.read_csv(url)

# Create integer-valued class
wine['class_int'] = wine['type'].replace(to_replace = ['red','white'], value = [int(0), int(1)])

a,b = st.slider("Use the slider to adjust the proportions allocated to training, validation, and testing.",
    min_value=0.0, max_value=1.0, value=(0.70,0.85), step=0.01)

X = wine[['density', 'alcohol']]
y = wine[['class_int']]

seed = st.number_input("Set a random state value between 0 and 1000.", value=int(123), min_value=int(0), max_value=int(1000))

# Create training/testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-a, random_state=seed)

# Split testing again into validation/testing
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=(1-b)/(1-a),  random_state=seed)

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Fit k-nearest neighbors on k={3, 5, 7, 9}
knnModel = KNeighborsClassifier()
parameters = {'n_neighbors': [3, 5, 7, 9]}
classifier = GridSearchCV(knnModel, parameters, return_train_score=True)
classifier.fit(X_train, np.ravel(y_train))



col01, col02, col03 = st.columns([1, 1, 1])

with col01:

    st.write("Training proportion:", round(a, 2))

    fig, ax = plt.subplots()

    p = sns.scatterplot(x=X_train[:,0], y=X_train[:,1], hue=np.ravel(y_train), 
                    style=np.ravel(y_train))
    p.set_xlabel('Density', fontsize=14)
    p.set_ylabel('Alcohol', fontsize=14)
    L = plt.legend(title="Type")
    L.get_texts()[0].set_text('red')
    L.get_texts()[1].set_text('white')
    plt.title('Training set', fontsize=16)

    st.pyplot(fig)


with col02:

    st.write("Validation proportion:", round(b-a, 2))

    fig, ax = plt.subplots()

    p = sns.scatterplot(x=X_val[:,0], y=X_val[:,1], hue=np.ravel(y_val), 
                    style=np.ravel(y_val))
    p.set_xlabel('Density', fontsize=14)
    p.set_ylabel('Alcohol', fontsize=14)
    L = plt.legend(title="Type")
    L.get_texts()[0].set_text('red')
    L.get_texts()[1].set_text('white')
    plt.title('Validation set', fontsize=16)

    st.pyplot(fig)


with col03:

    st.write("Testing proportion:", round(1-b, 2))

    fig, ax = plt.subplots()

    p = sns.scatterplot(x=X_test[:,0], y=X_test[:,1], hue=np.ravel(y_test), 
                    style=np.ravel(y_test))
    p.set_xlabel('Density', fontsize=14)
    p.set_ylabel('Alcohol', fontsize=14)
    L = plt.legend(title="Type")
    L.get_texts()[0].set_text('red')
    L.get_texts()[1].set_text('white')
    plt.title('Testing set', fontsize=16)

    st.pyplot(fig)


col11, col12, col13 = st.columns([1, 1, 1])

with col11:

    st.write('Fit k-nearest neighbors with k={3, 5, 7, 9} to the training set.')

    train_results = pd.DataFrame({'k': (3, 5, 7, 9), 'Score': classifier.cv_results_["mean_train_score"]})
    st.write(train_results)

with col12:

    st.write('Choose the "best" performing k on the validation set.')

    val_results = pd.DataFrame({'k': (3, 5, 7, 9), 'Score': classifier.cv_results_["mean_test_score"]})
    st.write(val_results)

    bestk_index = val_results['Score'].idxmax()

    st.write("Best k = ", val_results['k'][bestk_index])


with col13:

    st.write('Evaluate the "best" k from validation using the testing set.')

    best_model = classifier.best_estimator_

    st.write("Best k's score:", best_model.score(X_test, y_test).round(4))