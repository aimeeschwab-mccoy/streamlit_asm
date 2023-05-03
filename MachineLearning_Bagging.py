import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics


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
url = "https://raw.githubusercontent.com/aimeeschwab-mccoy/streamlit_asm/main/friedman1.csv"
data = pd.read_csv(url).dropna()

# Satisfaction = 1 for satisfied customers and = 0 for unsatisfied customers
#data['Satisfaction'] = label_binarize(data['Satisfaction'],  classes=['Satisfied', 'Unsatisfied'])

#labelBinary = LabelBinarizer()
# Gender == 1 if Male
#data['Gender'] = labelBinary.fit_transform(data['Gender'])
# Customer == 1 if Loyal
#data['Customer'] = labelBinary.fit_transform(data['Customer'])
# TravelType == 1 if Personal
#data['TravelType'] = labelBinary.fit_transform(data['TravelType'])
# Class == 1 if Business
#data['Class'] = labelBinary.fit_transform(data['Class'])

col01, col02 = st.columns([1, 1])

with col01:

    max_depth = st.number_input("Decision tree max depth", min_value=1, max_value=5)
    n = st.number_input("Ensemble size", min_value=10, max_value=100, value=10)

with col02: 

#    feature1 = st.selectbox(
#        "Feature 1",
#        [
#            'Gender', 'Customer','Age', 'TravelType','Class','Distance', 'InflightWifi', 'ConvenientTime','OnlineBooking','GateLocation',
#            'FoodDrink','OnlineBoarding','SeatComfort', 'InflightEntertainment','OnboardService','LegRoom', 'Baggage','CheckinService',
#            'InflightService','Cleanliness','DepartureDelay','ArrivalDelay'
#        ]
#    )

#   feature2 = st.selectbox(
#        "Feature 2",
#        [
#            'Gender', 'Customer','Age', 'TravelType','Class','Distance', 'InflightWifi', 'ConvenientTime','OnlineBooking','GateLocation',
#            'FoodDrink','OnlineBoarding','SeatComfort', 'InflightEntertainment','OnboardService','LegRoom', 'Baggage','CheckinService',
#            'InflightService','Cleanliness','DepartureDelay','ArrivalDelay'
#        ]
#    )

    X = data[["x.1", "x.2", "x.3", "x.4", "x.5"]]
    y = data[["y"]]

    st.write("Decision tree results")

    # Initialize the model
    dtModel = DecisionTreeRegressor(random_state=69, max_depth=max_depth)
    # Fit the model
    dtModel = dtModel.fit(X,y)

    fig, ax = plt.subplots()

    score1 = dtModel.score(X, y)

    st.write("Score = ", round(score1, 4))

    predvariance1 = dtModel.predict(X).var()

    #st.write(dtModel.predict(X))

    st.write("Prediction variance = ", round(predvariance1, 4))

    errors1 = dtModel.predict(X) - np.ravel(y)
    errorvariance1 = errors1.var()

    st.write("Error variance = ", round(errorvariance1, 4))


col11, col12 = st.columns([1, 1])

with col11:

    dataframe = st.checkbox('Show dataframe?')

    if dataframe: 

        st.dataframe(data[["x.1", "x.2", "x.3", "x.4", "x.5", "y"]].style.format("{:.1f}"))

with col12: 

    st.write("Bagged ensemble results")

    baggingModel = BaggingRegressor(random_state=69, n_estimators=n)
    baggingModel.fit(X, np.ravel(y))
    

    score2 = baggingModel.score(X, y)

    st.write("Score = ", round(score2, 4))

    predvariance2 = baggingModel.predict(X).var()

    st.write("Prediction variance = ", round(predvariance2, 4))

    errors2 = baggingModel.predict(X) - np.ravel(y)
    errorvariance2 = errors2.var()

    st.write("Error variance = ", round(errorvariance2, 4))

    plot = st.checkbox('Show predicted values?')

    if plot:

        p = sns.scatterplot(x = dtModel.predict(X), y=np.ravel(y), label='Decision tree')
        p = sns.scatterplot(x = baggingModel.predict(X), y=np.ravel(y), label='Bagging ensemble')
        p.set_xlabel('Predicted y', fontsize=14)
        p.set_ylabel('Actual y', fontsize=14)
        st.pyplot(fig)

        st.write("Scatter plot with predicted y on the horizontal axis and actual y on the vertical axis. A linear relationship exists between the predicted values and actual values using the bagging ensemble. Increasing the depth of the decision tree increases the number of possible values for predicted y.")