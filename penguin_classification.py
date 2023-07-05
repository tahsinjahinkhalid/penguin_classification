# import modules
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# specify streamlit options
st.set_page_config(layout="wide")


def user_input_features():
    island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
    sex = st.sidebar.selectbox("Gender", ('male', 'female'))
    bill_length_mm = st.sidebar.slider(
        'Bill length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.sidebar.slider(
        'Flipper length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.sidebar.slider(
        'Body mass (g)', 2700.0, 6300.0, 4207.0)
    data = {'island': island,
            'culmen_length_mm': bill_length_mm,
            'culmen_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': sex}
    features = pd.DataFrame(data, index=[0])
    return features


# markdown to introduce the webapp
st.write("""
# Web App: Penguin Specifies Classification

## By: Tahsin Jahin Khalid

## About this Web App:

This uses a trained classification model to predict the species of Penguin given specific input parameters or a CSV file upload of specific feature values.

Data has been obtained from the [Palmer Penguins](https://github.com/allisonhorst/palmerpenguins) R library by Allison Horst.
""")

# insert my portfolion link
st.write("[My Portfolio Website](tahsinjahinkhalid.github.io) if you have any job/role offers for me.")

st.sidebar.header("User Input Features")

# show user what a sample uploadable file looks like
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# collect user input features into a dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else: # when we dont have the file
    input_df = user_input_features()

# we input the cleaned CSV file..for the encoding phase
penguins_cleaned = pd.read_csv('data/penguins_cleaned.csv', encoding="utf-8", header=0)
# I had to remove that extra index column that snuck in
penguins = penguins_cleaned.drop(columns=['Unnamed: 0','species'], axis=1)
df = pd.concat([input_df, penguins], axis=0)

encode = ['sex', 'island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

# Selects only the first row (the user input data)
df = df[:1]

st.subheader("User Input Features")

if uploaded_file is not None:
    st.dataframe(df)
else:
    st.write("Awaiting CSV file to be uploaded.")
    st.write("Currently using example input parameters (shown below).")
    st.dataframe(df)

# reads the pickled model
with open("penguin_clf.pkl", "rb") as file:
    load_clf = pickle.load(file)

# apply model to make prediction
prediction = load_clf.predict(df)
prediction_probs = load_clf.predict_proba(df)

st.subheader("Prediction")
penguin_species = np.array(["Adelie", "Chinstrap", "Gentoo"])
st.write(f"**{penguin_species[prediction][0]}**")
st.subheader("Prediction Probability")
st.dataframe(prediction_probs)