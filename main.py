# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Penguins Prediction App
This app predicts the **Palmer Penguins** species!
Data obtained from [palmerpenguins]....
""")

st.sidebar.header('User Input Features')

st.markdown(""""
[Example de CSV input file]
""")


uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgesen'))
        sex = st.sidebar.selectbox('Sex', ('male', 'female'))
        bill_length_mm = st.sidebar.slider('Bill length(mm), 32.1,59.6,43.9')
        bill_depth_mm = st.sidebar.slider('Bill depth(mm), 13.1,21.5,17.2')
        flipper_length_mm = st.sidebar.slider('Flipper length(mm), 172.0,231.0,201.0')
        body_mass_g = st.sidebar.slider('Body mass(g), 2700.0,6300.0,4207.0')
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex
                }
        features = pd.DataFrame(data, index = [0])
        return features
    input_df = user_input_features()

penguins_raw = pd.read_csv('penguins_cleaned.csv')
penguins = penguins_raw.drop(columns = ['species'] )
df = pd.concat([input_df, penguins] , axis=0)

encode = ['sex', 'island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

df = df[:1]

st.subheader('User Input feature')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters')
    st.write(df)

load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)
st.subheader('prediction')
penguins_species = np.array(['Adelie', 'Chinstrap','Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction')
st.write(prediction_proba)