import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

url = "https://raw.githubusercontent.com/anshupandey/Machine-Learning-Using-Python/master/datasets/heart.csv"
df = pd.read_csv(url)

X = df.drop('target', axis=1)
y = df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier()
model.fit(X_scaled, y)

st.title("❤️ Heart Disease Prediction App")

def user_input():
    age = st.slider('Age', 29, 77, 55)
    sex = st.selectbox('Sex', [0, 1])
    cp = st.slider('Chest pain type', 0, 3, 1)
    trestbps = st.slider('Resting Blood Pressure', 94, 200, 120)
    chol = st.slider('Cholesterol', 126, 564, 240)
    fbs = st.selectbox('Fasting Blood Sugar > 120?', [0, 1])
    restecg = st.slider('Resting ECG', 0, 2, 1)
    thalach = st.slider('Max Heart Rate', 71, 202, 150)
    exang = st.selectbox('Exercise Induced Angina', [0, 1])
    oldpeak = st.slider('Oldpeak', 0.0, 6.2, 1.0)
    slope = st.slider('Slope', 0, 2, 1)
    ca = st.slider('CA (number of vessels)', 0, 4, 0)
    thal = st.slider('Thal', 0, 3, 2)

    data = np.array([age, sex, cp, trestbps, chol, fbs, restecg,
                     thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    return scaler.transform(data)

input_data = user_input()

if st.button('Predict'):
    result = model.predict(input_data)
    st.success('Heart Disease Risk: {}'.format('Yes' if result[0] == 1 else 'No'))
