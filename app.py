import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('logistic_regression.pkl')

# Title of the Streamlit app
st.title('Titanic Survival Prediction')

# User inputs
pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.slider('Age', 0, 100, 25)
sibsp = st.number_input('Number of Siblings/Spouses Aboard', 0, 10, 0)
parch = st.number_input('Number of Parents/Children Aboard', 0, 10, 0)
fare = st.slider('Fare', 0.0, 500.0, 30.0)
embarked = st.selectbox('Port of Embarkation', ['C', 'Q', 'S'])

# Convert categorical inputs
sex = 1 if sex == 'male' else 0
embarked_map = {'C': 0, 'Q': 1, 'S': 2}
embarked = embarked_map[embarked]

# Convert user inputs into a format suitable for prediction
inputs = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]],
                      columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

# Perform prediction
prediction = model.predict(inputs)[0]

# Display the prediction result
if prediction == 1:
    st.write('The passenger is likely to survive.')
else:
    st.write('The passenger is unlikely to survive.')
