import streamlit as st
from model import ScoringModel
from PIL import Image

scoring_model = ScoringModel()

image = Image.open('banner.png')
st.image(image)

st.title('Bank transactional scoring model demo')

st.write("""
Instructions:
Sample random data by selecting a specific method,
then apllication will render client's data and score using LGBM Model
""")

st.title('Sample data')

st.write("""
Sample random features from random loaner:
""")

btn_random = st.button('Predict random')

st.write("""
Sample data by class:
""")

btn_class_1 = st.button('Predict defaulter')

btn_class_0 = st.button('Predict non-defaulter')

st.title('Inspect input data (loaner transactional features) and model score')

def predict_on_click(dataframe):
    st.dataframe(dataframe)
    st.write('Client app_id:', str(dataframe.app_id.values[0]))
    st.write('Client defaulted (yes/no):', str(dataframe.flag.values[0]))
    score = scoring_model.predict(dataframe[scoring_model.features].values)
    st.write('Predicted score: ', f'{score}')

if btn_random:
    dataframe = scoring_model.data.sample(n=1)
    predict_on_click(dataframe)

elif btn_class_1:
    dataframe = scoring_model.data[scoring_model.data.flag == 1].sample(n=1)
    predict_on_click(dataframe)

elif btn_class_0:
    dataframe = scoring_model.data[scoring_model.data.flag == 0].sample(n=1)
    predict_on_click(dataframe)

else:
    st.write('First sample data for model to predict, using buttons')