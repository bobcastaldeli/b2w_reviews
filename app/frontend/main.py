"""This is the frontend of the app."""


import requests
import pandas as pd
import streamlit as st


st.title("Classificador de sentimentos dos reviews do clientes da B2W")
st.write(
    """
    Este aplicativo classifica o sentimento dos reviews dos clientes da B2W,
    como positivo ou negativo com base no texto do review que o cliente criou
    sobre um produto que o mesmo comprou.
    """
)


input_text = st.text_area("Enter your text here")


if st.button("Predict"):
    data = {"review": input_text}

    response = requests.post("http://127.0.0.1:8000/predict", json=data)

    prediction = response.json()["prediction"]
    probability = response.json()["probability"]

    prob_df = pd.DataFrame(
        {"class": ["Negativo", "Positivo"], "probability": probability}
    ).round(2)

    if prediction == 0:
        st.write("Review negativo")
    else:
        st.write("Review positivo")

    st.bar_chart(prob_df, x="class", y="probability")
