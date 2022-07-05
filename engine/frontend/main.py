# frontend/main.py

import requests
import streamlit as st
import pandas as pd


# https://discuss.streamlit.io/t/version-0-64-0-deprecation-warning-for-st-file-uploader-decoding/4465
st.set_option("deprecation.showfileUploaderEncoding", False)

# defines an h1 header
st.title("Machine Learning Pipeline")
st.markdown("This webapp allows users to train a supervised machine learning model that supports modern algorithms such as simple configuration of K-folds, train test split ratio."  \
            " It also supports modern algorithm training such as XGboost and Random Forest, hyperparameters tuning. \
            ")


# displays a file uploader widget
file = st.file_uploader("Choose an CSV File")

# displays a button
if st.button("Train!"):
    if file:
        json_data = {"file": pd.read_csv(file).dropna().head(50).to_dict()}
        res = requests.post(f"http://localhost:8080/", json=json_data)
        res_json = res.json()

        st.text("Accuracy: " + str(res_json['accuracy_score']))
        st.dataframe(data=pd.DataFrame.from_dict(res_json['confusion_matrix']))