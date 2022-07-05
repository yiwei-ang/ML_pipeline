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
if st.button("Start train!"):
    if file:
        json_data = {"file": pd.read_csv(file).to_json(orient='records')}
        res = requests.post(f"http://0.0.0.0:8080/", json=json_data)
        res_json = res.json()
        df_output = pd.read_json(res_json)

        st.text("Accuracy:", df_output.iloc[0])
        st.dataframe(data=df_output)
        #
        # fig = plt.figure(figsize=(10, 4))
        # plt.title('Frequency of tweets positive score')
        # sns.histplot(df_sentiment, x='sentiment_score_positive', bins=[0, 0.2, 0.4, 0.6, 0.8, 1]);
        # st.pyplot(fig)
        #
        # fig = plt.figure(figsize=(10, 4))
        # plt.title('Frequency of tweets negative score')
        # sns.histplot(df_sentiment, x='sentiment_score_negative', bins=[0, 0.2, 0.4, 0.6, 0.8, 1]);
        # st.pyplot(fig)
        #
        # fig = plt.figure(figsize=(10, 4))
        # plt.title('Frequency of tweets mixed score')
        # sns.histplot(df_sentiment, x='sentiment_score_mixed', bins=[0, 0.2, 0.4, 0.6, 0.8, 1]);
        # st.pyplot(fig)